# Copyright Author: Dr Tang Tiong Yew
"""
Ant Colony Optimisation (ACO) for Robot Swarm Route Finding in NVIDIA Isaac Sim
================================================================================
This script demonstrates multi-robot route finding and swarm path planning
using Ant Colony Optimisation on a map of connected waypoints (Romania Roadmap).

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics & visual simulation):
   Run with Isaac Sim's standalone python:
   `isaac-sim.standalone.bat python src/files/isaac_aco_route.py`
   OR `python.bat src/files/isaac_aco_route.py`

2. Matplotlib Fallback Mode (Standard Python 2D/3D graph visualization):
   `python src/files/isaac_aco_route.py`
"""

import sys
import time
import numpy as np

# Try importing NVIDIA Isaac Sim modules
HAS_ISAAC_SIM = False
try:
    from isaacsim import SimulationApp
    HAS_ISAAC_SIM = True
except ImportError:
    try:
        from omni.isaac.kit import SimulationApp
        HAS_ISAAC_SIM = True
    except ImportError:
        HAS_ISAAC_SIM = False


# =====================================================================
# Romania Map Topology & ACO Data Structure
# =====================================================================
LOCATION_LIST = [  # [x, y, name]
    [75, 125, 'Arad'],
    [100, 75, 'Zerind'],
    [125, 25, 'Oradea'],
    [265, 175, 'Sibiu'],
    [425, 175, 'Fagaras'],
    [320, 230, 'Rimnicu Vilcea'],
    [475, 310, 'Pitesti'],
    [350, 465, 'Craiova'],
    [185, 450, 'Drobeta'],
    [190, 390, 'Mehadia'],
    [185, 335, 'Lugoj'],
    [85, 280, 'Timisoara'],
    [640, 390, 'Bucharest'],
    [575, 485, 'Giurgiu'],
    [745, 340, 'Urziceni'],
    [875, 340, 'Hirsova'],
    [935, 440, 'Eforie'],
    [850, 225, 'Vaslui'],
    [760, 120, 'Iasi'],
    [625, 60, 'Neamt']
]

STEP_COST = [
    ['Arad', 'Zerind', 75],
    ['Zerind', 'Oradea', 71],
    ['Oradea', 'Sibiu', 151],
    ['Sibiu', 'Arad', 140],
    ['Sibiu', 'Fagaras', 99],
    ['Sibiu', 'Rimnicu Vilcea', 80],
    ['Fagaras', 'Bucharest', 211],
    ['Bucharest', 'Giurgiu', 90],
    ['Bucharest', 'Pitesti', 101],
    ['Pitesti', 'Rimnicu Vilcea', 97],
    ['Rimnicu Vilcea', 'Craiova', 146],
    ['Craiova', 'Pitesti', 138],
    ['Craiova', 'Drobeta', 120],
    ['Drobeta', 'Mehadia', 75],
    ['Mehadia', 'Lugoj', 70],
    ['Lugoj', 'Timisoara', 111],
    ['Arad', 'Timisoara', 118],
    ['Bucharest', 'Urziceni', 85],
    ['Urziceni', 'Vaslui', 142],
    ['Vaslui', 'Iasi', 92],
    ['Iasi', 'Neamt', 87],
    ['Urziceni', 'Hirsova', 98],
    ['Hirsova', 'Eforie', 86]
]


class CityNode:
    """Represents a city node in the road network."""
    def __init__(self, name, x, y):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.roads = []

    def add_road(self, road):
        if road not in self.roads:
            self.roads.append(road)


class RoadEdge:
    """Represents a connected road between two cities with distance cost & pheromone."""
    def __init__(self, city1, city2, cost, initial_pheromone=0.05):
        self.city1 = city1
        self.city2 = city2
        self.cost = float(cost)
        self.pheromone = float(initial_pheromone)

    def get_other_city(self, city):
        return self.city2 if city == self.city1 else self.city1

    def evaporate(self, rho):
        self.pheromone = (1.0 - rho) * self.pheromone

    def deposit(self, amount):
        self.pheromone += amount


class RobotAnt:
    """Simulated Ant Agent navigating through the graph."""
    def __init__(self, start_city):
        self.start_city = start_city
        self.current_city = start_city
        self.visited_cities = [start_city]
        self.path_roads = []

    def reset(self):
        self.current_city = self.start_city
        self.visited_cities = [self.start_city]
        self.path_roads = []

    def select_next_road(self, destination, alpha=1.0, beta=2.0):
        if self.current_city == destination:
            return None

        valid_roads = []
        probabilities = []

        for road in self.current_city.roads:
            next_city = road.get_other_city(self.current_city)
            if next_city not in self.visited_cities:
                valid_roads.append(road)
                # ACO transition rule: P ~ (tau^alpha) * (eta^beta)
                tau = road.pheromone ** alpha
                eta = (1.0 / road.cost) ** beta
                probabilities.append(tau * eta)

        if not valid_roads:
            return None  # Dead end

        prob_sum = sum(probabilities)
        if prob_sum == 0:
            probs = [1.0 / len(valid_roads)] * len(valid_roads)
        else:
            probs = [p / prob_sum for p in probabilities]

        chosen_road = np.random.choice(valid_roads, p=probs)
        return chosen_road

    def build_route(self, destination, alpha=1.0, beta=2.0, max_steps=30):
        self.reset()
        steps = 0
        while self.current_city != destination and steps < max_steps:
            road = self.select_next_road(destination, alpha, beta)
            if road is None:
                break  # Stuck or reached end
            next_city = road.get_other_city(self.current_city)
            self.path_roads.append(road)
            self.visited_cities.append(next_city)
            self.current_city = next_city
            steps += 1

        return self.current_city == destination

    def get_route_cost(self):
        return sum(road.cost for road in self.path_roads)


def build_network():
    cities = {name: CityNode(name, x, y) for x, y, name in LOCATION_LIST}
    roads = []
    for city1_name, city2_name, cost in STEP_COST:
        c1 = cities[city1_name]
        c2 = cities[city2_name]
        r = RoadEdge(c1, c2, cost)
        c1.add_road(r)
        c2.add_road(r)
        roads.append(r)
    return cities, roads


# =====================================================================
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_aco(n_ants=10, max_iterations=50, alpha=1.0, beta=2.0, rho=0.1):
    """Executes ACO Swarm Route Finding inside NVIDIA Isaac Sim photorealistic environment."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere, VisualSphere, VisualCuboid

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    cities, roads = build_network()
    origin = cities['Arad']
    destination = cities['Bucharest']

    # Scale 2D coordinates to 3D Isaac Sim world coords (-10m to 10m range)
    def to_isaac_coord(city):
        # Map x: [75, 935] -> [-8.0, 8.0], y: [25, 485] -> [-8.0, 8.0]
        x_m = -8.0 + (city.x - 75.0) / (935.0 - 75.0) * 16.0
        y_m = 8.0 - (city.y - 25.0) / (485.0 - 25.0) * 16.0
        return np.array([x_m, y_m, 0.2])

    # Spawn city node markers in Isaac Sim
    for name, city in cities.items():
        pos = to_isaac_coord(city)
        color = np.array([0.1, 0.8, 0.1]) if name == 'Arad' else (
            np.array([0.9, 0.1, 0.1]) if name == 'Bucharest' else np.array([0.6, 0.6, 0.6])
        )
        radius = 0.5 if name in ['Arad', 'Bucharest'] else 0.35
        world.scene.add(
            VisualSphere(
                prim_path=f"/World/Cities/{name}",
                name=f"city_{name}",
                position=pos,
                radius=radius,
                color=color
            )
        )

    # Create physical ant robots
    ant_prims = []
    for i in range(n_ants):
        start_pos = to_isaac_coord(origin) + np.random.uniform(-0.2, 0.2, size=3)
        start_pos[2] = 0.25
        ant_prim = world.scene.add(
            DynamicSphere(
                prim_path=f"/World/Ants/Ant_{i}",
                name=f"ant_robot_{i}",
                position=start_pos,
                radius=0.2,
                color=np.array([0.9, 0.6, 0.1])  # Amber color for ants
            )
        )
        ant_prims.append(ant_prim)

    ants = [RobotAnt(origin) for _ in range(n_ants)]
    ant_wp_indices = [1] * n_ants  # Track target waypoint index along best route for each ant robot
    world.reset()

    print("=======================================================")
    print(" Running NVIDIA Isaac Sim Robot Swarm ACO Simulation   ")
    print("=======================================================")

    best_route_names = None
    best_route_cost = float('inf')

    for iteration in range(max_iterations):
        if not simulation_app.is_running():
            break

        world.step(render=True)

        # 1. Build route for all ant robots
        successful_ants = []
        for ant in ants:
            reached = ant.build_route(destination, alpha=alpha, beta=beta)
            if reached:
                successful_ants.append(ant)
                cost = ant.get_route_cost()
                if cost < best_route_cost:
                    best_route_cost = cost
                    best_route_names = [c.name for c in ant.visited_cities]

        # 2. Pheromone Evaporation
        for road in roads:
            road.evaporate(rho)

        # 3. Pheromone Deposition
        for ant in successful_ants:
            cost = ant.get_route_cost()
            delta_tau = 100.0 / cost
            for road in ant.path_roads:
                road.deposit(delta_tau)

        # 4. Move Isaac Sim Robot Prims along the best path for visualization
        if best_route_names and len(best_route_names) > 1:
            for idx, ant_prim in enumerate(ant_prims):
                wp_idx = ant_wp_indices[idx]
                target_city_name = best_route_names[min(wp_idx, len(best_route_names) - 1)]
                target_city = cities[target_city_name]
                target_pos = to_isaac_coord(target_city)
                current_pos, _ = ant_prim.get_world_pose()
                direction = target_pos[:2] - current_pos[:2]
                dist = np.linalg.norm(direction)

                if dist < 0.3:
                    if wp_idx + 1 < len(best_route_names):
                        ant_wp_indices[idx] += 1
                    else:
                        # Reached destination city! Reset position back to origin to loop route
                        reset_pos = to_isaac_coord(origin) + np.random.uniform(-0.2, 0.2, size=3)
                        reset_pos[2] = 0.25
                        ant_prim.set_world_pose(position=reset_pos)
                        ant_wp_indices[idx] = 1
                else:
                    vel = (direction / dist) * 1.5
                    ant_prim.set_linear_velocity(np.array([vel[0], vel[1], 0.0]))

        if iteration % 5 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration:02d} | Successful Ants: {len(successful_ants)}/{n_ants} | "
                  f"Best Route Cost: {best_route_cost:.1f} km | Route: {' -> '.join(best_route_names or [])}")

    print(f"\n=======================================================")
    print(f" ACO Convergence Completed!")
    print(f" Optimal Route Cost: {best_route_cost:.1f} km")
    print(f" Optimal Path: {' -> '.join(best_route_names or [])}")
    print("=======================================================")

    simulation_app.close()


# =====================================================================
# 2. Standalone Matplotlib ACO Fallback Simulator
# =====================================================================
def run_matplotlib_fallback_aco(n_ants=10, max_iterations=50, alpha=1.0, beta=2.0, rho=0.1):
    """Fallback visualizer for standard Python environments without NVIDIA Isaac Sim."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib is required to run the fallback visualization.")
        print("    Install it via: pip install matplotlib")
        return

    print("=======================================================")
    print(" [Fallback Mode] Romania Route Finding ACO Simulation ")
    print(" (NVIDIA Isaac Sim module not detected in current Python)")
    print("=======================================================")

    cities, roads = build_network()
    origin = cities['Arad']
    destination = cities['Bucharest']

    ants = [RobotAnt(origin) for _ in range(n_ants)]

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("NVIDIA Isaac Sim Robot Route ACO - Romania Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Plot cities
    for name, city in cities.items():
        color = 'green' if name == 'Arad' else ('red' if name == 'Bucharest' else 'gray')
        size = 120 if name in ['Arad', 'Bucharest'] else 60
        ax.scatter(city.x, city.y, c=color, s=size, zorder=4)
        ax.text(city.x + 8, city.y - 8, name, fontsize=8, fontweight='bold', zorder=5)

    best_route_names = None
    best_route_cost = float('inf')
    road_lines = []
    best_path_lines = []

    for iteration in range(max_iterations):
        # Remove previous frame's road lines
        for line in road_lines + best_path_lines:
            line.remove()
        road_lines.clear()
        best_path_lines.clear()

        ax.set_title(f"Ant Colony Optimisation Robot Route | Iteration {iteration+1}/{max_iterations}")

        # 1. Ants build routes
        successful_ants = []
        for ant in ants:
            reached = ant.build_route(destination, alpha=alpha, beta=beta)
            if reached:
                successful_ants.append(ant)
                cost = ant.get_route_cost()
                if cost < best_route_cost:
                    best_route_cost = cost
                    best_route_names = [c.name for c in ant.visited_cities]

        # 2. Evaporate pheromones
        for road in roads:
            road.evaporate(rho)

        # 3. Deposit pheromones
        for ant in successful_ants:
            cost = ant.get_route_cost()
            delta_tau = 100.0 / cost
            for road in ant.path_roads:
                road.deposit(delta_tau)

        # Draw roads with thickness proportional to pheromone level
        for road in roads:
            c1, c2 = road.city1, road.city2
            lw = min(1.0 + road.pheromone * 5.0, 6.0)
            alpha_val = min(0.2 + road.pheromone * 2.0, 1.0)
            line, = ax.plot([c1.x, c2.x], [c1.y, c2.y], 'b-', linewidth=lw, alpha=alpha_val, zorder=2)
            road_lines.append(line)

        # Draw best route highlighted in red
        if best_route_names:
            for i in range(len(best_route_names) - 1):
                c1 = cities[best_route_names[i]]
                c2 = cities[best_route_names[i+1]]
                line, = ax.plot([c1.x, c2.x], [c1.y, c2.y], 'r--', linewidth=2.5, zorder=3)
                best_path_lines.append(line)

        try:
            plt.draw()
            plt.pause(0.05)
        except Exception:
            break

        if iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration+1:02d} | Successful Ants: {len(successful_ants)}/{n_ants} | "
                  f"Best Cost: {best_route_cost:.1f} km | Path: {' -> '.join(best_route_names or [])}")

    print(f"\nConvergence achieved! Best Route Cost = {best_route_cost:.1f} km")
    print(f"Path: {' -> '.join(best_route_names or [])}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    if HAS_ISAAC_SIM:
        run_isaac_sim_aco()
    else:
        run_matplotlib_fallback_aco()
