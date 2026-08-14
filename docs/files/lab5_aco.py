# Copyright Author: Dr Tang Tiong Yew
"""
Lab 5: Evolutionary Computation (Ant Colony Optimisation)
==========================================================
This script provides the complete solution for Lab 5 exercises, including:
1. Romania road network graph representation (`City`, `Road`, `Ant` classes)
2. Ant path construction without revisiting cities
3. Pheromone evaporation and deposition mechanisms
4. Termination based on path dominance threshold (>=90%)
5. Matplotlib graph visualization with dynamic line thickness for pheromones
6. Parameters for pheromone, heuristic, evaporation, and swarm size experiments

Execution Mode:
`python3 src/files/lab5_aco.py`
"""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


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


class City:
    def __init__(self, name):
        self.name = name
        self.roads = []
        self.coordinates = []

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def add_road(self, road):
        if road not in self.roads:
            self.roads.append(road)


class Road:
    def __init__(self, connected_cities, cost, pheromone=0.01):
        self.connected_cities = connected_cities
        self.cost = float(cost)
        self.pheromone = float(pheromone)

    def set_pheromone(self, pheromone):
        self.pheromone = float(pheromone)

    def evaporate_pheromone(self, rho):
        self.pheromone = (1.0 - rho) * self.pheromone

    def deposit_pheromone(self, ants, deposit_mult=1.0, cost_power=1.0):
        for ant in ants:
            if self in ant.path:
                length = ant.get_path_length()
                if length > 0:
                    self.pheromone += deposit_mult / (length ** cost_power)


class Ant:
    def __init__(self):
        self.cities = []
        self.path = []

    def reset(self):
        self.cities = []
        self.path = []

    def get_path(self, origin, destination, alpha=1.0, beta=2.0):
        self.reset()
        self.cities.append(origin)

        max_steps = 50
        steps = 0
        while self.cities[-1] != destination and steps < max_steps:
            current_city = self.cities[-1]
            available_roads = [
                road
                for road in current_city.roads
                if self._other_city(road, current_city) not in self.cities
            ]

            if not available_roads:
                break

            # Standard ACO transition probability: pheromone^alpha * heuristic^beta.
            desirabilities = [
                (road.pheromone ** alpha) * ((1.0 / road.cost) ** beta)
                for road in available_roads
            ]
            total_p = sum(desirabilities)
            if total_p == 0:
                probs = [1.0 / len(available_roads)] * len(available_roads)
            else:
                probs = [value / total_p for value in desirabilities]

            chosen_road = np.random.choice(available_roads, p=probs)
            next_city = self._other_city(chosen_road, current_city)

            self.path.append(chosen_road)
            self.cities.append(next_city)
            steps += 1

        return self.cities[-1] == destination

    @staticmethod
    def _other_city(road, current_city):
        return road.connected_cities[1] if road.connected_cities[0] == current_city else road.connected_cities[0]

    def _remove_loops(self):
        i = 0
        while i < len(self.cities):
            city = self.cities[i]
            if self.cities.count(city) > 1:
                # Find last occurrence index
                last_idx = len(self.cities) - 1 - self.cities[::-1].index(city)
                # Remove loop segment in cities and path
                self.cities = self.cities[:i+1] + self.cities[last_idx+1:]
                self.path = self.path[:i] + self.path[last_idx:]
            i += 1

    def get_path_length(self):
        return sum(road.cost for road in self.path)


def get_percentage_of_dominant_path(ants, destination=None):
    successful = [
        ant
        for ant in ants
        if ant.cities and (destination is None or ant.cities[-1] == destination)
    ]
    paths = [tuple(c.name for c in ant.cities) for ant in successful]
    if not paths:
        return 0.0
    counts = Counter(paths)
    most_common = counts.most_common(1)[0]
    # Failed ants cannot count toward the population-wide dominance threshold.
    return most_common[1] / len(ants)


def build_romania_network():
    cities = {}
    for coord1, coord2, name in LOCATION_LIST:
        cities[name] = City(name)
        cities[name].set_coordinates([coord1, coord2])

    roads = []
    for city1, city2, cost in STEP_COST:
        road = Road([cities[city1], cities[city2]], cost)
        cities[city1].add_road(road)
        cities[city2].add_road(road)
        roads.append(road)

    return cities, roads


def create_graph(cities):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    cities_x = [city.coordinates[0] for key, city in cities.items()]
    cities_y = [city.coordinates[1] for key, city in cities.items()]
    ax.scatter(cities_x, cities_y, c='darkred', s=50, zorder=5)
    
    for name, city in cities.items():
        ax.text(city.coordinates[0] + 5, city.coordinates[1] + 5, name, fontsize=8)
        
    ax.set_title("Ant Colony Optimisation - Romania Route Finding (Arad -> Bucharest)")
    ax.set_aspect(aspect=1.0)
    return fig, ax


def draw_pheromone(ax, roads):
    lines = []
    for road in roads:
        from_coord = road.connected_cities[0].coordinates
        to_coord = road.connected_cities[1].coordinates
        coord_x = [from_coord[0], to_coord[0]]
        coord_y = [from_coord[1], to_coord[1]]
        width = max(0.5, (road.pheromone ** 0.5) * 5)
        line = ax.plot(coord_x, coord_y, c='teal', linewidth=width, alpha=0.7)
        lines.append(line)
    return lines


def main():
    print("=====================================================")
    print(" Lab 5: Ant Colony Optimisation - Exercises         ")
    print("=====================================================")

    cities, roads = build_romania_network()
    origin = cities['Arad']
    destination = cities['Bucharest']

    n_ant = 10
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    initial_pheromone = 0.01

    for road in roads:
        road.set_pheromone(initial_pheromone)

    ants = [Ant() for _ in range(n_ant)]

    max_iteration = 200
    percentage_target = 0.9
    iteration = 0
    best_ant_snapshot = None
    best_cost_overall = float('inf')

    print("\n--- Running ACO Romania Route Optimization ---")
    fig, ax = create_graph(cities)

    while iteration < max_iteration:
        successful_ants = []
        for ant in ants:
            if ant.get_path(origin, destination, alpha, beta):
                successful_ants.append(ant)
                cost = ant.get_path_length()
                if cost < best_cost_overall:
                    best_cost_overall = cost
                    best_ant_snapshot = ([city.name for city in ant.cities], cost)

        for road in roads:
            road.evaporate_pheromone(rho)
            road.deposit_pheromone(successful_ants)

        dom_perc = get_percentage_of_dominant_path(successful_ants, destination)
        iteration += 1

        if iteration % 20 == 0 or dom_perc >= percentage_target:
            print(
                f"Iteration {iteration:03d} | Successful Ants: {len(successful_ants)}/{n_ant} | "
                f"Dominant Path Ratio: {dom_perc * 100:.1f}% | Best-So-Far Cost: {best_cost_overall:.1f} km"
            )

        if successful_ants and dom_perc >= percentage_target:
            break

    draw_pheromone(ax, roads)
    plt.tight_layout()
    plt.show()

    # Report the best path observed during this stochastic search.
    if best_ant_snapshot:
        path_names, path_cost = best_ant_snapshot
        print(f"\n[SUCCESS] Best ACO Route Found: {' -> '.join(path_names)}")
        print(f"          Best Route Distance: {path_cost:.1f} km")
    else:
        print("\n[WARNING] No ant reached Bucharest within the iteration limit.")

    print("\n[SUCCESS] Lab 5 ACO exercises completed successfully.")


if __name__ == '__main__':
    main()
