# Copyright Author: Dr Tang Tiong Yew
"""
Particle Swarm Optimisation (PSO) for Robot Swarm Simulation
============================================================
This script demonstrates multi-robot target localization using Particle Swarm Optimisation.

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics simulation):
   Run with Isaac Sim's standalone python:
   `isaac-sim.standalone.bat python src/files/isaac_pso_swarm.py`
   OR `python.bat src/files/isaac_pso_swarm.py`

2. Matplotlib Swarm Fallback Mode (Standard Python 2D/3D simulation):
   `python src/files/isaac_pso_swarm.py`
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
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_pso(n_robots=10, max_iterations=200):
    """Executes the Robot Swarm PSO inside NVIDIA Isaac Sim photorealistic environment."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere, VisualCuboid

    class RobotParticleIsaac:
        def __init__(self, robot_id, initial_pos, world):
            self.id = robot_id
            self.position = np.array(initial_pos, dtype=np.float64)
            self.velocity = np.zeros(2, dtype=np.float64)
            self.best_position = np.copy(self.position[:2])
            self.best_fitness = float('inf')

            self.prim = world.scene.add(
                DynamicSphere(
                    prim_path=f"/World/Robot_{robot_id}",
                    name=f"robot_{robot_id}",
                    position=self.position,
                    radius=0.3,
                    color=np.array([0.1, 0.6, 0.9])
                )
            )

        def evaluate_fitness(self, target_pos):
            current_pos, _ = self.prim.get_world_pose()
            self.position[:2] = current_pos[:2]
            fitness = np.linalg.norm(self.position[:2] - target_pos[:2])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = np.copy(self.position[:2])
            return fitness

        def update_motion(self, alpha, beta, global_best_pos, max_speed=1.5):
            r1, r2 = beta[0], beta[1]
            cognitive = alpha[0] * r1 * (self.best_position - self.position[:2])
            social = alpha[1] * r2 * (global_best_pos[:2] - self.position[:2])
            self.velocity = 0.5 * self.velocity + cognitive + social

            speed = np.linalg.norm(self.velocity)
            if speed > max_speed:
                self.velocity = (self.velocity / speed) * max_speed

            linear_velocity = np.array([self.velocity[0], self.velocity[1], 0.0])
            self.prim.set_linear_velocity(linear_velocity)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    target_pos = np.array([10.0, 10.0, 0.5])
    world.scene.add(
        VisualCuboid(
            prim_path="/World/TargetBeacon",
            name="target_beacon",
            position=target_pos,
            scale=np.array([0.6, 0.6, 1.0]),
            color=np.array([1.0, 0.0, 0.0])
        )
    )

    alpha = [0.15, 0.2]
    position_bounds = [-15.0, 15.0]

    robots = []
    for i in range(n_robots):
        init_x = np.random.uniform(position_bounds[0], position_bounds[1])
        init_y = np.random.uniform(position_bounds[0], position_bounds[1])
        init_pos = [init_x, init_y, 0.3]
        robots.append(RobotParticleIsaac(i, init_pos, world))

    world.reset()
    global_best_pos = None
    global_best_fitness = float('inf')

    print("=======================================================")
    print(" Running NVIDIA Isaac Sim Robot Swarm PSO Simulation   ")
    print("=======================================================")

    iteration = 0
    while simulation_app.is_running() and iteration < max_iterations:
        world.step(render=True)

        for robot in robots:
            fitness = robot.evaluate_fitness(target_pos)
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = np.copy(robot.position[:2])

        beta = [np.random.random(), np.random.random()]

        for robot in robots:
            robot.update_motion(alpha, beta, global_best_pos)

        if iteration % 10 == 0:
            print(f"Iteration {iteration:03d} | Global Best Distance to Target: {global_best_fitness:.3f} m")

        iteration += 1

    print(f"\nTarget Beacon Reached at {global_best_pos} with distance {global_best_fitness:.3f}m")
    simulation_app.close()


# =====================================================================
# 2. Standalone Matplotlib Swarm Fallback Simulator
# =====================================================================
def run_matplotlib_fallback_pso(n_robots=10, max_iterations=100):
    """Fallback visualizer for standard Python environments without NVIDIA Isaac Sim."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib is required to run the fallback visualization.")
        print("    Install it via: pip install matplotlib")
        return

    print("=======================================================")
    print(" [Fallback Mode] Standard 2D Swarm Robotics PSO        ")
    print(" (NVIDIA Isaac Sim module not detected in current Python)")
    print("=======================================================")

    target_pos = np.array([10.0, 10.0])
    position_bounds = [-15.0, 15.0]
    alpha = [0.15, 0.2]

    # Initialize robot particles
    positions = np.random.uniform(position_bounds[0], position_bounds[1], size=(n_robots, 2))
    velocities = np.zeros((n_robots, 2))
    personal_bests = np.copy(positions)
    personal_best_fitness = np.array([np.linalg.norm(p - target_pos) for p in positions])

    global_best_idx = np.argmin(personal_best_fitness)
    global_best_pos = np.copy(personal_bests[global_best_idx])
    global_best_fitness = personal_best_fitness[global_best_idx]

    # Setup animation plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(position_bounds[0] - 2, position_bounds[1] + 2)
    ax.set_ylim(position_bounds[0] - 2, position_bounds[1] + 2)
    ax.set_title("Robot Swarm PSO Target Localization (Isaac Sim Logic Simulation)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True)

    # Plot target beacon
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target Beacon')
    # Plot robot particles
    robot_dots, = ax.plot(positions[:, 0], positions[:, 1], 'bo', markersize=8, label='Robots (Particles)')
    gbest_dot, = ax.plot(global_best_pos[0], global_best_pos[1], 'gx', markersize=12, markeredgewidth=3, label='Global Best')
    ax.legend(loc='upper left')

    for iteration in range(max_iterations):
        # 1. Fitness evaluation
        for i in range(n_robots):
            fitness = np.linalg.norm(positions[i] - target_pos)
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_bests[i] = np.copy(positions[i])

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_pos = np.copy(positions[i])

        # 2. Velocity & position updates
        r1, r2 = np.random.random(), np.random.random()
        for i in range(n_robots):
            cognitive = alpha[0] * r1 * (personal_bests[i] - positions[i])
            social = alpha[1] * r2 * (global_best_pos - positions[i])
            velocities[i] = 0.5 * velocities[i] + cognitive + social

            # Speed constraint
            speed = np.linalg.norm(velocities[i])
            if speed > 1.5:
                velocities[i] = (velocities[i] / speed) * 1.5

            positions[i] += velocities[i]

        # Update visualization
        robot_dots.set_data(positions[:, 0], positions[:, 1])
        gbest_dot.set_data([global_best_pos[0]], [global_best_pos[1]])
        ax.set_title(f"Robot Swarm PSO | Iteration {iteration+1}/{max_iterations} | Best Distance: {global_best_fitness:.2f}m")
        try:
            plt.draw()
            plt.pause(0.05)
        except Exception:
            break

        if global_best_fitness < 0.1:
            print(f"Target reached at iteration {iteration+1}!")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    if HAS_ISAAC_SIM:
        run_isaac_sim_pso()
    else:
        run_matplotlib_fallback_pso()
