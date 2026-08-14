# Copyright Author: Dr Tang Tiong Yew
"""
NVIDIA Isaac Sim & Standalone Python Script:
Genetic Algorithm (GA) for Mobile Robot Trajectory & Obstacle Avoidance Optimisation

This script demonstrates how a Genetic Algorithm (GA) searches for a short,
collision-free path through an arena. The best path found is not guaranteed globally optimal.

Features:
1. Dual Execution Modes:
   - Standalone Python Mode (using Matplotlib for 2D visualization if Isaac Sim is absent)
   - Photorealistic NVIDIA Isaac Sim Mode (GPU physics & 3D USD stage rendering)
2. GA Architecture:
   - Binary & Real-Valued Waypoint Encoding
   - Fitness evaluation incorporating path length and exact segment-obstacle collision penalties
   - Roulette Wheel Parent Selection, One-Point Crossover, and Bit-Flip Mutation
"""

import os
import sys
import math
import random
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
# Problem & Environment Setup
# =====================================================================
ARENA_BOUNDS = (-8.0, 8.0)
START_POS = np.array([-6.0, -6.0, 0.25])
GOAL_POS = np.array([6.0, 6.0, 0.25])

# List of spherical obstacles: (x, y, radius)
OBSTACLES = [
    (0.0, 0.0, 1.8),
    (-2.5, 2.5, 1.4),
    (2.5, -2.5, 1.4),
    (-3.0, -2.0, 1.2),
    (2.0, 3.5, 1.2)
]

NUM_WAYPOINTS = 6  # Intermediate waypoints per chromosome
GENE_BITS = 6       # 6 bits per coordinate (0..63) -> maps to (-8.0, 8.0)

def binary2value(binary_str, min_val=-8.0, max_val=8.0):
    """Converts a binary string to a continuous float in [min_val, max_val]."""
    dec = int(binary_str, 2)
    max_dec = (1 << len(binary_str)) - 1
    return min_val + (dec / max_dec) * (max_val - min_val)

def value2binary(val, min_val=-8.0, max_val=8.0, bits=GENE_BITS):
    """Converts a continuous float in [min_val, max_val] to a binary string."""
    val_clamped = max(min_val, min(max_val, val))
    max_dec = (1 << bits) - 1
    dec = int(round((val_clamped - min_val) / (max_val - min_val) * max_dec))
    return format(dec, f'0{bits}b')

# =====================================================================
# Genetic Algorithm Functions
# =====================================================================
def decode_chromosome(chromosome):
    """Decodes a binary chromosome string into a list of 3D waypoints [Start, W1..Wn, Goal]."""
    waypoints = [START_POS[:2]]
    chunk_size = GENE_BITS * 2
    for i in range(0, len(chromosome), chunk_size):
        x_bin = chromosome[i : i + GENE_BITS]
        y_bin = chromosome[i + GENE_BITS : i + chunk_size]
        x = binary2value(x_bin)
        y = binary2value(y_bin)
        waypoints.append(np.array([x, y]))
    waypoints.append(GOAL_POS[:2])
    return waypoints

def generate_individual():
    """Generates a random binary chromosome string for intermediate waypoints."""
    chrom_len = NUM_WAYPOINTS * GENE_BITS * 2
    return ''.join(random.choice(['0', '1']) for _ in range(chrom_len))

def generate_population(pop_size):
    """Initialises a population of random chromosomes."""
    return [generate_individual() for _ in range(pop_size)]


def evaluate_trajectory(waypoints):
    """Return path length and the number of obstacle intersections."""
    path_length = 0.0
    num_collisions = 0
    for p1, p2 in zip(waypoints, waypoints[1:]):
        segment = p2 - p1
        segment_length_sq = float(np.dot(segment, segment))
        path_length += float(np.linalg.norm(segment))
        for ox, oy, radius in OBSTACLES:
            center = np.array([ox, oy])
            if segment_length_sq == 0:
                closest = p1
            else:
                t = float(
                    np.clip(
                        np.dot(center - p1, segment) / segment_length_sq,
                        0.0,
                        1.0,
                    )
                )
                closest = p1 + t * segment
            if np.linalg.norm(closest - center) <= radius + 0.3:
                num_collisions += 1
    return path_length, num_collisions

def calculate_fitness(chromosome):
    """
    Calculates fitness of a chromosome trajectory.
    Fitness = 1000 / (1.0 + path_length*0.2 + num_collisions*40.0)
    """
    waypoints = decode_chromosome(chromosome)
    path_length, num_collisions = evaluate_trajectory(waypoints)

    fitness = 1000.0 / (1.0 + path_length * 0.2 + num_collisions * 40.0)
    return max(0.0001, fitness)

def select_parents(population, fitnesses):
    """Roulette Wheel Selection to select parent pairs."""
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    parents = []
    for _ in range(len(population)):
        parent = np.random.choice(population, p=probs)
        parents.append(parent)
    return parents

def crossover(parent1, parent2, p_crossover=0.85):
    """Performs One-Point Crossover between two parent chromosomes."""
    if random.random() > p_crossover:
        return parent1, parent2
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(chromosome, p_mutation=0.03):
    """Performs Bit-Flip Mutation on a chromosome."""
    chrom_list = list(chromosome)
    for i in range(len(chrom_list)):
        if random.random() < p_mutation:
            chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
    return ''.join(chrom_list)

def run_ga_optimization(pop_size=40, max_generations=60, p_crossover=0.85, p_mutation=0.03):
    """Executes the Genetic Algorithm optimization loop."""
    population = generate_population(pop_size)
    best_overall_chrom = None
    best_overall_fitness = -1.0
    
    print('=======================================================')
    print(' Running Genetic Algorithm Robot Trajectory Optimisation ')
    print('=======================================================')
    
    for gen in range(max_generations):
        fitnesses = [calculate_fitness(chrom) for chrom in population]
        
        # Track best individual
        max_idx = int(np.argmax(fitnesses))
        if fitnesses[max_idx] > best_overall_fitness:
            best_overall_fitness = fitnesses[max_idx]
            best_overall_chrom = population[max_idx]
            
        if (gen + 1) % 10 == 0 or gen == 0 or gen == max_generations - 1:
            waypoints = decode_chromosome(population[max_idx])
            path_length, collisions = evaluate_trajectory(waypoints)
            print(
                f'Generation {gen+1:02d}/{max_generations:02d} | '
                f'Best Fitness: {fitnesses[max_idx]:.2f} | '
                f'Path: {path_length:.2f}m | Collisions: {collisions}'
            )
            
        # Selection
        parents = select_parents(population, fitnesses)
        
        # Crossover & Mutation
        next_population = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % pop_size]
            o1, o2 = crossover(p1, p2, p_crossover)
            next_population.append(mutate(o1, p_mutation))
            next_population.append(mutate(o2, p_mutation))
            
        population = next_population[:pop_size]
        # Preserve the best-so-far chromosome (elitism).
        population[0] = best_overall_chrom
        
    return best_overall_chrom, best_overall_fitness

# =====================================================================
# Isaac Sim Photorealistic 3D Simulation
# =====================================================================
def run_isaac_sim(best_chrom):
    """Runs the evolved robot trajectory inside NVIDIA Isaac Sim."""
    if not HAS_ISAAC_SIM:
        print('[INFO] NVIDIA Isaac Sim environment not detected. Running Standalone Matplotlib Plotter.')
        run_matplotlib_visualization(best_chrom)
        return

    print('=== Launching NVIDIA Isaac Sim Simulation Stage ===')
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({'headless': False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({'headless': False})

    # Isaac Sim 5+ renamed the legacy ``omni.isaac.core`` package.
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import VisualSphere, VisualCuboid
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualSphere, VisualCuboid
        from omni.isaac.core.utils.viewports import set_camera_view

    world = World(stage_units_in_meters=1.0)
    # Create a local ground plane.  ``add_default_ground_plane`` fetches an
    # Isaac sample USD from the internet, which is unavailable on this host.
    import omni.usd
    from omni.physx.scripts import physicsUtils
    from pxr import Gf
    physicsUtils.add_ground_plane(
        omni.usd.get_context().get_stage(),
        "/World/GroundPlane",
        "Z",
        20.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.35, 0.35, 0.35),
    )

    # -----------------------------------------------------------------
    # Presentation layer: a clean, high-contrast "planning arena" makes
    # the GA result easier to inspect than the default grey stage.
    # -----------------------------------------------------------------
    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom, UsdLux

    # Soft studio lighting works without any external USD assets.
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/AmbientSky")
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(0.16, 0.22, 0.38))
    key_light = UsdLux.SphereLight.Define(stage, "/World/Lighting/KeyLight")
    key_light.CreateRadiusAttr(2.5)
    key_light.CreateIntensityAttr(22000.0)
    key_light.CreateColorAttr(Gf.Vec3f(0.72, 0.86, 1.0))
    key_light.AddTranslateOp().Set(Gf.Vec3d(-6.0, -8.0, 12.0))
    rim_light = UsdLux.SphereLight.Define(stage, "/World/Lighting/RimLight")
    rim_light.CreateRadiusAttr(2.0)
    rim_light.CreateIntensityAttr(15000.0)
    rim_light.CreateColorAttr(Gf.Vec3f(1.0, 0.36, 0.16))
    rim_light.AddTranslateOp().Set(Gf.Vec3d(8.0, 6.0, 9.0))

    # A dark inset floor, four raised rails, and corner beacons frame the
    # playable space while retaining the local physics ground plane above.
    world.scene.add(VisualCuboid(
        prim_path="/World/Arena/Floor", name="arena_floor",
        position=np.array([0.0, 0.0, -0.12]),
        scale=np.array([16.8, 16.8, 0.18]), color=np.array([0.035, 0.06, 0.11])
    ))
    rail_specs = [
        ("North", [0.0, 8.25, 0.18], [16.8, 0.16, 0.35]),
        ("South", [0.0, -8.25, 0.18], [16.8, 0.16, 0.35]),
        ("East", [8.25, 0.0, 0.18], [0.16, 16.8, 0.35]),
        ("West", [-8.25, 0.0, 0.18], [0.16, 16.8, 0.35]),
    ]
    for name, position, scale in rail_specs:
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Arena/Rails/{name}", name=f"rail_{name.lower()}",
            position=np.array(position), scale=np.array(scale),
            color=np.array([0.05, 0.55, 0.86])
        ))
    for idx, (x, y) in enumerate(((-7.85, -7.85), (-7.85, 7.85), (7.85, -7.85), (7.85, 7.85))):
        world.scene.add(VisualSphere(
            prim_path=f"/World/Arena/Beacons/Beacon_{idx}", name=f"beacon_{idx}",
            position=np.array([x, y, 0.38]), radius=0.16,
            color=np.array([1.0, 0.42, 0.08])
        ))

    # Spawn Start & Goal Markers
    world.scene.add(VisualSphere(
        prim_path='/World/StartMarker', name='start_marker',
        position=START_POS, radius=0.48, color=np.array([0.05, 0.95, 0.38])
    ))
    world.scene.add(VisualSphere(
        prim_path='/World/GoalMarker', name='goal_marker',
        position=GOAL_POS, radius=0.48, color=np.array([1.0, 0.16, 0.20])
    ))

    # Spawn Obstacles in Isaac Sim Stage
    for idx, (ox, oy, r) in enumerate(OBSTACLES):
        world.scene.add(VisualSphere(
            prim_path=f'/World/Obstacles/Obstacle_{idx}',
            name=f'obstacle_{idx}',
            position=np.array([ox, oy, r]),
            radius=r, color=np.array([0.32, 0.12, 0.48])
        ))
        # A low halo distinguishes the obstacle footprint from its 3D body.
        world.scene.add(VisualCuboid(
            prim_path=f'/World/Obstacles/Halo_{idx}', name=f'obstacle_halo_{idx}',
            position=np.array([ox, oy, 0.025]),
            scale=np.array([2.0 * (r + 0.22), 2.0 * (r + 0.22), 0.025]),
            color=np.array([0.95, 0.16, 0.48])
        ))

    # Decode Evolved Waypoints & Spawn Visual Waypoint Markers
    best_waypoints = decode_chromosome(best_chrom)
    # Lay a glowing-looking cyan route on the floor.  Individual cuboids keep
    # the result readable from the overhead camera and require no asset files.
    for idx, (p1, p2) in enumerate(zip(best_waypoints, best_waypoints[1:])):
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        yaw = math.atan2(delta[1], delta[0])
        world.scene.add(VisualCuboid(
            prim_path=f'/World/Trajectory/Segment_{idx}', name=f'trajectory_segment_{idx}',
            position=np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0.055]),
            scale=np.array([length, 0.09, 0.035]),
            orientation=np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
            color=np.array([0.05, 0.84, 1.0])
        ))
    for idx, wp in enumerate(best_waypoints[1:-1]):
        world.scene.add(VisualSphere(
            prim_path=f'/World/Waypoints/WP_{idx}',
            name=f'wp_{idx}',
            position=np.array([wp[0], wp[1], 0.18]), radius=0.20,
            color=np.array([0.05, 0.78, 1.0])
        ))

    # Spawn the animated robot.  A visual (kinematic) sphere gives a smooth,
    # deterministic path rather than relying on rigid-body friction settings.
    robot_prim = world.scene.add(VisualSphere(
        prim_path='/World/Robot/GA_Robot',
        name='ga_robot',
        position=START_POS,
        radius=0.34,
        color=np.array([1.0, 0.62, 0.05])
    ))

    world.reset()
    set_camera_view(
        eye=[19.0, -23.0, 24.0],
        target=[0.0, 0.0, 0.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Animate the robot along the evolved route.  Repeating the route makes
    # the movement easy to observe in the Isaac Sim viewport.
    print('[INFO] Executing visible GA trajectory in NVIDIA Isaac Sim Stage...')
    robot_position = START_POS.copy()
    wp_idx = 1
    speed_mps = 2.0
    physics_dt = 1.0 / 60.0
    while simulation_app.is_running():
        world.step(render=True)
        target_wp = np.array([*best_waypoints[wp_idx], START_POS[2]])
        direction = target_wp - robot_position
        distance = np.linalg.norm(direction)
        if distance <= speed_mps * physics_dt:
            robot_position = target_wp
            wp_idx += 1
            if wp_idx == len(best_waypoints):
                wp_idx = 1
                robot_position = START_POS.copy()
        else:
            robot_position += direction / distance * speed_mps * physics_dt
        robot_prim.set_world_pose(position=robot_position)

    simulation_app.close()

def run_matplotlib_visualization(best_chrom):
    """Plots 2D trajectory of evolved GA robot when Isaac Sim is not active."""
    try:
        import matplotlib.pyplot as plt
        waypoints = np.array(decode_chromosome(best_chrom))
        
        plt.figure(figsize=(8, 8))
        plt.title('Genetic Algorithm (GA) Robot Trajectory Optimisation')
        
        # Plot Obstacles
        for ox, oy, r in OBSTACLES:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.6)
            plt.gca().add_patch(circle)
            
        # Plot Trajectory & Waypoints
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'o--', color='darkorange', label='Evolved GA Trajectory')
        plt.plot(START_POS[0], START_POS[1], 'go', markersize=12, label='Start')
        plt.plot(GOAL_POS[0], GOAL_POS[1], 'ro', markersize=12, label='Goal')
        
        plt.xlim(ARENA_BOUNDS)
        plt.ylim(ARENA_BOUNDS)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True)
        plt.legend()
        print('[INFO] Displaying Matplotlib Plot. Close window to finish.')
        plt.show()
    except Exception as e:
        print(f'[WARNING] Could not plot Matplotlib figure: {e}')

# =====================================================================
# Main Entry Point
# =====================================================================
if __name__ == '__main__':
    best_chrom, best_fit = run_ga_optimization(pop_size=40, max_generations=60)
    print(f'\n[SUCCESS] Evolved Best Chromosome: {best_chrom}')
    print(f'[SUCCESS] Evolved Best Fitness: {best_fit:.4f}')
    
    if HAS_ISAAC_SIM:
        run_isaac_sim(best_chrom)
    else:
        run_matplotlib_visualization(best_chrom)
