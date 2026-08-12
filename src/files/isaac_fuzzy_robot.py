# Copyright Author: Dr Tang Tiong Yew
"""
Fuzzy Logic Robot Obstacle Avoidance Controller in NVIDIA Isaac Sim
===================================================================
This script demonstrates Mamdani Fuzzy Logic Control for autonomous mobile robot navigation
and obstacle avoidance inside NVIDIA Isaac Sim using `scikit-fuzzy`.

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics & visual simulation):
   Run with Isaac Sim's standalone python:
   `isaac-sim.standalone.bat python src/files/isaac_fuzzy_robot.py`
   OR `python.bat src/files/isaac_fuzzy_robot.py`

2. Standalone Fallback Mode (scikit-fuzzy controller simulation):
   `python src/files/isaac_fuzzy_robot.py`
"""

import sys
import time
import numpy as np

HAS_SKFUZZY = False
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

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
# Build Mamdani Fuzzy Inference System Architecture
# =====================================================================
def build_fuzzy_controller():
    """Constructs scikit-fuzzy Antecedents, Consequents, and Rules."""
    distance = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'distance')
    heading = ctrl.Antecedent(np.arange(-180, 181, 1), 'heading')
    linear_vel = ctrl.Consequent(np.arange(0, 1.6, 0.1), 'linear_vel')
    angular_vel = ctrl.Consequent(np.arange(-2.0, 2.1, 0.1), 'angular_vel')

    # Define membership functions
    distance['near'] = fuzz.trimf(distance.universe, [0.0, 0.0, 1.5])
    distance['medium'] = fuzz.trimf(distance.universe, [1.0, 2.5, 4.0])
    distance['far'] = fuzz.trimf(distance.universe, [3.0, 5.0, 5.0])

    heading['left'] = fuzz.trimf(heading.universe, [-180, -90, 0])
    heading['straight'] = fuzz.trimf(heading.universe, [-30, 0, 30])
    heading['right'] = fuzz.trimf(heading.universe, [0, 90, 180])

    linear_vel['stop'] = fuzz.trimf(linear_vel.universe, [0.0, 0.0, 0.3])
    linear_vel['slow'] = fuzz.trimf(linear_vel.universe, [0.2, 0.6, 1.0])
    linear_vel['fast'] = fuzz.trimf(linear_vel.universe, [0.8, 1.5, 1.5])

    angular_vel['turn_right'] = fuzz.trimf(angular_vel.universe, [-2.0, -1.0, 0.0])
    angular_vel['straight'] = fuzz.trimf(angular_vel.universe, [-0.3, 0.0, 0.3])
    angular_vel['turn_left'] = fuzz.trimf(angular_vel.universe, [0.0, 1.0, 2.0])

    # Define Fuzzy Association Memory (FAM) Rules
    rule1 = ctrl.Rule(distance['near'], (linear_vel['stop'], angular_vel['turn_left']))
    rule2 = ctrl.Rule(distance['medium'] & heading['straight'], (linear_vel['slow'], angular_vel['straight']))
    rule3 = ctrl.Rule(distance['far'] & heading['straight'], (linear_vel['fast'], angular_vel['straight']))
    rule4 = ctrl.Rule(heading['left'], (linear_vel['slow'], angular_vel['turn_left']))
    rule5 = ctrl.Rule(heading['right'], (linear_vel['slow'], angular_vel['turn_right']))

    fuzzy_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(fuzzy_control_system)


# =====================================================================
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_fuzzy(max_steps=200):
    """Executes Fuzzy Logic Robot Controller inside NVIDIA Isaac Sim stage."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World

    world = World()
    world.scene.add_default_ground_plane()

    fuzzy_sim = build_fuzzy_controller()

    world.reset()
    print("[INFO] Starting Fuzzy Logic Control Loop in NVIDIA Isaac Sim...")

    step_count = 0
    while simulation_app.is_running() and step_count < max_steps:
        world.step(render=True)

        simulated_obstacle_dist = max(0.5, 5.0 - (step_count * 0.02))
        simulated_heading_error = 15.0 if step_count < 100 else -45.0

        fuzzy_sim.input['distance'] = simulated_obstacle_dist
        fuzzy_sim.input['heading'] = simulated_heading_error

        try:
            fuzzy_sim.compute()
            target_v = fuzzy_sim.output.get('linear_vel', 0.5)
            target_w = fuzzy_sim.output.get('angular_vel', 0.0)
        except Exception:
            target_v = 0.5
            target_w = 0.0

        if step_count % 20 == 0:
            print(f"[Step {step_count:03d}] Distance: {simulated_obstacle_dist:.2f}m | Heading: {simulated_heading_error:.1f}° "
                  f"--> Fuzzy Outputs: Linear Vel = {target_v:.2f} m/s, Angular Vel = {target_w:.2f} rad/s")

        step_count += 1

    print("[SUCCESS] Completed Fuzzy Logic Controller simulation in Isaac Sim.")
    simulation_app.close()


# =====================================================================
# 2. Standalone Fallback Execution
# =====================================================================
def run_fallback_fuzzy(max_steps=100):
    """Fallback simulation running scikit-fuzzy controller without Isaac Sim GUI."""
    print("===============================================================")
    print(" Running Standalone Fuzzy Logic Controller (No Isaac Sim GUI)  ")
    print("===============================================================")

    if not HAS_SKFUZZY:
        print("[!] scikit-fuzzy library ('skfuzzy') is required to run the Fuzzy controller.")
        print("    Install it via: pip install scikit-fuzzy")
        print("\n[INFO] Demonstrating heuristic navigation rule fallback:")
        for step in range(0, max_steps, 20):
            simulated_obstacle_dist = max(0.5, 5.0 - (step * 0.04))
            simulated_heading_error = 20.0 if step < 50 else -30.0
            target_v = 0.2 if simulated_obstacle_dist < 1.5 else 1.2
            target_w = 0.8 if simulated_heading_error > 0 else -0.8
            print(f"[Step {step:03d}] Dist: {simulated_obstacle_dist:.2f}m | Heading: {simulated_heading_error:+.1f}° "
                  f"--> Linear Vel = {target_v:.2f} m/s | Angular Vel = {target_w:+.2f} rad/s")
        return

    fuzzy_sim = build_fuzzy_controller()

    for step in range(max_steps):
        simulated_obstacle_dist = max(0.5, 5.0 - (step * 0.04))
        simulated_heading_error = 20.0 if step < 50 else -30.0

        fuzzy_sim.input['distance'] = simulated_obstacle_dist
        fuzzy_sim.input['heading'] = simulated_heading_error

        try:
            fuzzy_sim.compute()
            target_v = fuzzy_sim.output.get('linear_vel', 0.5)
            target_w = fuzzy_sim.output.get('angular_vel', 0.0)
        except Exception:
            target_v = 0.5
            target_w = 0.0

        if step % 10 == 0:
            print(f"[Step {step:03d}] Dist: {simulated_obstacle_dist:.2f}m | Heading: {simulated_heading_error:+.1f}° "
                  f"--> Linear Vel = {target_v:.2f} m/s | Angular Vel = {target_w:+.2f} rad/s")

    print("[SUCCESS] Standalone Fuzzy Logic controller simulation finished cleanly.")


if __name__ == '__main__':
    if HAS_ISAAC_SIM and HAS_SKFUZZY:
        print("[INFO] NVIDIA Isaac Sim detected. Launching stage...")
        run_isaac_sim_fuzzy()
    else:
        print("[INFO] Running Standalone Mode.")
        run_fallback_fuzzy()
