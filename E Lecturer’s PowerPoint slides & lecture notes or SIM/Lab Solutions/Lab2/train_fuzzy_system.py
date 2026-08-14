# Copyright Author: Dr Tang Tiong Yew
"""
Lab 2: Fuzzy Systems
====================
This script provides the complete solution for Lab 2 exercises, including:
1. Mamdani Fuzzy Control System for Train Brake & Throttle Control
2. 3D Control Surface / Output Space Plotting using Matplotlib
3. Fuzzy Tipping Recommendation System

Execution Mode:
`python3 src/files/lab2_fuzzy.py`
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    from skfuzzy import membership as mf
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False


def run_train_control_system():
    print("\n--- 1. Fuzzy Control System for a Train ---")
    
    # 1. Antecedents & Consequents
    speed = ctrl.Antecedent(np.arange(0, 85.1, 0.1), 'speed')
    distance = ctrl.Antecedent(np.arange(0, 3000.1, 1), 'distance')
    brake = ctrl.Consequent(np.arange(0, 100.1, 0.5), 'brake')
    throttle = ctrl.Consequent(np.arange(0, 100.1, 0.5), 'throttle')

    # 2. Membership Functions
    # Speed
    speed['stopped'] = mf.trimf(speed.universe, [0, 0, 2])
    speed['very slow'] = mf.trimf(speed.universe, [1, 2.5, 4])
    speed['slow'] = mf.trimf(speed.universe, [2.5, 6.5, 10.5])
    speed['medium fast'] = mf.trimf(speed.universe, [6.5, 26.5, 46.5])
    speed['fast'] = mf.trimf(speed.universe, [26.5, 70, 85])

    # Distance
    distance['at'] = mf.trimf(distance.universe, [0, 0, 2])
    distance['very near'] = mf.trimf(distance.universe, [1, 3, 5])
    distance['near'] = mf.trimf(distance.universe, [3, 101.5, 200])
    distance['medium far'] = mf.trimf(distance.universe, [100, 1550, 3000])
    distance['far'] = mf.trimf(distance.universe, [1500, 2250, 3000])

    # Brake
    brake['no'] = mf.trimf(brake.universe, [0, 0, 40])
    brake['very slight'] = mf.trimf(brake.universe, [20, 50, 80])
    brake['slight'] = mf.trimf(brake.universe, [70, 83.5, 97])
    brake['medium'] = mf.trimf(brake.universe, [95, 97, 99])
    brake['full'] = mf.trimf(brake.universe, [98, 100, 100])

    # Throttle
    throttle['no'] = mf.trimf(throttle.universe, [0, 0, 2])
    throttle['very slight'] = mf.trimf(throttle.universe, [1, 3, 5])
    throttle['slight'] = mf.trimf(throttle.universe, [3, 16.5, 30])
    throttle['medium'] = mf.trimf(throttle.universe, [20, 50, 80])
    throttle['full'] = mf.trimf(throttle.universe, [60, 80, 100])

    # 3. Rules Definition
    rule1 = ctrl.Rule(distance['at'] & speed['stopped'], (brake['full'], throttle['no']))
    rule2 = ctrl.Rule(distance['very near'] & speed['stopped'], (brake['full'], throttle['very slight']))
    rule3 = ctrl.Rule(distance['at'] & speed['very slow'], (brake['full'], throttle['no']))
    rule4 = ctrl.Rule(distance['very near'] & speed['very slow'], (brake['medium'], throttle['very slight']))
    rule5 = ctrl.Rule(distance['near'] & speed['very slow'], (brake['slight'], throttle['very slight']))
    rule6 = ctrl.Rule(distance['at'] & speed['slow'], (brake['full'], throttle['no']))
    rule7 = ctrl.Rule(distance['very near'] & speed['slow'], (brake['medium'], throttle['very slight']))
    rule8 = ctrl.Rule(distance['near'] & speed['slow'], (brake['very slight'], throttle['slight']))
    rule9 = ctrl.Rule(distance['medium far'] & speed['medium fast'], (brake['very slight'], throttle['medium']))
    rule10 = ctrl.Rule(distance['far'] & speed['medium fast'], (brake['no'], throttle['full']))
    rule11 = ctrl.Rule(distance['medium far'] & speed['fast'], (brake['very slight'], throttle['medium']))
    rule12 = ctrl.Rule(distance['far'] & speed['fast'], (brake['no'], throttle['full']))

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12]
    train_ctrl = ctrl.ControlSystem(rules=rules)
    train = ctrl.ControlSystemSimulation(control_system=train_ctrl)

    # Test single point
    test_speed = 30.0
    test_dist = 2000.0
    train.input['speed'] = test_speed
    train.input['distance'] = test_dist
    try:
        train.compute()
        print(f"Inputs: Speed={test_speed} km/h, Distance={test_dist} m")
        print(f"Outputs: Brake={train.output.get('brake', 0.0):.2f}%, Throttle={train.output.get('throttle', 0.0):.2f}%")
    except Exception as e:
        print(f"Simulation warning: {e}")

    # Plot 3D output space
    print("Generating 3D Control Surface Plots...")
    x, y = np.meshgrid(np.linspace(speed.universe.min(), speed.universe.max(), 30),
                       np.linspace(distance.universe.min(), distance.universe.max(), 30))
    z_brake = np.zeros_like(x, dtype=float)
    z_throttle = np.zeros_like(x, dtype=float)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            train.input['speed'] = x[i, j]
            train.input['distance'] = y[i, j]
            try:
                train.compute()
                z_brake[i, j] = train.output.get('brake', 0.0)
                z_throttle[i, j] = train.output.get('throttle', 0.0)
            except Exception:
                z_brake[i, j] = 0.0
                z_throttle[i, j] = 0.0

    def plot3d(x_grid, y_grid, z_grid, title_label):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
        ax.set_title(title_label)
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Distance (m)")
        ax.set_zlabel("Output (%)")
        plt.tight_layout()
        plt.show()

    plot3d(x, y, z_brake, "Train Fuzzy Control Surface - Brake Power (%)")
    plot3d(x, y, z_throttle, "Train Fuzzy Control Surface - Throttle Level (%)")


def run_tipping_recommendation_system():
    print("\n--- 2. Fuzzy Tipping Recommendation System ---")
    
    service = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'service')
    food = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'food')
    tips = ctrl.Consequent(np.arange(0, 30.1, 0.5), 'tips')

    service['poor'] = mf.trimf(service.universe, [0, 0, 5])
    service['average'] = mf.trimf(service.universe, [0, 5, 10])
    service['good'] = mf.trimf(service.universe, [5, 10, 10])

    food['poor'] = mf.trimf(food.universe, [0, 0, 5])
    food['average'] = mf.trimf(food.universe, [0, 5, 10])
    food['good'] = mf.trimf(food.universe, [5, 10, 10])

    tips['low'] = mf.trimf(tips.universe, [0, 0, 15])
    tips['medium'] = mf.trimf(tips.universe, [0, 15, 30])
    tips['high'] = mf.trimf(tips.universe, [15, 30, 30])

    r1 = ctrl.Rule(service['poor'] & food['poor'], tips['low'])
    r2 = ctrl.Rule(service['poor'] & food['average'], tips['low'])
    r3 = ctrl.Rule(service['poor'] & food['good'], tips['medium'])
    r4 = ctrl.Rule(service['average'] & food['poor'], tips['low'])
    r5 = ctrl.Rule(service['average'] & food['average'], tips['medium'])
    r6 = ctrl.Rule(service['average'] & food['good'], tips['high'])
    r7 = ctrl.Rule(service['good'] & food['poor'], tips['medium'])
    r8 = ctrl.Rule(service['good'] & food['average'], tips['high'])
    r9 = ctrl.Rule(service['good'] & food['good'], tips['high'])

    tipping_ctrl = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    test_service = 6.5
    test_food = 8.0
    tipping.input['service'] = test_service
    tipping.input['food'] = test_food
    tipping.compute()

    res_tip = tipping.output['tips']
    print(f"Inputs: Service={test_service}/10, Food={test_food}/10")
    print(f"Computed Tip Recommendation: {res_tip:.2f}%")


def main():
    print("=====================================================")
    print(" Lab 2: Fuzzy Systems - Exercise Solutions          ")
    print("=====================================================")

    if not HAS_SKFUZZY:
        print("[!] scikit-fuzzy is not installed. Run: pip install scikit-fuzzy")
        return

    run_train_control_system()
    run_tipping_recommendation_system()
    print("\n[SUCCESS] Lab 2 exercises executed successfully.")


if __name__ == '__main__':
    main()
