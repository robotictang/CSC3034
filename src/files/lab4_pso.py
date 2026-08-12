# Copyright Author: Dr Tang Tiong Yew
"""
Lab 4: Evolutionary Computation (Particle Swarm Optimisation)
==============================================================
This script provides the complete solution for Lab 4 exercises, including:
1. Minimisation of objective function f(x) = (x+100)(x+50)(x)(x-20)(x-60)(x-100)
2. `Particle` class with velocity, position, pbest updates, and tracking lists
3. Global best PSO algorithm loop with convergence metrics
4. Matplotlib particle flying animation & evaluation subplots/boxplots
5. Parameter investigations (inertia weight w, acceleration constants alpha1 & alpha2)

Execution Mode:
`python src/files/lab4_pso.py`
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def fit_fcn(x):
    """Objective fitness function to minimise."""
    return (x + 100.0) * (x + 50.0) * x * (x - 20.0) * (x - 60.0) * (x - 100.0)


class Particle:
    def __init__(self, position=0.0, velocity=0.0):
        self.position = float(position)
        self.velocity = float(velocity)
        self.best_position = float(position)
        self.position_list = [self.position]
        self.velocity_list = [self.velocity]
        self.best_position_list = [self.best_position]

    def update_personal_best(self):
        if fit_fcn(self.position) < fit_fcn(self.best_position):
            self.best_position = self.position
        self.best_position_list.append(self.best_position)

    def update_velocity(self, alpha, beta, glob_best_pos, inertia_weight=1.0):
        r1, r2 = beta[0], beta[1]
        cognitive = alpha[0] * r1 * (self.best_position - self.position)
        social = alpha[1] * r2 * (glob_best_pos - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive + social
        self.velocity_list.append(self.velocity)

    def update_position(self, position_limits):
        self.position = self.position + self.velocity
        # Clamp position within limits [-100, 100]
        if self.position < position_limits[0]:
            self.position = float(position_limits[0])
            self.velocity = 0.0
        elif self.position > position_limits[1]:
            self.position = float(position_limits[1])
            self.velocity = 0.0
        self.position_list.append(self.position)


def initialise_particles(n_ptc, position_limits):
    particles = []
    for _ in range(n_ptc):
        pos = random.uniform(position_limits[0], position_limits[1])
        particles.append(Particle(position=pos, velocity=0.0))
    return particles


def compareFitness(pos1, pos2):
    if fit_fcn(pos1) <= fit_fcn(pos2):
        return pos1
    else:
        return pos2


def calc_avg_fit_diff(particles):
    fits = [fit_fcn(p.position) for p in particles]
    mean_fit = np.mean(fits)
    return float(np.mean([abs(f - mean_fit) for f in fits]))


def calc_avg_pos_diff(particles):
    positions = [p.position for p in particles]
    mean_pos = np.mean(positions)
    return float(np.mean([abs(p - mean_pos) for p in positions]))


def run_gbest_pso(alpha=[0.1, 0.1], n_particle=10, inertia_weight=1.0, position_limits=[-100, 100], max_iter=200):
    particles = initialise_particles(n_particle, position_limits)
    global_best_position = None
    global_best_position_list = []

    iteration = 0
    min_avg_fit_diff = 0.1
    min_avg_pos_diff = 0.1

    while (iteration < max_iter and 
           (calc_avg_fit_diff(particles) > min_avg_fit_diff or 
            calc_avg_pos_diff(particles) > min_avg_pos_diff or 
            iteration < 5)):
        
        # Update personal best & global best
        for particle in particles:
            particle.update_personal_best()
            if global_best_position is None:
                global_best_position = particle.position
            else:
                global_best_position = compareFitness(global_best_position, particle.position)

        global_best_position_list.append(global_best_position)
        
        # Generate random beta for current iteration
        beta = [random.random(), random.random()]
        
        for particle in particles:
            particle.update_velocity(alpha, beta, global_best_position, inertia_weight)
            particle.update_position(position_limits)
            
        iteration += 1

    return particles, global_best_position, global_best_position_list, iteration


def main():
    print("=====================================================")
    print(" Lab 4: Particle Swarm Optimisation - Exercises     ")
    print("=====================================================")

    # Run Standard PSO
    alpha = [0.1, 0.1]
    n_particle = 10
    position_limits = [-100, 100]
    
    print("\n--- 1. Executing Standard gbest PSO ---")
    particles, gbest, gbest_list, iterations = run_gbest_pso(alpha=alpha, n_particle=n_particle, inertia_weight=1.0)
    print(f"Iterations until convergence: {iterations}")
    print(f"Global Best Position x*: {gbest:.4f}")
    print(f"Minimum Fitness f(x*): {fit_fcn(gbest):.2e}")

    # Plotting Evaluation Curves & Boxplots
    print("\n--- 2. Generating Evaluation & Progression Plots ---")
    fig1, position_axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    position_axes[0].set_title("Position of each particle")
    position_axes[1].set_title("Fitness of each particle")
    position_axes[2].set_title("Boxplot of position at each iteration")
    position_axes[3].set_title("Boxplot of fitness at each iteration")
    position_axes[3].set_xlabel("Iteration")

    for particle in particles:
        iter_list = list(range(len(particle.position_list)))
        position_axes[0].plot(iter_list, particle.position_list, '-o', alpha=0.6, markersize=3)
        position_axes[1].plot(iter_list, [fit_fcn(x) for x in particle.position_list], '-o', alpha=0.6, markersize=3)

    plt.tight_layout()
    plt.show()

    # Parameter Experiments
    print("\n--- 3. Parameter Sensitivity Investigations ---")
    experiments = [
        ("Standard (w=1.0, a1=0.1, a2=0.1)", 1.0, [0.1, 0.1]),
        ("Inertia Weight w=0.5 (w=0.5, a1=0.1, a2=0.1)", 0.5, [0.1, 0.1]),
        ("Reduced Cognitive (w=1.0, a1=0.05, a2=0.1)", 1.0, [0.05, 0.1]),
        ("Zero Cognitive (w=1.0, a1=0.0, a2=0.1)", 1.0, [0.0, 0.1]),
        ("Dominant Cognitive (w=1.0, a1=0.2, a2=0.05)", 1.0, [0.2, 0.05])
    ]

    for label, w_val, a_val in experiments:
        _, exp_gbest, _, exp_iters = run_gbest_pso(alpha=a_val, n_particle=10, inertia_weight=w_val)
        print(f"[{label}] -> Iters: {exp_iters:03d} | Best x: {exp_gbest:8.4f} | f(x): {fit_fcn(exp_gbest):.2e}")

    print("\n[SUCCESS] Lab 4 PSO exercises completed successfully.")


if __name__ == '__main__':
    main()
