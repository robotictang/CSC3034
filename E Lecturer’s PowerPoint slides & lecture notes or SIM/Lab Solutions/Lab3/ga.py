# Copyright Author: Dr Tang Tiong Yew
"""
Lab 3: Evolutionary Computation (Genetic Algorithm)
===================================================
This script provides the complete solution for Lab 3 exercises, including:
1. Binary feature encoding and decoding (`value2binary`, `binary2value`)
2. Population initialization (`generatePopulation`)
3. Fitness calculation (`calculateFitness`) for paper square cutting problem
4. Parent selection (`selectParents` via Roulette Wheel)
5. Single-point crossover (`crossover`)
6. Uniform bit-flip mutation (`mutate`)
7. Population convergence distance calculation (`findOverallDistance`)
8. Complete GA optimization loop execution

Execution Mode:
`python3 src/files/lab3_ga.py`
"""

import random
import numpy as np


def value2binary(value, bits=6):
    """Converts an integer decimal value to its binary string representation."""
    val_clamped = min((1 << bits) - 1, max(0, int(value)))
    return format(val_clamped, f'0{bits}b')


def binary2value(binary_str):
    """Converts a binary string representation to its integer decimal value."""
    return int(binary_str, 2)


def generatePopulation(pop_size, pop_min, pop_max):
    """Generates the first generation randomly based on population size and value range."""
    return [random.randint(pop_min, pop_max) for _ in range(pop_size)]


def calculateFitness(value, w=20, h=15):
    """
    Calculates the fitness of a chromosome (side length x in cm).
    Maximises total used paper area for a sheet of width w and height h.
    """
    x = value
    if x <= 0 or x > min(w, h):
        return 0.0
    
    num_w = w // x
    num_h = h // x
    num_squares = num_w * num_h
    area = x * x
    
    fitness = num_squares * area
    return float(fitness)


def selectParents(chromosomes, pop_size):
    """Selects parent pairs using Roulette Wheel Selection."""
    fitnesses = [calculateFitness(c) for c in chromosomes]
    total_fit = sum(fitnesses)
    
    if total_fit == 0:
        probs = [1.0 / len(chromosomes)] * len(chromosomes)
    else:
        probs = [f / total_fit for f in fitnesses]
        
    parents = []
    num_pairs = pop_size // 2
    for _ in range(num_pairs):
        p1 = int(np.random.choice(chromosomes, p=probs))
        p2 = int(np.random.choice(chromosomes, p=probs))
        parents.append([p1, p2])
        
    return parents


def crossover(parents, bits=6):
    """Performs single-point crossover on a pair of parent chromosomes."""
    p1, p2 = parents[0], parents[1]
    b1 = value2binary(p1, bits)
    b2 = value2binary(p2, bits)
    
    point = random.randint(1, bits - 1)
    o1_bin = b1[:point] + b2[point:]
    o2_bin = b2[:point] + b1[point:]
    
    o1 = binary2value(o1_bin)
    o2 = binary2value(o2_bin)
    return [o1, o2]


def mutate(chromosome, p_mutation=0.05, bits=6, value_min=1, value_max=10):
    """Mutate a chromosome and repair it to the permitted problem domain."""
    b_str = list(value2binary(chromosome, bits))
    for i in range(len(b_str)):
        if random.random() < p_mutation:
            b_str[i] = '1' if b_str[i] == '0' else '0'
    mutated = binary2value(''.join(b_str))
    return min(value_max, max(value_min, mutated))


def findOverallDistance(chromosomes):
    """Calculates overall distance among fitnesses of all chromosomes."""
    fitnesses = [calculateFitness(c) for c in chromosomes]
    mean_fit = np.mean(fitnesses)
    diffs = [abs(f - mean_fit) for f in fitnesses]
    return float(np.mean(diffs))


def main():
    print("=====================================================")
    print(" Lab 3: Genetic Algorithm - Exercise Solutions      ")
    print("=====================================================")

    # Test individual functions
    print("\n--- 1. Testing Binary Conversion Functions ---")
    print(f"value2binary(10): {value2binary(10)}")
    print(f"binary2value('1001'): {binary2value('1001')}")

    print("\n--- 2. Initializing Population ---")
    init_pop = generatePopulation(8, 0, 10)
    print(f"Random initial population (8 chromosomes): {init_pop}")

    print("\n--- 3. Testing Fitness Calculation & Selection ---")
    sample_fit = calculateFitness(5)
    print(f"Fitness of chromosome x=5: {sample_fit}")

    parents = selectParents(init_pop, len(init_pop))
    print(f"Selected parent pairs: {parents}")

    offsprings = crossover(parents[0])
    print(f"Crossover of {parents[0]} -> Offsprings: {offsprings}")

    mutated_val = mutate(15, 0.1)
    print(f"Mutation of 15 (p_m=0.1) -> {mutated_val}")

    dist = findOverallDistance(init_pop)
    print(f"Overall fitness distance of initial population: {dist:.2f}")

    # Full GA Run
    print("\n--- 4. Executing Full Genetic Algorithm Loop ---")
    pop_size = 10
    pop_min = 1
    pop_max = 10
    curr_iter = 0
    max_iter = 100
    min_overalldistance = 0.5
    p_mutation = 0.05

    population = [generatePopulation(pop_size, pop_min, pop_max)]
    best_solution = max(population[0], key=calculateFitness)
    best_fitness = calculateFitness(best_solution)

    while curr_iter < max_iter and findOverallDistance(population[-1]) > min_overalldistance:
        curr_iter += 1
        current_pop = population[-1]
        
        # Select parents
        parent_pairs = selectParents(current_pop, len(current_pop))
        
        # Perform crossover
        offsprings = []
        for p in parent_pairs:
            new_off = crossover(p)
            offsprings.extend(new_off)
            
        # Perform mutation
        mutated_pop = [mutate(o, p_mutation) for o in offsprings]
        # Preserve the best solution found so far (elitism).
        candidate = max(mutated_pop, key=calculateFitness)
        candidate_fitness = calculateFitness(candidate)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        mutated_pop[0] = best_solution
        population.append(mutated_pop)

        if curr_iter % 10 == 0 or curr_iter == 1:
            best_in_gen = max(mutated_pop, key=calculateFitness)
            best_fit = calculateFitness(best_in_gen)
            print(f"Generation {curr_iter:02d} | Best Chromosome (side length x): {best_in_gen} cm | Fitness: {best_fit:.1f}")

    print("\n=====================================================")
    print(f" GA Optimisation Finished at Generation {curr_iter}")
    print(f" Best Side Length x = {best_solution} cm")
    print(f" Max Fitness (Paper Area Used) = {best_fitness:.1f} cm^2")
    print("=====================================================")


if __name__ == '__main__':
    main()
