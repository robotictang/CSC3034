# Copyright Author: Dr Tang Tiong Yew
"""
Lab 1: Refresh on Python
========================
This script provides the complete solution for Lab 1 exercises, including:
1. Amoeba population calculation & sequence generation
2. Scatter plot of amoeba sequence (Month 0 to Month 100)
3. Fibonacci ratio analysis (approaching the Golden Ratio)
4. Coordinate sequence generation & Golden Spiral plotting with Matplotlib Arc
5. Random selection & coin tossing simulation (50-50, biased, and 3-option choice)

Execution Mode:
`python3 src/files/lab1.py`
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


def numberofamoeba(month):
    """Calculates the number of amoeba at the beginning of the given month."""
    if month == 0:
        return 1
    if month == 1:
        return 1
    a, b = 1, 1
    for _ in range(2, month + 1):
        a, b = b, a + b
    return b


def numberofamoebaseq(month):
    """Returns the sequence of amoeba numbers from Month 0 up to the given month."""
    if month == 0:
        return [1]
    seq = [1, 1]
    for i in range(2, month + 1):
        seq.append(seq[-1] + seq[-2])
    return seq[:month + 1]


def generatecoordinatesfromseries(series):
    """Generates spiral coordinates based on Fibonacci series and sign sequence (+,+), (+,-), (-,-), (-,+)."""
    coords = [[0, 0]]
    signs = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
    for i, num in enumerate(series):
        sign = signs[i % 4]
        prev = coords[-1]
        coords.append([prev[0] + sign[0] * num, prev[1] + sign[1] * num])
    return coords


def generatecenters(coordinates):
    """Generates centers of arcs from the sequence of coordinates."""
    centers = []
    for i, coord in enumerate(coordinates):
        if i == 0:
            centers.append([coord[0], coord[1]])
        elif i == 1:
            centers[-1][0] = coord[0]
        else:
            centers.append([centers[-1][0], centers[-1][1]])
            if i % 2 == 0:
                centers[-1][1] = coord[1]
            else:
                centers[-1][0] = coord[0]
    return centers


def plotspiral(axis, series, centers):
    """Plots the golden spiral arcs using matplotlib Arc patch."""
    angle = 90
    for number, center in zip(series, centers):
        arc = Arc(
            xy=center,
            width=2 * number,
            height=2 * number,
            angle=angle,
            theta1=0,
            theta2=90,
            color='crimson',
            linewidth=1.5
        )
        axis.add_patch(arc)
        angle -= 90


def tossCoin(prob_head=0.5):
    """Simulates a coin toss event given head probability."""
    if random.random() < prob_head:
        return 'head'
    else:
        return 'tail'


def chooseFromThree():
    """Simulates random selection out of 3 options: A (20%), B (50%), C (30%)."""
    r = random.random()
    if r < 0.2:
        return 'Option A'
    elif r < 0.7:
        return 'Option B'
    else:
        return 'Option C'


def main():
    print("=====================================================")
    print(" Lab 1: Refresh on Python - Exercise Solutions      ")
    print("=====================================================")

    # 1. Amoeba Community
    print("\n--- 1. Amoeba Community ---")
    month_val = 6
    print(f"Number of amoeba at Month {month_val}: {numberofamoeba(month_val)}")
    print(f"Sequence up to Month 4: {numberofamoebaseq(4)}")

    # Scatter plot for Month 0 to Month 100
    n_months = 100
    seq_100 = numberofamoebaseq(n_months)
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(seq_100)), seq_100, color='teal', s=15)
    plt.title("Amoeba Population Growth (Month 0 to Month 100)")
    plt.xlabel("Month")
    plt.ylabel("Number of Amoebas")
    plt.yscale("log")  # Log scale due to rapid Fibonacci growth
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Fibonacci and Golden Ratio
    print("\n--- 2. Fibonacci and Golden Ratio ---")
    fib_seq = numberofamoebaseq(15)
    ratios = [fib_seq[i+1] / fib_seq[i] for i in range(len(fib_seq) - 1)]
    golden_ratio = (1 + 5**0.5) / 2
    print(f"Fibonacci Sequence: {fib_seq}")
    print(f"Consecutive Ratios: {[round(r, 4) for r in ratios]}")
    print(f"Golden Ratio Target (phi): {golden_ratio:.5f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(ratios) + 1), ratios, 'o-', color='darkorange', label='Fibonacci Ratio')
    plt.axhline(y=golden_ratio, color='navy', linestyle='--', label=f'Golden Ratio ({golden_ratio:.4f})')
    plt.title("Ratio of Consecutive Fibonacci Numbers -> Golden Ratio")
    plt.xlabel("Term Index")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Random Selection based on Probability
    print("\n--- 3. Random Selection based on Probability ---")
    fair_tosses = [tossCoin(0.5) for _ in range(10)]
    biased_tosses = [tossCoin(0.2) for _ in range(10)]
    option_choices = [chooseFromThree() for _ in range(10)]
    print(f"10 Fair Coin Tosses (50/50): {fair_tosses}")
    print(f"10 Biased Coin Tosses (20/80 Head/Tail): {biased_tosses}")
    print(f"10 Random 3-Option Choices: {option_choices}")

    # 4. Golden Spiral Arc Plot
    print("\n--- 4. Additional: Plot Arc to Form Golden Spiral ---")
    n = 10  # Spiral for length <= 93
    seq = numberofamoebaseq(n)
    coords = generatecoordinatesfromseries(seq)
    centers = generatecenters(coords)

    fig, ax = plt.subplots(figsize=(7, 7))
    coords_np = np.array(coords)
    ax.plot(coords_np[:, 0], coords_np[:, 1], 'o--', color='gray', alpha=0.5, label='Waypoint Line')
    plotspiral(ax, seq, centers)

    ax.set_aspect('equal')
    ax.set_title("Golden Spiral Generated with Matplotlib Arc Patches")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n[SUCCESS] Lab 1 exercises executed successfully.")


if __name__ == '__main__':
    main()
