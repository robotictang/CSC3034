# Copyright Author: Dr Tang Tiong Yew
"""
Lab 7: Artificial Neural Networks (Hyperplane & Decision Boundary Visualisation)
================================================================================
This script demonstrates decision boundary / hyperplane visualization for Multi-Layer
Perceptron (MLP) classifiers on the Iris dataset using scikit-learn and Matplotlib.

Execution Mode:
`python3 src/files/ann_hyperplane.py`
"""

import sys
import os
import numpy as np

# Ensure helper modules in the same directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    import vis
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    MISSING_PKG = str(e)


def main():
    if not HAS_DEPENDENCIES:
        print(f"[!] Missing required dependency: {MISSING_PKG}")
        print("    Please install dependencies via: pip install scikit-learn matplotlib")
        return

    print("==================================================================")
    print(" Lab 7: Artificial Neural Networks - Hyperplane Visualisation     ")
    print("==================================================================")

    # 1. Load Iris Data
    print("\n[1] Preparing Iris dataset (2 input features: Sepal Width & Petal Length)...")
    iris = datasets.load_iris()
    X = [[d[1], d[2]] for d in iris.data]
    Y = iris.target

    # 2. Preprocess & Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. First Configuration: Basic MLP Classifier (2 hidden neurons)
    print("\n[2] Training initial MLPClassifier (hidden_layer_sizes=(2,), max_iter=1000)...")
    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    print("    Visualising initial 2D decision boundary area...")
    fig1 = plt.figure(figsize=(7, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_title("MLP Decision Boundary (1 Hidden Layer, 2 Neurons)")
    vis.vis2d(ax1, mlp, X_train, y_train, X_test, y_test)
    plt.tight_layout()
    plt.show()

    # 4. Compare Activation Functions
    print("\n[3] Comparing Activation Functions ('identity', 'logistic', 'tanh', 'relu')...")
    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    fig2 = plt.figure(figsize=(14, 4))
    fig2.suptitle("Decision Boundaries Across Different Activation Functions", fontsize=14)

    for i, actfcn in enumerate(activation_functions):
        mlp_act = MLPClassifier(hidden_layer_sizes=(3,), activation=actfcn, max_iter=1000, random_state=42)
        mlp_act.fit(X_train, y_train)
        ax = fig2.add_subplot(1, len(activation_functions), i + 1)
        ax.set_title(f"Activation: {actfcn}")
        vis.vis2d(ax, mlp_act, X_train, y_train, X_test, y_test)

    plt.tight_layout()
    plt.show()

    # 5. Grid Search: Activation Functions vs Hidden Layer Architectures
    print("\n[4] Evaluating combinations of Activation Functions and Layer Depths...")
    hidden_layers = [(3,), (3, 3), (3, 3, 3)]
    fig3 = plt.figure(figsize=(14, 9))
    fig3.suptitle("Hyperplane Grid: Activations vs Hidden Layer Architectures", fontsize=14)

    for i, actfcn in enumerate(activation_functions):
        for j, hlyr in enumerate(hidden_layers):
            mlp_grid = MLPClassifier(hidden_layer_sizes=hlyr, activation=actfcn, max_iter=1000, random_state=42)
            mlp_grid.fit(X_train, y_train)
            score = round(mlp_grid.score(X_test, y_test), 2)
            
            ax = fig3.add_subplot(len(hidden_layers), len(activation_functions), j * len(activation_functions) + i + 1)
            ax.set_title(f"{actfcn},{hlyr},Score:{score}", fontsize=10)
            vis.vis2d(ax, mlp_grid, X_train, y_train, X_test, y_test)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # 6. Multi-Feature Input Visualisation (All 4 Iris features)
    print("\n[5] Training model with all 4 Iris input features...")
    X_train4, X_test4, y_train4, y_test4 = train_test_split(iris.data, iris.target, train_size=0.8, random_state=42)
    scaler4 = StandardScaler()
    scaler4.fit(X_train4)
    X_train4 = scaler4.transform(X_train4)
    X_test4 = scaler4.transform(X_test4)

    mlp4 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, random_state=42)
    mlp4.fit(X_train4, y_train4)
    accuracy = mlp4.score(X_test4, y_test4)
    print(f"    Full 4-feature MLP Test Accuracy: {accuracy * 100:.2f}%")

    print("    Rendering parallel coordinates decision visualizer (vis3d)...")
    fig4 = plt.figure(figsize=(10, 6))
    fig4.suptitle("Parallel Coordinates Multi-Feature Decision Area (All 4 Iris Features)", fontsize=13)
    axes = vis.vis3d(fig4, mlp4, X_train4, y_train4, X_test4, y_test4)
    for i, a in enumerate(axes):
        a.set_title(f"Class: {iris.target_names[i]}")
        a.set_xticklabels([])
        a.get_yaxis().set_visible(False)
    axes[-1].set_xticklabels(iris.feature_names)
    plt.tight_layout()
    plt.show()

    print("\n[SUCCESS] Lab 7 Hyperplane visualisation pipeline completed successfully.")


if __name__ == '__main__':
    main()
