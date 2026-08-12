# Copyright Author: Dr Tang Tiong Yew
"""
Lab 6: Artificial Neural Networks (Supervised Learning)
=======================================================
This script demonstrates multi-layer perceptron (MLP) classification using scikit-learn
on the Wine dataset, including data normalization, training, predictions, evaluation,
and visualization of the neural network architecture and weights.

Execution Mode:
`python src/files/ann_wine_mlp.py`
"""

import sys
import numpy as np
import pandas as pd

# Check for required machine learning and plotting libraries
try:
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    MISSING_PKG = str(e)


def visualise(mlp):
    """
    Visualises the neural network structure and connection weights using Matplotlib.
    Neuron positions are plotted as nodes and connection weights as line thickness.
    """
    # Get number of neurons in each layer
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    # Calculate coordinates of each neuron on the graph
    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [[[l, (n + 1) * (y_range[1] / (layer + 1))] for n in range(layer)] for l, layer in enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x, y in layer]
    y_neurons = [y for layer in loc_neurons for x, y in layer]

    # Identify range of connection weights
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    # Prepare figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Neural Network Structure & Connection Weights")
    ax.set_facecolor("#f4f4f6")

    # Draw neurons
    ax.scatter(x_neurons, y_neurons, s=150, c="#007acc", zorder=5, edgecolors="black")

    # Draw connection lines with width proportional to weight
    for l, layer in enumerate(mlp.coefs_):
        for i, neuron in enumerate(layer):
            for j, w in enumerate(neuron):
                norm_w = (w - weight_range[0]) / (weight_range[1] - weight_range[0] + 1e-9)
                ax.plot(
                    [loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],
                    [loc_neurons[l][i][1], loc_neurons[l + 1][j][1]],
                    color="white",
                    linewidth=(norm_w * 5 + 0.2) * 1.2,
                    zorder=3
                )
                ax.plot(
                    [loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],
                    [loc_neurons[l][i][1], loc_neurons[l + 1][j][1]],
                    color="#555555",
                    linewidth=norm_w * 5 + 0.2,
                    zorder=4
                )

    ax.set_xticks(range(len(n_neurons)))
    ax.set_xticklabels([f"Layer {i+1}" for i in range(len(n_neurons))])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    if not HAS_DEPENDENCIES:
        print(f"[!] Missing required library: {MISSING_PKG}")
        print("    Please install dependencies via: pip install scikit-learn pandas matplotlib")
        return

    print("==================================================================")
    print(" Lab 6: Artificial Neural Networks - Wine Dataset Classification ")
    print("==================================================================")

    # 1. Load data
    print("\n[1] Loading Wine dataset...")
    data = datasets.load_wine()
    
    # 2. Examine dataset with pandas summary statistics
    print("\n[2] Statistical Summary of Wine Dataset:")
    wine = pd.DataFrame(data.data, columns=data.feature_names)
    wine['target'] = data.target
    print(wine.describe().transpose()[['mean', 'std', 'min', '50%', 'max']])

    # 3. Split dataset (80% train, 20% test)
    print("\n[3] Splitting data into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=0.8, random_state=42
    )

    # 4. Standardise features using StandardScaler
    print("\n[4] Normalizing feature ranges with StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nScaled X_train Summary Statistics:")
    print(pd.DataFrame(X_train, columns=data.feature_names).describe().transpose()[['mean', 'std', 'min', 'max']])

    # 5. Construct ANN model (1 hidden layer of 2 neurons, max_iter=1000)
    print("\n[5] Constructing MLPClassifier (hidden_layer_sizes=(2,), max_iter=1000)...")
    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)

    # 6. Initial partial fit to visualize weights before full training
    print("\n[6] Visualising initial network weights after 1 epoch (partial_fit)...")
    mlp.partial_fit(X_train, y_train, classes=np.unique(data.target))
    visualise(mlp)

    # 7. Train the model fully
    print("\n[7] Training ANN model on full training set...")
    mlp.fit(X_train, y_train)
    print(f"    Training completed in {mlp.n_iter_} iterations.")

    # 8. Model predictions
    print("\n[8] Running predictions on test set...")
    predictions = mlp.predict(X_test)

    # 9. Evaluate performance
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=data.target_names))

    # 10. Visualise final trained network weights
    print("\n[10] Visualising final trained network weights...")
    visualise(mlp)

    print("\n[SUCCESS] Lab 6 ANN classification pipeline completed successfully.")


if __name__ == '__main__':
    main()
