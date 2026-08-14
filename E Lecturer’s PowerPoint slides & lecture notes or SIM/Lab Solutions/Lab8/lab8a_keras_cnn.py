# Copyright Author: Dr Tang Tiong Yew
"""
Lab 8a: Artificial Neural Networks using Keras (Image Classification)
======================================================================
This script provides the complete solution for Lab 8a exercises, including:
1. CIFAR-10 image dataset loading, pixel normalization, and train/val split
2. Sample image visualization grid
3. 3-layer Convolutional Neural Network (CNN) construction using tf.keras
4. Model compilation with Adam optimizer and SparseCategoricalCrossentropy
5. Model training for 5 epochs and accuracy evaluation
6. Test set predictions & visual grid of predicted vs ground-truth labels

Execution Mode:
`python3 src/files/lab8a_keras_cnn.py`
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    HAS_TF = True
except ImportError:
    HAS_TF = False


def main():
    print("=========================================================")
    print(" Lab 8a: Image Classification Using Keras (CIFAR-10)     ")
    print("=========================================================")

    if not HAS_TF:
        print("[!] TensorFlow / Keras not found. Install via: pip install tensorflow scikit-learn")
        return

    # 1. Load and Preprocess Data
    print("\n[1] Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    print("    Normalizing pixel values to range [0.0, 1.0]...")
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print("    Splitting training data into train (80%) and validation (20%) sets...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels.ravel(),
    )

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Visualize Sample Images
    print("    Displaying sample training images grid...")
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i])
        plt.title(class_names[train_labels[i][0]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 3. Build 3-layer CNN Architecture
    print("\n[2] Constructing 3-Layer Convolutional Neural Network (CNN)...")
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()

    # 4. Compile Model
    print("\n[3] Compiling Keras model...")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # 5. Train Model
    print("\n[4] Training model for 5 epochs...")
    history = model.fit(
        train_images, train_labels,
        epochs=5,
        validation_data=(val_images, val_labels),
        batch_size=32
    )

    # 6. Evaluate Model
    print("\n[5] Evaluating model performance on test set...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\n---> Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Plot Accuracy & Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Make & Visualize Predictions
    print("\n[6] Generating predictions on test dataset...")
    predictions = model.predict(test_images)
    pred_classes = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(25):
        axes[i].imshow(test_images[i])
        pred_label = class_names[pred_classes[i]]
        true_label = class_names[test_labels[i][0]]
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n[SUCCESS] Lab 8a Keras CNN pipeline completed successfully.")


if __name__ == '__main__':
    main()
