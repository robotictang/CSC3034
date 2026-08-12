# Lab 8a: ANN using Keras
## Image Classification Using Keras

We are using Tensorflow for this lab. Install it using: `pip install tensorflow`


The libraries used in this lab:
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
```

This section will go through how to build and train a simple image classification model using Keras.

```mermaid
graph TD
    Data["1. Load CIFAR-10 Dataset (50,000 train, 10,000 test)"] --> Norm["2. Normalize Pixel Values (0.0 to 1.0)"]
    Norm --> Split["3. Train/Val Split (80% Train, 20% Val)"]
    Split --> Arch["4. Define 3-Layer CNN Architecture"]
    Arch --> Comp["5. Compile Model (Adam, SparseCategoricalCrossentropy)"]
    Comp --> Fit["6. Train Model (model.fit: 5 Epochs, batch size 8)"]
    Fit --> Eval["7. Evaluate Accuracy & Loss on Test Data"]
    Eval --> Predict["8. Generate Predictions & Visualise Grid"]
```

### Load and Preprocess Data

We’ll use the [CIFAR-10 dataset](https://keras.io/api/datasets/cifar10/), which is included in Keras. Load the dataset into the training and testing sets using the code below.

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
```

Normalize data by scaling pixel values to be between 0 and 1. There are several benefits to normalizing data, with the most important being that it prevents any specific range of values from dominating the learning process.

```python
train_images, test_images = train_images / 255.0, test_images / 255.0
```

Split the training data into training and validation sets.

```python
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)
```

Let's define label for data visualization. The class label is not used during training. It is simply a name that indicates what the class number represents.

```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

Plot some of the images to see how the data look like.

```python
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis("off")
```

### Build the Model

We are building a 3-layers Convolutional Neural Network (CNN). A typical convolutional block had a convolutinal layer, activation function, pooling operator. 

The convolutional layer is responsible for feature extraction, while the associated activation function introduces non-linearity. The pooling layer reduces the spatial dimensions of the feature map.

The last layer is the output layer, with the number of neurons in the dense layer. In this case, there are 10 neurons. This is typically used for a classification task with 10 classes (e.g., digits 0-9 in digit classification). This layer converts the logits into probabilities. The Softmax function normalizes the output so that the sum of all probabilities is 1, making it easier to interpret the model's predictions.

```mermaid
graph LR
    Input["Input Image (32x32x3)"] --> C1["Conv2D (32, 3x3, ReLU)"]
    C1 --> P1["MaxPool2D (2x2)"]
    P1 --> C2["Conv2D (64, 3x3, ReLU)"]
    C2 --> P2["MaxPool2D (2x2)"]
    P2 --> C3["Conv2D (64, 3x3, ReLU)"]
    C3 --> Flat["Flatten Layer"]
    Flat --> D1["Dense Layer (64, ReLU)"]
    D1 --> Out["Dense Output (10, Softmax)"]
```

```python
model = models.Sequential([
    # convolutional block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
     # convolutional block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # reshape feature map to 1D
    layers.Flatten(),
    # fully-connected layer
    layers.Dense(64, activation='relu'),
    # output layer
    layers.Dense(10, activation='softmax')
])
```

### Train the Model

`model.compile` is used to configure the model for training.
  - The Adam optimizer is an adaptive learning rate optimization algorithm that’s widely used in training deep learning models. 
  - Sparse Categorical Crossentropy loss function is used for multi-class classification problems where the target labels are integers (not one-hot encoded). It measures the difference between the true labels and the predicted probabilities. logits is the model's output.
    - from_logits=True: This parameter indicates that the model’s output is not a probability distribution (i.e., the output layer does not use a softmax activation function).
  - Accuracy Metric specifies that the model's performance will be evaluated using accuracy, which is the proportion of correctly predicted instances out of the total instances. It's a common metric for classification tasks.

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

`model.fit` trains the model on the provided training data.
  - train_images: The input images for training.
  - train_labels: The corresponding labels for the training images.
  - epochs: The number of times the model will iterate over the entire training dataset. In this case, the model will train for 5 epochs.
  - validation_data=(test_images, test_labels): This tuple provides the validation data, which is used to evaluate the model's performance on unseen data after each epoch. It consists of:
    - test_images: The input images for validation.
    - test_labels: The corresponding labels for the validation images.
    - batch_size: The number of samples per gradient update. The training data will be divided into batches of 8 samples, and the model's weights will be updated after each batch.

```python
history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels), batch_size=8)
```

### Evaluate the Model

`model.evaluate` evaluates the performance of the trained model on the test dataset.
  - test_images: The images from the test dataset.
  - test_labels: The corresponding labels for the test images.
  - verbose=2: This parameter controls the verbosity mode. A value of 2 means that the function will print one line per epoch.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

### Make Predictions

`model.predict(test_images)` takes the test images as input and outputs the predicted probabilities for each class.
  - predictions: This variable stores the predicted probabilities for each test image. Each element in predictions is an array of probabilities corresponding to the different classes.

```python
predictions = model.predict(test_images)
```

### Visualise Predictions


```python
pred_classes = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(test_images[i])
    axes[i].set_title("Predict: %s \nTrue: %s" % (class_names[np.argmax(test_labels[i])], class_names[pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
```

---

You can download the full Python script here: [lab8a_keras_cnn.py](files/lab8a_keras_cnn.py)

---


## NVIDIA Isaac Sim Example: Virtual Camera Image Capture and Deep Learning Classification

NVIDIA Isaac Sim (built on NVIDIA Omniverse) allows developers to simulate high-fidelity virtual environments with physical sensors, such as RGB cameras, depth sensors, and LiDARs. In this section, we demonstrate how to capture synthetic images from a virtual camera inside NVIDIA Isaac Sim and pass them to a Keras deep learning model for classification.

You can download the full Python script here: [isaac_vision_classifier.py](files/isaac_vision_classifier.py)

### Prerequisites & Dependencies

To run Isaac Sim Python standalone scripts, you need NVIDIA Isaac Sim installed along with its Python environment (`omni.isaac.sensor`, `omni.isaac.core`, `omni.isaac.kit`).

### Step 1: Initialize Isaac Sim and Virtual Camera Sensor

We create a virtual camera sensor attached to a scene containing simulation target objects.

```python
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim in non-headless mode (set headless=True for background runs)
simulation_app = SimulationApp({"headless": False})

import numpy as np
import tensorflow as tf
from omni.isaac.core import World
from omni.isaac.sensor import Camera

# Initialize the simulation world
world = World()
world.scene.add_default_ground_plane()

# Create a virtual camera sensor positioned in the stage looking at the target area
camera = Camera(
    prim_path="/World/RGB_Camera",
    position=np.array([2.0, 2.0, 1.5]),
    target=np.array([0.0, 0.0, 0.5]),
    resolution=(224, 224)  # Match input size for deep learning model
)

# Initialize camera sensor
camera.initialize()
world.reset()
```

### Step 2: Capture Synthetic Image Frame

Step the simulation world physics to render the scene, then retrieve the captured RGBA frame from the virtual camera sensor.

```python
# Advance simulation step to render scene
world.step(render=True)

# Capture RGBA image from the virtual camera (shape: 224x224x4, uint8)
rgba_data = camera.get_rgba()

# Extract RGB channels (drop alpha channel)
rgb_image = rgba_data[:, :, :3]
print(f"Captured synthetic image shape from Isaac Sim: {rgb_image.shape}")
```

### Step 3: Deep Learning Image Classification Pipeline

Now preprocess the synthetic RGB frame captured from Isaac Sim and pass it through a trained Keras CNN model or pre-trained network (e.g., MobileNetV2 / custom CNN) to classify objects in the virtual scene.

```python
# Preprocess image for Deep Learning model
img_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
img_tensor = tf.expand_dims(img_tensor / 255.0, axis=0)  # Normalize pixel values [0, 1] and add batch dimension

# Load or use trained Keras model (e.g., model from earlier section or pre-trained classifier)
# Here we use MobileNetV2 as a representative image classification model
classification_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Run inference on captured Isaac Sim image
predictions = classification_model.predict(img_tensor)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

print("\n--- Deep Learning Detection / Classification Results ---")
for class_id, class_name, score in decoded_predictions:
    print(f"Detected Object: {class_name} | Confidence: {score * 100:.2f}%")

# Close Isaac Sim application cleanly when done
simulation_app.close()
```

---

<footer style="text-align: center; margin-top: 2em; opacity: 0.8; font-size: 0.85em;">
Copyright Author: Dr Tang Tiong Yew
</footer>
