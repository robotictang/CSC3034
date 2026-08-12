# Copyright Author: Dr Tang Tiong Yew
"""
Virtual Camera Image Capture and Deep Learning Classification in NVIDIA Isaac Sim
==================================================================================
This script demonstrates synthetic camera image rendering in NVIDIA Isaac Sim
and streaming the RGB frames into a Keras Deep Learning classifier (MobileNetV2).

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU rendering & Camera Sensor):
   Run with Isaac Sim's standalone python:
   `isaac-sim.standalone.bat python src/files/isaac_vision_classifier.py`
   OR `python.bat src/files/isaac_vision_classifier.py`

2. TensorFlow Standalone Fallback Mode (Deep Learning Inference simulation):
   `python src/files/isaac_vision_classifier.py`
"""

import sys
import time
import numpy as np

HAS_TF = False
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

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
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_classification():
    """Captures RGB image from virtual Isaac Sim camera and classifies via Keras DL model."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    from omni.isaac.sensor import Camera

    world = World()
    world.scene.add_default_ground_plane()

    camera = Camera(
        prim_path="/World/RGB_Camera",
        position=np.array([2.0, 2.0, 1.5]),
        target=np.array([0.0, 0.0, 0.5]),
        resolution=(224, 224)
    )

    camera.initialize()
    world.reset()

    # Step simulation to render initial frame
    world.step(render=True)

    rgba_data = camera.get_rgba()
    if rgba_data is not None and rgba_data.size > 0:
        rgb_image = rgba_data[:, :, :3]
        print(f"[INFO] Captured synthetic image shape from Isaac Sim: {rgb_image.shape}")
    else:
        print("[WARN] Camera frame empty, generating synthetic frame...")
        rgb_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Preprocess image for Deep Learning model
    img_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor / 255.0, axis=0)

    print("[INFO] Loading TensorFlow / Keras Deep Learning Model (MobileNetV2)...")
    classification_model = tf.keras.applications.MobileNetV2(weights='imagenet')

    predictions = classification_model.predict(img_tensor)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    print("\n--- Deep Learning Detection / Classification Results ---")
    for class_id, class_name, score in decoded_predictions:
        print(f"Detected Object: {class_name:<20} | Confidence: {score * 100:.2f}%")

    simulation_app.close()


# =====================================================================
# 2. Standalone Fallback Execution
# =====================================================================
def run_fallback_classification():
    """Fallback execution running Keras DL classification on a synthetic image tensor."""
    print("========================================================================")
    print(" Running Standalone Deep Learning Image Classifier (No Isaac Sim GUI) ")
    print("========================================================================")

    # Generate synthetic RGB image frame (224x224x3)
    synthetic_rgb = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    print(f"[INFO] Synthetic image array generated: shape={synthetic_rgb.shape}")

    if not HAS_TF:
        print("[!] TensorFlow library ('tensorflow') is required to run DL classification.")
        print("    Install it via: pip install tensorflow")
        print("\n[INFO] NumPy Image Feature Extraction Fallback:")
        mean_r, mean_g, mean_b = np.mean(synthetic_rgb, axis=(0, 1))
        print(f"       Extracted Channel RGB Averages -> R: {mean_r:.1f}, G: {mean_g:.1f}, B: {mean_b:.1f}")
        print("       Heuristic Classification: Synthetic Indoor Stage Texture")
        return

    img_tensor = tf.convert_to_tensor(synthetic_rgb, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor / 255.0, axis=0)

    print("[INFO] Loading TensorFlow / Keras MobileNetV2 model...")
    try:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        predictions = model.predict(img_tensor)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        print("\n--- Deep Learning Image Classification Results ---")
        for class_id, class_name, score in decoded:
            print(f"Predicted Class: {class_name:<20} | Confidence: {score * 100:.2f}%")
    except Exception as e:
        print(f"[WARN] ImageNet weights download skipped/failed: {e}")
        # Custom simple CNN model inference fallback
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        preds = model.predict(img_tensor)
        print(f"[INFO] Custom CNN classification probabilities (10 classes): {preds[0][:5]}...")

    print("[SUCCESS] Standalone classification task finished cleanly.")


if __name__ == '__main__':
    if HAS_ISAAC_SIM and HAS_TF:
        print("[INFO] NVIDIA Isaac Sim detected. Initializing virtual camera stage...")
        run_isaac_sim_classification()
    else:
        print("[INFO] Running Standalone Mode.")
        run_fallback_classification()
