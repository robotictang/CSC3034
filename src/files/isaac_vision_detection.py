# Copyright Author: Dr Tang Tiong Yew
r"""
Real-Time Stream Capture and Deep Learning Object Detection in NVIDIA Isaac Sim
================================================================================
This script demonstrates camera capture and optional MobileNet SSD inference.
Real detections require local Caffe model files; fabricated detections are never reported.

Execution Modes:
1. NVIDIA Isaac Sim Mode (3D rendering and camera stream capture):
   Run with Isaac Sim's standalone python:
   Windows: `C:\isaacsim\python.bat src\files\isaac_vision_detection.py`
   Linux: `~/isaacsim/python.sh src/files/isaac_vision_detection.py`

2. OpenCV Standalone Fallback Mode (Object Detection Stream Simulation):
   `python3 src/files/isaac_vision_detection.py`
"""

import sys
import time
import numpy as np

# Try importing OpenCV and TensorFlow
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

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


CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]


# =====================================================================
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_detection(max_frames=100, visualization_fps=2):
    """Continuous camera stream capture & object detection in NVIDIA Isaac Sim."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    # Isaac Sim 6 moved the legacy ``omni.isaac`` modules under
    # ``isaacsim``.  These APIs remain supported by the installed version.
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera

    world = World()

    # Avoid ``add_default_ground_plane()``, which references a remote Isaac
    # asset.  A native USD plane keeps this example self-contained when the
    # asset server is unavailable.
    from pxr import UsdGeom

    ground_plane = UsdGeom.Plane.Define(world.stage, "/World/GroundPlane")
    ground_plane.CreateAxisAttr("Z")
    ground_plane.CreateWidthAttr(20.0)
    ground_plane.CreateLengthAttr(20.0)

    camera = Camera(
        prim_path="/World/RobotCamera",
        position=np.array([3.0, 3.0, 2.0]),
        resolution=(640, 480)
    )
    camera.initialize()
    world.reset()

    # Attempt loading MobileNet SSD Caffe model if files exist
    net = None
    if HAS_OPENCV:
        try:
            net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
            print("[INFO] OpenCV MobileNet SSD network loaded successfully.")
        except Exception:
            print("[WARN] MobileNet SSD files not found; frames will be captured without detection.")

    print("[INFO] Starting real-time Isaac Sim image capture & detection loop...")

    frame_count = 0
    empty_frame_attempts = 0
    frame_delay = 1.0 / visualization_fps if visualization_fps > 0 else 0.0
    while simulation_app.is_running() and frame_count < max_frames:
        world.step(render=True)

        rgba_frame = camera.get_rgba()
        if rgba_frame is None or rgba_frame.size == 0:
            empty_frame_attempts += 1
            if empty_frame_attempts >= 300:
                print("[WARN] Camera produced no frames after 300 render steps; stopping.")
                break
            continue
        empty_frame_attempts = 0

        rgb_frame = rgba_frame[:, :, :3]
        (h, w) = rgb_frame.shape[:2]

        if net is not None and HAS_OPENCV:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(rgb_frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5,
                swapRB=True,
            )
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    if 0 <= idx < len(CLASSES):
                        label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
                        print(f"[Frame {frame_count:03d}] Detected {label} at [{startX}, {startY}, {endX}, {endY}]")
        elif frame_count % 15 == 0:
            print(f"[Frame {frame_count:03d}] Captured frame; detection skipped (model unavailable).")

        frame_count += 1

        # Let the viewport render at a human-observable pace instead of
        # completing the demonstration as fast as the hardware allows.
        if frame_delay:
            time.sleep(frame_delay)

    print("[SUCCESS] Completed real-time camera capture loop.")
    simulation_app.close()


# =====================================================================
# 2. Standalone Fallback Execution
# =====================================================================
def run_fallback_detection(max_frames=50):
    """Generate synthetic frames to test stream plumbing without claiming detections."""
    print("=======================================================================")
    print(" Running Standalone Real-Time Object Detection (No Isaac Sim GUI)     ")
    print("=======================================================================")

    for frame_count in range(max_frames):
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        (h, w) = synthetic_frame.shape[:2]

        if frame_count % 10 == 0:
            mean_value = float(synthetic_frame.mean())
            print(f"[Frame {frame_count:03d}] Synthetic frame captured | Mean intensity: {mean_value:.1f}")

    print("[SUCCESS] Standalone stream plumbing test finished; no detector was run.")


if __name__ == '__main__':
    if HAS_ISAAC_SIM:
        print("[INFO] NVIDIA Isaac Sim detected. Starting stream capture loop...")
        run_isaac_sim_detection()
    else:
        print("[INFO] NVIDIA Isaac Sim environment not detected. Running Standalone Mode.")
        run_fallback_detection()
