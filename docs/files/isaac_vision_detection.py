# Copyright Author: Dr Tang Tiong Yew
"""
Real-Time Stream Capture and Deep Learning Object Detection in NVIDIA Isaac Sim
================================================================================
This script demonstrates continuous synthetic camera video stream capture in NVIDIA Isaac Sim,
performing deep learning object detection (MobileNet SSD / OpenCV DNN) frame-by-frame.

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics & visual camera stream):
   Run with Isaac Sim's standalone python:
   `isaac-sim.standalone.bat python src/files/isaac_vision_detection.py`
   OR `python.bat src/files/isaac_vision_detection.py`

2. OpenCV Standalone Fallback Mode (Object Detection Stream Simulation):
   `python src/files/isaac_vision_detection.py`
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
            print("[INFO] MobileNet SSD caffe model files not found locally. Using heuristic/dummy detection parser.")

    print("[INFO] Starting real-time Isaac Sim image capture & detection loop...")

    frame_count = 0
    frame_delay = 1.0 / visualization_fps if visualization_fps > 0 else 0.0
    while simulation_app.is_running() and frame_count < max_frames:
        world.step(render=True)

        rgba_frame = camera.get_rgba()
        if rgba_frame is None or rgba_frame.size == 0:
            continue

        rgb_frame = rgba_frame[:, :, :3]
        (h, w) = rgb_frame.shape[:2]

        if net is not None and HAS_OPENCV:
            blob = cv2.dnn.blobFromImage(cv2.resize(rgb_frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
                    print(f"[Frame {frame_count:03d}] Detected {label} at [{startX}, {startY}, {endX}, {endY}]")
        else:
            # Simulated object detection logging
            if frame_count % 15 == 0:
                dummy_idx = (frame_count // 15) % (len(CLASSES) - 1) + 1
                conf = 0.82 + (frame_count % 10) * 0.01
                bbox = [50, 60, 200, 220]
                print(f"[Frame {frame_count:03d}] Detected {CLASSES[dummy_idx]}: {conf*100:.1f}% at bbox={bbox}")

        frame_count += 1

        # Let the viewport render at a human-observable pace instead of
        # completing the demonstration as fast as the hardware allows.
        if frame_delay:
            time.sleep(frame_delay)

    print("[SUCCESS] Completed real-time simulation object detection loop.")
    simulation_app.close()


# =====================================================================
# 2. Standalone Fallback Execution
# =====================================================================
def run_fallback_detection(max_frames=50):
    """Fallback simulation running continuous video stream detection on synthetic frames."""
    print("=======================================================================")
    print(" Running Standalone Real-Time Object Detection (No Isaac Sim GUI)     ")
    print("=======================================================================")

    for frame_count in range(max_frames):
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        (h, w) = synthetic_frame.shape[:2]

        if frame_count % 10 == 0:
            target_class = CLASSES[(frame_count // 10) % len(CLASSES)]
            confidence = 0.75 + (frame_count % 5) * 0.04
            startX, startY = 100 + frame_count * 2, 80 + frame_count
            endX, endY = startX + 150, startY + 120
            
            if HAS_OPENCV:
                cv2.rectangle(synthetic_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(synthetic_frame, f"{target_class}: {confidence*100:.1f}%", 
                            (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"[Frame {frame_count:03d}] Object Detected: {target_class:<12} | Confidence: {confidence*100:.1f}% | BBox: [{startX}, {startY}, {endX}, {endY}]")

    print("[SUCCESS] Standalone real-time object detection stream simulation finished cleanly.")


if __name__ == '__main__':
    if HAS_ISAAC_SIM:
        print("[INFO] NVIDIA Isaac Sim detected. Starting stream capture loop...")
        run_isaac_sim_detection()
    else:
        print("[INFO] NVIDIA Isaac Sim environment not detected. Running Standalone Mode.")
        run_fallback_detection()
