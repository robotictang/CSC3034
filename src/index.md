---
template: home.html
title: Overview
---

<h1 style="text-align: center">CSC3034 Computational Intelligence</h1>

**This site hosts the lab sheets for the module of CSC3034 Computational Intelligence in the Department of Computing and Information Systems (DCIS) at Sunway University.**

## Aim

The aim of these labs is to guide students in implementing basic computational intelligence (CI) algorithms with and without Python libraries, bridging theoretical optimization concepts with real-world physical simulations.

## Information

The labs are designed to follow the schedule of the lectures; therefore, you will require the knowledge from previous lectures to conduct each lab session.

## Schedule 

*The schedule is subject to change.*

<div class="timeline">
    <div class="container right">
        <div class="date">Week 2</div>
        <div class="content"><a href="lab1.html">Lab 1</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 4</div>
        <div class="content"><a href="lab2.html">Lab 2</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 6</div>
        <div class="content"><a href="lab3.html">Lab 3</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 7</div>
        <div class="content"><a href="lab4.html">Lab 4</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 8</div>
        <div class="content"><a href="lab5.html">Lab 5</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 9</div>
        <div class="content"><a href="lab6.html">Lab 6</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 10</div>
        <div class="content"><a href="lab7.html">Lab 7</a></div>
    </div>
    <div class="container right">
        <div class="date">Week 12</div>
        <div class="content"><a href="lab8a.html">Lab 8a</a> &amp; <a href="lab8b.html">Lab 8b</a></div>
    </div>
</div>

---

## NVIDIA Isaac Sim Integration

In this module, advanced practical sessions incorporate **NVIDIA Isaac Sim**, a state-of-the-art, GPU-accelerated 3D robotics simulation platform built on **NVIDIA Omniverse** and **PhysX 5**. Powered by OpenUSD (Universal Scene Description), Isaac Sim enables photo-realistic rendering (RTX ray tracing) and precise physical dynamics for swarm robotics and autonomous systems.

!!! info "Why NVIDIA Isaac Sim in Computational Intelligence?"
    Traditional CI labs evaluate algorithms using 2D mathematical functions or static plots. Integrating Isaac Sim allows students to observe how metaheuristic search and swarm intelligence algorithms behave when deployed on **physical robotic agents** with kinematics, momentum, waypoints, and 3D spatial constraints.

### Practical Applications in the Labs

NVIDIA Isaac Sim examples are integrated into the Evolutionary Computation (EC) practicals:

*   **[Lab 3: Genetic Algorithms (GA)](lab3.md)**: Evolving 3D spatial waypoint trajectories for mobile robots navigating around obstacles in a bounded arena.
*   **[Lab 4: Particle Swarm Optimization (PSO)](lab4.md)**: Physical multi-robot swarm target search, driving dynamic robot prims toward global best coordinates ($p_g$) in real time.
*   **[Lab 5: Ant Colony Optimization (ACO)](lab5.md)**: Swarm route-finding on continuous waypoint graphs with dynamic pheromone decay and deposition mapped to physical robot velocities.

---

## System Requirements & Prerequisites

To run the NVIDIA Isaac Sim 3D physical simulation mode, your system should meet the following minimum requirements:

| Requirement | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 2060 (or RTX Quadro equivalent) | NVIDIA GeForce RTX 3070 / 4070 or higher |
| **GPU Memory (VRAM)** | 8 GB VRAM | 12 GB+ VRAM |
| **System RAM** | 16 GB RAM | 32 GB RAM |
| **Operating System** | Windows 10 / 11 (64-bit) or Ubuntu 20.04 / 22.04 LTS | Windows 11 (64-bit) or Ubuntu 22.04 LTS |
| **NVIDIA Driver** | Version 535.xx or newer (with CUDA 12.x support) | Latest Production Branch / Studio Driver |
| **Disk Space** | 50 GB free disk space | NVMe SSD with 100 GB free space |

!!! note "No RTX GPU Available?"
    If your system does not meet the hardware requirements for Isaac Sim, **do not worry!** All practical scripts feature an automatic fallback mode that executes the algorithm logic and renders 2D animated visualizations using standard **Matplotlib** (see [Fallback Execution Mode](#fallback-execution-mode-cpu-matplotlib) below).

---

## NVIDIA Isaac Sim Installation & Setup Steps

Follow these steps to set up NVIDIA Isaac Sim on your workstation:

### Step 1: Install NVIDIA Omniverse Launcher

1. Download the **NVIDIA Omniverse Launcher** from the official website:  
   [https://www.nvidia.com/en-us/omniverse/](https://www.nvidia.com/en-us/omniverse/)
2. Run the installer and sign in with your NVIDIA account.

### Step 2: Install Isaac Sim App

1. Open **Omniverse Launcher** and navigate to the **Exchange** tab.
2. Search for **Isaac Sim**.
3. Click **Install** (select the latest release, e.g., Isaac Sim 4.2.0 or 4.5.0).
4. Wait for the installation to complete. Note the installation path:
    *   **Windows**: `%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0`  
        *(e.g., `C:\Users\<YourUsername>\AppData\Local\ov\pkg\isaac-sim-4.2.0`)*
    *   **Linux**: `~/.local/share/ov/pkg/isaac-sim-4.2.0`

### Step 3: Verify Environment Setup

Isaac Sim provides a bundled Python environment wrapper script (`isaac-sim.standalone.bat` on Windows or `python.sh` on Linux) that automatically imports all required Omniverse, USD, and PhysX libraries.

Ensure you can access this wrapper executable from your terminal or add the Isaac Sim installation directory to your system environment variables.

---

## Running Practical Examples in Isaac Sim

The repository includes standalone scripts designed for Isaac Sim under the `src/files/` directory:

*   `src/files/isaac_hexapod_drl.py` (Lab 1)
*   `src/files/isaac_fuzzy_robot.py` (Lab 2)
*   `src/files/isaac_ga_robot.py` (Lab 3)
*   `src/files/isaac_pso_swarm.py` (Lab 4)
*   `src/files/isaac_aco_route.py` (Lab 5)
*   `src/files/isaac_vision_classifier.py` (Lab 8a)
*   `src/files/isaac_vision_detection.py` (Lab 8b)

Run the scripts using the Isaac Sim standalone Python launcher:

=== "Windows (CMD / PowerShell)"

    Open Command Prompt or PowerShell and navigate to the repository root directory:

    ```cmd
    cd "path\to\ci-labs"
    ```

    Execute the desired practical script using `isaac-sim.standalone.bat`:

    ```cmd
    :: Lab 1: Hexapod DRL Locomotion
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_hexapod_drl.py

    :: Lab 2: Fuzzy Logic Robot Controller
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_fuzzy_robot.py

    :: Lab 3: GA Trajectory Evolution
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_ga_robot.py

    :: Lab 4: PSO Swarm Simulation
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_pso_swarm.py

    :: Lab 5: ACO Swarm Route Finding
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_aco_route.py

    :: Lab 8a: Vision Classifier
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_vision_classifier.py

    :: Lab 8b: Vision Object Detector
    "%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2.0\isaac-sim.standalone.bat" python src/files/isaac_vision_detection.py
    ```

=== "Linux (Terminal)"

    Open Terminal and navigate to the repository root directory:

    ```bash
    cd /path/to/ci-labs
    ```

    Execute the desired practical script using `python.sh`:

    ```bash
    # Lab 1: Hexapod DRL Locomotion
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_hexapod_drl.py

    # Lab 2: Fuzzy Logic Robot Controller
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_fuzzy_robot.py

    # Lab 3: GA Trajectory Evolution
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_ga_robot.py

    # Lab 4: PSO Swarm Simulation
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_pso_swarm.py

    # Lab 5: ACO Swarm Route Finding
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_aco_route.py

    # Lab 8a: Vision Classifier
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_vision_classifier.py

    # Lab 8b: Vision Object Detector
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh src/files/isaac_vision_detection.py
    ```

---

## Fallback Execution Mode (CPU / Matplotlib)

If Isaac Sim is not installed or your system lacks an NVIDIA RTX GPU, all practical scripts automatically detect the missing environment (`isaacsim` / `omni` imports) and run in **Standalone Matplotlib Mode**.

Simply run the script with your standard Python interpreter:

```bash
# Run using standard Python (Matplotlib 2D visualization / Standalone Mode)
python src/files/isaac_hexapod_drl.py
python src/files/isaac_fuzzy_robot.py
python src/files/isaac_ga_robot.py
python src/files/isaac_pso_swarm.py
python src/files/isaac_aco_route.py
python src/files/isaac_vision_classifier.py
python src/files/isaac_vision_detection.py
```

This ensures that all students can complete the practical exercises and observe the algorithmic behavior regardless of hardware availability.

---

## NVIDIA Physical AI Learning Resources

For supplementary self-paced learning and deep dives into physical AI and robotics simulation, students are encouraged to explore the **[NVIDIA Physical AI Learning Portal](https://docs.nvidia.com/learning/physical-ai/)**.

Physical AI brings artificial intelligence into systems that perceive, reason about, and act in the physical world—including autonomous robots, smart sensors, and digital twins. Key self-paced learning modules and courses available on the portal include:

*   **[Getting Started with Isaac Sim](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/index.html)**: Building robots from scratch, configuring physics properties, adding sensors, and running simulations.
*   **[Getting Started with Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html)**: Reinforcement learning and GPU-accelerated policy training for thousands of robots in parallel.
*   **[Learn OpenUSD](https://docs.nvidia.com/learn-openusd/latest/index.html)**: Universal Scene Description (USD) fundamentals for 3D asset modularity and scene composition.
*   **[Sim-to-Real Transfer with SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)**: Transferring policies trained in simulation to physical robot arm hardware.
*   **[Getting Started with Isaac ROS](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-ros/latest/index.html)**: ROS 2 acceleration using NVIDIA NITROS for real-time perception and navigation systems.

!!! tip "NVIDIA Physical AI Learning Hub"
    Visit [https://docs.nvidia.com/learning/physical-ai/](https://docs.nvidia.com/learning/physical-ai/) for official NVIDIA courses, interactive cloud launchables (Brev), and developer certifications.

---

<footer style="text-align: center; margin-top: 2em; opacity: 0.8; font-size: 0.85em;">
Copyright Author: Dr Tang Tiong Yew
</footer>
