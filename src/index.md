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
    Traditional CI labs evaluate algorithms using 2D mathematical functions or static plots. Integrating Isaac Sim lets students visualise computational-intelligence outputs in a 3D scene and, in selected examples, observe simple physics-driven proxies. These introductory scripts do not all model complete robot kinematics, sensors, or actuators.

### Practical Applications in the Labs

NVIDIA Isaac Sim examples are integrated into the Evolutionary Computation (EC) practicals:

*   **[Lab 3: Genetic Algorithms (GA)](lab3.md)**: Evolving planar waypoint trajectories around obstacles, then visualising the best trajectory in a 3D stage.
*   **[Lab 4: Particle Swarm Optimization (PSO)](lab4.md)**: Driving dynamic sphere proxies toward global-best coordinates ($p_g$) in a physics scene.
*   **[Lab 5: Ant Colony Optimization (ACO)](lab5.md)**: Route-finding on a discrete waypoint graph with pheromone decay and interpolated visual ant markers.

---

## System Requirements & Prerequisites

The following table summarizes the NVIDIA-published requirements for Isaac Sim
6.0 on x86-64 systems. Requirements change between releases, so check the
[official requirements page](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
and run the compatibility checker before installation.

| Requirement | Minimum | Good |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4080 | NVIDIA GeForce RTX 5080 |
| **GPU Memory (VRAM)** | 16 GB | 16 GB or more |
| **System RAM** | 32 GB | 64 GB |
| **Operating System** | Windows 11 or Ubuntu 22.04/24.04 | Windows 11 or Ubuntu 22.04/24.04 |
| **NVIDIA Driver** | Linux 580.95.05 / Windows 581.42 | Current NVIDIA-tested driver |
| **Storage** | 50 GB SSD | 500 GB SSD |

!!! note "No RTX GPU Available?"
    If your system does not meet the hardware requirements for Isaac Sim, **do not worry!** All practical scripts feature an automatic fallback mode that executes the algorithm logic and renders 2D animated visualizations using standard **Matplotlib** (see [Fallback Execution Mode](#fallback-execution-mode-cpu-matplotlib) below).

---

## NVIDIA Isaac Sim Installation & Setup Steps

The Omniverse Launcher was deprecated on 1 October 2025. Use NVIDIA's current
[workstation installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
instead:

1. Download the latest Isaac Sim workstation archive for Windows or Linux.
2. Extract it to a short, stable path such as `C:\isaacsim` or `~/isaacsim`.
3. Run `post_install.bat` on Windows or `./post_install.sh` on Linux.
4. Run the compatibility checker:
    - Windows: `isaac-sim.compatibility_check.bat`
    - Linux: `./isaac-sim.compatibility_check.sh`
5. Launch the application with `isaac-sim.bat` on Windows or
   `./isaac-sim.sh` on Linux.

Isaac Sim supplies its own Python 3.12 environment. Run standalone scripts with
`python.bat` on Windows or `./python.sh` on Linux. Install extra packages into
that environment; for example:

=== "Windows"

    ```cmd
    C:\isaacsim\python.bat -m pip install scikit-fuzzy
    ```

=== "Linux"

    ```bash
    ~/isaacsim/python.sh -m pip install scikit-fuzzy
    ```

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

Run the scripts from the Isaac Sim installation directory using its bundled
Python launcher:

=== "Windows (CMD / PowerShell)"

    Open Command Prompt or PowerShell and navigate to the repository root directory:

    ```cmd
    cd "path\to\ci-labs"
    ```

    Execute the desired practical script using `python.bat`:

    ```cmd
    set REPO=C:\path\to\ci-labs

    C:\isaacsim\python.bat "%REPO%\src\files\isaac_hexapod_drl.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_fuzzy_robot.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_ga_robot.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_pso_swarm.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_aco_route.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_vision_classifier.py"
    C:\isaacsim\python.bat "%REPO%\src\files\isaac_vision_detection.py"
    ```

=== "Linux (Terminal)"

    Open Terminal and navigate to the repository root directory:

    ```bash
    cd /path/to/ci-labs
    ```

    Execute the desired practical script using `python.sh`:

    ```bash
    REPO=/path/to/ci-labs

    ~/isaacsim/python.sh "$REPO/src/files/isaac_hexapod_drl.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_fuzzy_robot.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_ga_robot.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_pso_swarm.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_aco_route.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_vision_classifier.py"
    ~/isaacsim/python.sh "$REPO/src/files/isaac_vision_detection.py"
    ```

---

## Fallback Execution Mode (CPU / Matplotlib)

If Isaac Sim is not installed, the practical scripts detect the missing
environment and run a standalone fallback. These fallbacks demonstrate the
algorithm or data flow; they are not substitutes for Isaac Sim physics,
sensors, policy training, or real object-detection inference.

Simply run the script with your standard Python interpreter:

```bash
# Run using standard Python (Matplotlib 2D visualization / Standalone Mode)
python3 src/files/isaac_hexapod_drl.py
python3 src/files/isaac_fuzzy_robot.py
python3 src/files/isaac_ga_robot.py
python3 src/files/isaac_pso_swarm.py
python3 src/files/isaac_aco_route.py
python3 src/files/isaac_vision_classifier.py
python3 src/files/isaac_vision_detection.py
```

The core computational-intelligence exercises remain usable without NVIDIA
hardware. Sections that depend on physics or sensors should be completed on a
compatible lab workstation.

## Run Every Standard-Python Example

From the repository root, run these non-Isaac Sim examples with the activated
course environment. Execute the commands individually; close each plot window
before starting the next plotting example.

=== "Windows (CMD / PowerShell)"

    ```cmd
    python src\files\ann_hyperplane.py
    python src\files\ann_wine_mlp.py
    python src\files\lab1.py
    python src\files\lab2_fuzzy.py
    python src\files\lab3_ga.py
    python src\files\lab4_pso.py
    python src\files\lab5_aco.py
    python src\files\lab8a_keras_cnn.py
    python src\files\lab8b_keras_lstm.py
    python src\files\vis.py
    ```

=== "Linux (Terminal)"

    ```bash
    python3 src/files/ann_hyperplane.py
    python3 src/files/ann_wine_mlp.py
    python3 src/files/lab1.py
    python3 src/files/lab2_fuzzy.py
    python3 src/files/lab3_ga.py
    python3 src/files/lab4_pso.py
    python3 src/files/lab5_aco.py
    python3 src/files/lab8a_keras_cnn.py
    python3 src/files/lab8b_keras_lstm.py
    python3 src/files/vis.py
    ```

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
