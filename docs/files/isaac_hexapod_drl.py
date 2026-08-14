# Copyright Author: Dr Tang Tiong Yew
r"""
Deep Reinforcement Learning for 6-Legged Walking Robot in NVIDIA Isaac Sim
==========================================================================
This script demonstrates the observation/action data flow of an actor-critic
locomotion policy. It does not train PPO or load a trained checkpoint.

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics & visual simulation):
   Run with Isaac Sim's standalone python:
   Windows: `C:\isaacsim\python.bat src\files\isaac_hexapod_drl.py`
   Linux: `~/isaacsim/python.sh src/files/isaac_hexapod_drl.py`

2. Standalone Fallback Mode (Policy & Math simulation):
   `python3 src/files/isaac_hexapod_drl.py`
"""

import sys
import numpy as np

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
# Actor-Critic Neural Network Policy Scaffold
# =====================================================================
if HAS_TORCH:
    class HexapodPolicy(nn.Module):
        def __init__(self, obs_dim=45, action_dim=18):
            super(HexapodPolicy, self).__init__()
            # Actor Network (Policy: outputs target joint angles)
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
                nn.Tanh()  # Output normalized actions in range [-1, 1]
            )
            # Critic Network (Value function baseline)
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 1)
            )
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, state):
            action_mean = self.actor(state)
            state_value = self.critic(state)
            return action_mean, state_value
else:
    class HexapodPolicy:
        """NumPy standalone fallback policy network when PyTorch is absent."""
        def __init__(self, obs_dim=45, action_dim=18):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            np.random.seed(42)
            self.W1 = np.random.randn(obs_dim, 64) * 0.1
            self.W2 = np.random.randn(64, action_dim) * 0.1

        def forward_numpy(self, obs):
            h = np.tanh(np.dot(obs, self.W1))
            action = np.tanh(np.dot(h, self.W2))
            return action


# =====================================================================
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_drl(num_episodes=5, max_steps=300):
    """Executes Hexapod Robot DRL inside NVIDIA Isaac Sim photorealistic environment."""
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except ImportError:
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})

    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicSphere
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:  # Isaac Sim 4.x compatibility
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicSphere
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction

    world = World()
    world.scene.add_default_ground_plane()

    # Load Isaac's Ant reference articulation. Replace this path with a
    # compatible hexapod USD for a true six-legged exercise.
    assets_root_path = get_assets_root_path()

    is_articulated = True
    try:
        if not assets_root_path:
            raise RuntimeError("Isaac asset root is unavailable")
        robot_usd_path = assets_root_path + "/Isaac/Robots/Ant/ant.usd"
        add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/LeggedRobot")
        hexapod_robot = world.scene.add(
            Robot(
                prim_path="/World/LeggedRobot",
                name="legged_robot",
                position=np.array([0.0, 0.0, 0.5])
            )
        )
    except Exception as e:
        print(f"[WARN] Could not load USD asset: {e}. Spawning base robot prim...")
        hexapod_robot = world.scene.add(
            DynamicSphere(
                prim_path="/World/LeggedRobot",
                name="legged_robot_marker",
                position=np.array([0.0, 0.0, 0.5]),
                radius=0.4
            )
        )
        is_articulated = False

    world.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
    if is_articulated:
        initial_joint_positions = np.asarray(hexapod_robot.get_joint_positions(), dtype=float)
        action_dim = len(initial_joint_positions)
    else:
        action_dim = 18
    obs_dim = 9 + 2 * action_dim
    policy = HexapodPolicy(obs_dim=obs_dim, action_dim=action_dim)
    if HAS_TORCH:
        policy = policy.to(device)
        policy.eval()

    print(f"[INFO] Actor-critic policy scaffold initialized on {device}; articulation DOF={action_dim}")
    warned_action_failure = False

    for episode in range(num_episodes):
        world.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            world.step(render=True)

            try:
                joint_positions = hexapod_robot.get_joint_positions()
                joint_velocities = hexapod_robot.get_joint_velocities()
                base_lin_vel = hexapod_robot.get_linear_velocity()
                base_ang_vel = hexapod_robot.get_angular_velocity()
            except Exception:
                joint_positions = np.zeros(action_dim)
                joint_velocities = np.zeros(action_dim)
                base_lin_vel = np.array([0.5, 0.01, 0.0])
                base_ang_vel = np.zeros(3)

            obs = np.concatenate([base_lin_vel, base_ang_vel, np.zeros(3), joint_positions, joint_velocities])
            if HAS_TORCH:
                state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_mean, _ = policy(state_tensor)
                    action = action_mean.squeeze(0).cpu().numpy()
            else:
                action = policy.forward_numpy(obs)

            if is_articulated:
                try:
                    hexapod_robot.apply_action(ArticulationAction(joint_positions=action))
                except Exception as exc:
                    if not warned_action_failure:
                        print(f"[WARN] Joint command could not be applied: {exc}")
                        warned_action_failure = True

            forward_vel = base_lin_vel[0]
            drift_penalty = abs(base_lin_vel[1])
            reward = forward_vel * 2.0 - drift_penalty * 0.5 - 0.01 * np.sum(np.square(action))
            episode_reward += reward

        print(f"Episode {episode + 1}/{num_episodes} - Total DRL Locomotion Reward: {episode_reward:.2f}")

    print("[SUCCESS] Completed actor-critic policy rollout in NVIDIA Isaac Sim.")
    simulation_app.close()


# =====================================================================
# 2. PyTorch / NumPy Standalone Fallback Execution
# =====================================================================
def run_fallback_drl(num_episodes=5, max_steps=300):
    """Exercise an untrained policy's tensor flow without Isaac Sim GUI."""
    print("==================================================================")
    print(" Running Standalone Hexapod DRL Policy Simulation (No Isaac Sim)  ")
    print("==================================================================")

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = HexapodPolicy(obs_dim=45, action_dim=18).to(device)
        policy.eval()
        print(f"[INFO] Untrained PyTorch policy initialized on device: {device}")
    else:
        policy = HexapodPolicy(obs_dim=45, action_dim=18)
        print("[INFO] PyTorch not installed. Using NumPy neural policy fallback.")

    for episode in range(num_episodes):
        episode_reward = 0.0
        # Simulating state
        joint_positions = np.random.normal(0, 0.1, 18)
        joint_velocities = np.random.normal(0, 0.05, 18)
        base_lin_vel = np.array([0.4, 0.02, 0.0])
        base_ang_vel = np.array([0.01, 0.01, 0.0])

        for step in range(max_steps):
            obs = np.concatenate([base_lin_vel, base_ang_vel, np.zeros(3), joint_positions, joint_velocities])
            
            if HAS_TORCH:
                state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_mean, _ = policy(state_tensor)
                    action = action_mean.squeeze(0).cpu().numpy()
            else:
                action = policy.forward_numpy(obs)

            # Update simulated physics motion
            base_lin_vel[0] = max(0.1, base_lin_vel[0] + 0.001 * np.mean(action[:6]))
            joint_positions += 0.01 * action

            forward_vel = base_lin_vel[0]
            drift_penalty = abs(base_lin_vel[1])
            reward = forward_vel * 2.0 - drift_penalty * 0.5 - 0.01 * np.sum(np.square(action))
            episode_reward += reward

        print(f"Episode {episode + 1}/{num_episodes} - Total DRL Locomotion Reward: {episode_reward:.2f}")

    print("[SUCCESS] Standalone DRL simulation finished cleanly.")


if __name__ == '__main__':
    if HAS_ISAAC_SIM:
        print("[INFO] NVIDIA Isaac Sim detected. Starting simulation stage...")
        run_isaac_sim_drl()
    else:
        print("[INFO] NVIDIA Isaac Sim environment not detected. Running Standalone Mode.")
        run_fallback_drl()
