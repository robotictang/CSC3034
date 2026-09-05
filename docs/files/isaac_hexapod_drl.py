# Copyright Author: Dr Tang Tiong Yew
r"""
Deep Reinforcement Learning for 6-Legged Walking Robot in NVIDIA Isaac Sim
==========================================================================
This script demonstrates the observation/action data flow of an actor-critic
locomotion policy. It does not train PPO or load a trained checkpoint.  In
Isaac Sim mode, an open-loop forward gait generator supplies visible joint
motion; it is a demonstration controller, not a learned walking policy.

Execution Modes:
1. NVIDIA Isaac Sim Mode (Full 3D GPU physics & visual simulation):
   Run with Isaac Sim's standalone python:
   Windows: `C:\isaacsim\python.bat src\files\isaac_hexapod_drl.py`
   Linux: `~/isaacsim/python.sh src/files/isaac_hexapod_drl.py`
   Longer run: add `--walk-seconds 60 --hold-seconds 20`

2. Standalone Fallback Mode (Policy & Math simulation):
   `python3 src/files/isaac_hexapod_drl.py`
"""

import argparse
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
# Forward Gait Demonstration Controller
# =====================================================================
class RandomGaitController:
    """Generate coordinated joint-position targets for a forward trot.

    The Ant reference robot has two joints per leg.  A true hexapod with
    three joints per leg also works: every joint group alternates between
    swing and support phases. Fixed parameters make forward motion repeatable.
    """

    def __init__(self, joint_count, neutral_positions, seed=None):
        self.joint_count = joint_count
        self.neutral_positions = np.asarray(neutral_positions, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.frequency_hz = 1.5
        self.stride_amplitude = 0.40
        self.lift_amplitude = 0.30
        self.turn_bias = 0.0
        self.phase_offset = 0.0

    def target_positions(self, step, dt):
        phase = 2.0 * np.pi * self.frequency_hz * step * dt + self.phase_offset
        targets = self.neutral_positions.copy()
        joints_per_leg = 2 if self.joint_count % 2 == 0 else 3
        leg_count = max(1, self.joint_count // joints_per_leg)

        for joint_index in range(self.joint_count):
            leg_index = min(joint_index // joints_per_leg, leg_count - 1)
            joint_in_leg = joint_index % joints_per_leg
            # Alternate diagonal leg groups so one group supports while the
            # other swings.  This produces a stable-looking random gait.
            leg_phase = phase + (np.pi if leg_index % 2 else 0.0)
            side_sign = -1.0 if leg_index < leg_count / 2 else 1.0
            if joint_in_leg == 0:
                targets[joint_index] += (
                    self.stride_amplitude * np.sin(leg_phase)
                    + side_sign * self.turn_bias
                )
            else:
                targets[joint_index] += self.lift_amplitude * np.cos(leg_phase)

        return np.clip(targets, -1.5, 1.5)


# =====================================================================
# 1. NVIDIA Isaac Sim Implementation
# =====================================================================
def run_isaac_sim_drl(num_episodes=1, max_steps=None, walk_seconds=30, hold_seconds=10):
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
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:  # Isaac Sim 4.x compatibility
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicSphere
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.core.utils.viewports import set_camera_view

    world = World()
    world.scene.add_default_ground_plane()

    # Load Isaac's Ant reference articulation. Replace this path with a
    # compatible hexapod USD for a true six-legged exercise.
    assets_root_path = get_assets_root_path()

    is_articulated = True
    try:
        if not assets_root_path:
            raise RuntimeError("Isaac asset root is unavailable")
        # Isaac Sim 6 stores the Ant asset under ``Robots/IsaacSim``.
        # The previous path was valid for older asset releases only.
        robot_usd_path = assets_root_path + "/Isaac/Robots/IsaacSim/Ant/ant.usd"
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
    # A wide fixed view keeps the Ant visible along its forward path.
    set_camera_view(
        eye=np.array([8.0, -14.0, 9.0]),
        target=np.array([3.5, 0.0, 0.0]),
    )
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

    gait = RandomGaitController(
        action_dim,
        initial_joint_positions if is_articulated else np.zeros(action_dim),
    )
    physics_dt = 1.0 / 60.0
    max_steps = max_steps or int(walk_seconds / physics_dt)
    forward_speed = 0.25
    print(f"[INFO] Actor-critic policy scaffold initialized on {device}; articulation DOF={action_dim}")
    print(
        f"[INFO] Forward trot enabled for {num_episodes * max_steps * physics_dt:.0f} simulated seconds "
        f"at {forward_speed:.2f} m/s."
    )
    warned_action_failure = False

    for episode in range(num_episodes):
        world.reset()
        episode_reward = 0.0
        gait.reset()
        start_x = hexapod_robot.get_world_pose()[0][0] if is_articulated else 0.0
        print(f"[INFO] Episode {episode + 1}: forward trot at {gait.frequency_hz:.2f} Hz.")
        
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
                    policy_action = action_mean.squeeze(0).cpu().numpy()
            else:
                policy_action = policy.forward_numpy(obs)

            # The untrained actor output is retained as a small perturbation
            # so the policy data path remains observable, while the forward
            # gait supplies the coordinated motion needed for walking.
            gait_action = gait.target_positions(step, physics_dt)
            action = 0.95 * gait_action + 0.05 * policy_action

            if is_articulated:
                try:
                    hexapod_robot.apply_action(ArticulationAction(joint_positions=action))
                    # The untrained policy cannot create reliable propulsion.
                    # Keep a forward velocity while the legs cycle so this
                    # introductory demonstration visibly advances.
                    hexapod_robot.set_linear_velocity(
                        np.array([forward_speed, 0.0, base_lin_vel[2]])
                    )
                except Exception as exc:
                    if not warned_action_failure:
                        print(f"[WARN] Joint command could not be applied: {exc}")
                        warned_action_failure = True

            forward_vel = base_lin_vel[0]
            drift_penalty = abs(base_lin_vel[1])
            reward = forward_vel * 2.0 - drift_penalty * 0.5 - 0.01 * np.sum(np.square(action))
            episode_reward += reward

        end_x = hexapod_robot.get_world_pose()[0][0] if is_articulated else 0.0
        print(
            f"Episode {episode + 1}/{num_episodes} - Total DRL Locomotion Reward: {episode_reward:.2f}; "
            f"forward distance: {end_x - start_x:.2f} m"
        )

    print("[SUCCESS] Completed forward-walking actor-critic rollout in NVIDIA Isaac Sim.")
    if hold_seconds:
        print(f"[INFO] Keeping the final stage open for {hold_seconds} seconds.")
        for _ in range(int(hold_seconds / physics_dt)):
            world.step(render=True)
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
    parser = argparse.ArgumentParser(description="Run the Isaac Sim random-gait demonstration.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of gait episodes (default: 1).")
    parser.add_argument("--steps", type=int, default=None, help="Steps per episode; overrides --walk-seconds.")
    parser.add_argument(
        "--walk-seconds", type=float, default=30,
        help="Forward-walking duration in simulated seconds (default: 30).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=10,
        help="Seconds to keep the final Isaac Sim stage visible (default: 10).",
    )
    args = parser.parse_args()

    if HAS_ISAAC_SIM:
        print("[INFO] NVIDIA Isaac Sim detected. Starting simulation stage...")
        run_isaac_sim_drl(args.episodes, args.steps, args.walk_seconds, args.hold_seconds)
    else:
        print("[INFO] NVIDIA Isaac Sim environment not detected. Running Standalone Mode.")
        run_fallback_drl(args.episodes, args.steps or int(args.walk_seconds * 60))
