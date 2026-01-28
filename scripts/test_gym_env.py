"""Simple test script for the SplatSim Gym environment wrapper.

Equivalent to running:
    python scripts/launch_nodes.py --robot_name robot_iphone_w_engine_new --robot sim_ur_pybullet_small_engine_new_interactive

Usage:
    python scripts/test_gym_env.py
    python scripts/test_gym_env.py --robot_name robot_iphone_w_engine_new --cam_i 3
    python scripts/test_gym_env.py --camera_names '["wrist_rgb"]'
    python scripts/test_gym_env.py --camera_names '["base_rgb", "wrist_rgb"]'
"""

import argparse
import json
import numpy as np

from splatsim.gym_env import make_single_env, list_envs


def main():
    parser = argparse.ArgumentParser(description="Test SplatSim Gym environment")
    parser.add_argument("--robot_name", type=str, default="robot_iphone_w_engine_new",
                        help="Name of robot splat to use")
    parser.add_argument("--cam_i", type=int, default=3,
                        help="Camera index for base_rgb")
    parser.add_argument("--camera_names", type=str, default='["base_rgb"]',
                        help='JSON list of camera names (e.g., \'["wrist_rgb"]\' or \'["base_rgb", "wrist_rgb"]\')')
    parser.add_argument("--no_gripper", action="store_true",
                        help="Disable gripper")
    parser.add_argument("--debug_mode", type=str, default=None,
                        help="Debug mode (e.g., 'no_background')")
    args = parser.parse_args()

    print("Available environments:", list_envs())

    # Parse camera names from JSON string
    camera_names = json.loads(args.camera_names)

    # Create a single small engine environment with the same config as launch_nodes.py
    # This is equivalent to:
    #   python scripts/launch_nodes.py --robot_name robot_iphone_w_engine_new --robot sim_ur_pybullet_small_engine_new_interactive
    env = make_single_env(
        "upright_small_engine_new",
        cfg={
            "robot_name": args.robot_name,
            "camera_names": camera_names,
            "cam_i": args.cam_i,
            "use_gripper": not args.no_gripper,
            "debug_mode": args.debug_mode,
        },
        render_mode="rgb_array",
    )

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Reset the environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    for cam_name in camera_names:
        if cam_name in obs:
            print(f"{cam_name} shape: {obs[cam_name].shape}")
    print(f"Initial info: {info}")

    # Run a few random steps
    print("\nRunning 10 random steps...")
    for i in range(10):
        # Sample random action
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, "
              f"truncated={truncated}, is_success={info['is_success']}")

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    # Clean up
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
