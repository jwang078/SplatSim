#!/usr/bin/env python3
"""
Script to load a .ply Gaussian splat and apply a hardcoded transformation matrix.
"""

import torch
import numpy as np
from argparse import ArgumentParser

# Import Gaussian splat utilities
from gaussian_splatting.gaussian_renderer import GaussianModel
from splatsim.utils.robot_splat_render_utils import SplatSimObject, transform_object


def main():
    parser = ArgumentParser(description="Load and transform a Gaussian splat")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the .ply file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save transformed .ply file (optional)")
    args = parser.parse_args()

    # For side view
    # transformation_matrix = np.array([
    #     [0.177658, -0.174348, 0.407577, -1.760707],
    #     [-0.036058, 0.431951, 0.200491, -1.300489],
    #     [-0.441833, -0.105356, 0.147522, -0.066176],
    #     [0.000000, 0.000000, 0.000000, 1.000000]
    # ])

    # For front view
    transformation_matrix = np.array([
        [0.977827, 0.065384, -0.198944, 3.358797],
        [-0.007756, 0.960663, 0.277610, -0.184045],
        [0.209269, -0.269912, 0.939869, -2.938772],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])

    print(f"Loading Gaussian splat from: {args.ply_path}")

    # Create a Gaussian model and load the .ply file
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply_path)

    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")

    # Create a SplatSimObject wrapper
    # Note: We need a minimal object_config for transform_object to work
    splatsim_obj = SplatSimObject(
        name="loaded_splat",
        splat_name="loaded_splat",
        gaussians=gaussians,
        object_config={
            "base_position": [[0, 0, 0]]  # Default base position
        }
    )

    # Convert transformation matrix to torch tensor
    transform_tensor = torch.from_numpy(transformation_matrix).to(
        device=gaussians.get_xyz.device
    ).float()

    print("\nApplying transformation matrix:")
    print(transformation_matrix)

    # Apply the transformation (inplace=True modifies the splatsim_obj directly)
    transform_object(
        splatsim_obj=splatsim_obj,
        transform=transform_tensor,
        use_base_position=False,  # Set to False since we want to apply the transform as-is
        inplace=True
    )

    print("\nTransformation applied successfully!")
    print(f"New center of mass: {splatsim_obj.gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()}")

    # Optionally save the transformed splat
    if args.output_path:
        print(f"\nSaving transformed splat to: {args.output_path}")
        splatsim_obj.gaussians.save_ply(args.output_path)
        print("Saved successfully!")

    return splatsim_obj


if __name__ == "__main__":
    main()
