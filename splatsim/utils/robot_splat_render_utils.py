import copy

import torch
from e3nn import o3
import einops
import numpy as np
import pybullet as p

from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from splatsim.utils.utils_fk import *

@dataclass
class ArticulationConfig:
    initial_joint_positions: List[float]
    joint_signs: List[int] # To handle joint direction conventions
    segmentation_labels: Optional[List[int]] = None # List of which link each point belongs to
    segmented_list: Optional[List[List[int]]] = None # List of splat indices per joint
    initial_link_poses: Optional[List[Tuple[List[float], List[float]]]] = None # List of (pos, quat) per link at initial joint positions]]

@dataclass
class SplatSimObject:
    name: str
    splat_name: str
    sim_id: int = None
    mass: float = 0.0 # Default to static object
    gaussians: Any = None 
    grasp_configs: List[dict] = field(default_factory=list)
    object_config: dict[Any] = None # The format in configs/object_configs/objects.yaml
    transformations_cache: dict[Any] = None
    is_articulated: bool = False # For example, the robot has is_articulated=True. An object with is_articulated should have articulation_config
    articulation_config: Optional[ArticulationConfig] = None

def get_curr_link_states(robot_uid, use_link_centers=True):
    link_states = []
    num_joints = p.getNumJoints(robot_uid)

    for joint_index in range(num_joints):
        link_state = p.getLinkState(robot_uid, joint_index, computeForwardKinematics=True)
        if use_link_centers:
            link_states.append({
                "pos": link_state[0],
                "q": link_state[1]
            })
        else:
            link_states.append({
                "pos": link_state[4],
                "q": link_state[5]
            })
    
    return link_states


def get_transfomration_list(robot_uid, initial_link_states, use_link_centers=True):
    num_joints = p.getNumJoints(robot_uid)

    new_joints = get_curr_link_states(robot_uid, use_link_centers=use_link_centers)

    if len(initial_link_states) != num_joints or len(new_joints) != num_joints:
        print(f"Error: Number of joints mismatch. Initial: {len(initial_link_states)}, New: {len(new_joints)}, Expected: {num_joints}")

    transformations_list = []
    for joint_index in range(num_joints):
        # x, y, z, q
        input_1 = (initial_link_states[joint_index]["pos"][0], initial_link_states[joint_index]["pos"][1], initial_link_states[joint_index]["pos"][2], np.array(initial_link_states[joint_index]["q"]))
        input_2 = (new_joints[joint_index]["pos"][0], new_joints[joint_index]["pos"][1], new_joints[joint_index]["pos"][2], np.array(new_joints[joint_index]["q"]))
        # this takes in q = (x,y,z,w) format for quaternions
        r_rel, t = compute_transformation(input_1, input_2)
        r_rel = torch.from_numpy(r_rel).to(device='cuda').float()
        t = torch.from_numpy(t).to(device='cuda').float()
        
        transformations_list.append((r_rel, t))

    return transformations_list

def crop_splat(splatsim_obj: SplatSimObject, keep_within_aabb=True):
    pc = splatsim_obj.gaussians
    aabb = splatsim_obj.object_config.get("aabb", {"bounding_box": None})['bounding_box']
    if aabb is None:
        return

    xyz_obj = copy.deepcopy(pc._xyz)

    #segment according to axis aligned bounding box
    segmented_indices = ((xyz_obj[:, 0] > aabb[0][0]) & (xyz_obj[:, 0] < aabb[1][0]) & (xyz_obj[:, 1] > aabb[0][1] ) & (xyz_obj[:, 1] < aabb[1][1]) & (xyz_obj[:, 2] > aabb[0][2] ) & (xyz_obj[:, 2] < aabb[1][2]))
    if not keep_within_aabb:
        segmented_indices = ~segmented_indices

    # Combine splats of robot and of objects
    with torch.no_grad():
        # gaussians.active_sh_degree = 0
        splatsim_obj.gaussians._xyz = pc._xyz[segmented_indices]
        splatsim_obj.gaussians._rotation = pc._rotation[segmented_indices]
        splatsim_obj.gaussians._opacity = pc._opacity[segmented_indices]
        splatsim_obj.gaussians._features_rest = pc._features_rest[segmented_indices]
        splatsim_obj.gaussians._features_dc = pc._features_dc[segmented_indices]
        splatsim_obj.gaussians._scaling = pc._scaling[segmented_indices]
    
def transform_means(splatsim_obj: SplatSimObject, transformations_list, use_base_position=True, inplace=False):
    # xyz is in global frame. pc is in splat frame
    pc = splatsim_obj.gaussians
    robot_uid = splatsim_obj.sim_id
    xyz = copy.deepcopy(splatsim_obj.gaussians.get_xyz)
    segmented_list = splatsim_obj.articulation_config.segmented_list

    # Note: this function does NOT handle transformation matrices with scaling factors in the rotation matrix area
    # Assume that it was handled already in object creation. The objects do not change size during runtime
    scales_obj = pc._scaling # pc.get_scaling is exp(pc._scaling)

    rot = pc.get_rotation
    opacity = pc.get_opacity_raw
    with torch.no_grad():
        shs_dc = pc._features_dc.clone()
        shs_featrest = pc._features_rest.clone()

    if use_base_position:
        base_position = torch.tensor(splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0], device='cuda').float()

    for joint_index in range(p.getNumJoints(robot_uid)):
        r_rel, t = transformations_list[joint_index] # T between initial link and current link
        segment = segmented_list[joint_index]

        if use_base_position:
            rotated_base_position = base_position @ r_rel.T
            xyz[segment] = torch.matmul(r_rel, xyz[segment].T + base_position[:, None]).T - rotated_base_position + t
        else:
            xyz[segment] = torch.matmul(r_rel, xyz[segment].T).T + t
        
        # Defining rotation matrix for the covariance
        rot_rotation_matrix = r_rel # (inv_rotation_matrix*scale_robot) @ r_rel @ rotation_matrix
        # o3.quaternion to matrix takes in (w,x,y,z) format for quaternions
        rot[segment] = o3.matrix_to_quaternion(rot_rotation_matrix @ o3.quaternion_to_matrix(rot[segment]))

        #transform the shs features
        shs_feat = shs_featrest[segment]
        shs_feat = transform_shs(shs_feat, rot_rotation_matrix)
        with torch.no_grad():
            shs_featrest[segment] = shs_feat
           
    return xyz, rot, opacity, scales_obj, shs_featrest, shs_dc

def create_cuboid_gaussians(
    side_lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing: float = 0.01,
    color_rgb: tuple[int, int, int] = (139, 69, 19),  # A nice brown
    device: str = "cuda:0"
) -> dict[str, torch.Tensor]:
    """
    Generates the parameters for a dense Gaussian splat cuboid. Center is in the middle of the block

    Args:
        side_lengths: (lx, ly, lz) of the cuboid.
        spacing: The distance between the center of each splat.
                 Smaller spacing = higher resolution = more splats.
        color_rgb: (R, G, B) tuple (0-255) for the brown color.
        device: The torch device to create tensors on.

    Returns:
        A dictionary of tensors (xyz, scales, rotations, opacity,
        features_dc, features_rest) that can be loaded into a 
        Gaussian splat model.
    """
    
    lx, ly, lz = side_lengths
    
    # 1. Create a 3D grid of points (the xyz)
    x = torch.arange(-lx / 2, lx / 2, step=spacing, device=device)
    y = torch.arange(-ly / 2, ly / 2, step=spacing, device=device)
    z = torch.arange(-lz / 2, lz / 2, step=spacing, device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    
    xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    num_points = xyz.shape[0]
    if num_points == 0:
        print("Warning: 0 points generated. Check side_lengths and spacing.")
        return {}

    # 2. Create small initial scales
    # We set them in log-space (as _scaling)
    # A small fraction of the spacing is a good start.
    initial_scale = torch.log(torch.tensor(spacing * 0.1, device=device))
    scales = torch.full((num_points, 3), initial_scale, device=device)

    # 3. Create initial rotations (no rotation)
    # (w, x, y, z) format for quaternion
    rotations = torch.zeros((num_points, 4), device=device)
    rotations[:, 0] = 1.0  # Identity quaternion

    # 4. Create opacities
    # We set them in logit-space (as _opacity)
    # logit(0.95) is approx 2.94, so sigmoid(2.94) is 0.95 (very opaque)
    opacity = torch.full((num_points, 1), 2.94, device=device)

    # 5. Set the color (Spherical Harmonics)
    
    # Normalize color from [0, 255] to [0, 1]
    color_normalized = torch.tensor(color_rgb, device=device, dtype=torch.float32) / 255.0
    
    # Convert to the format expected by the 3DGS SH DC component
    # C0 = 0.28209479
    # The DC (degree 0) feature is (color - 0.5) / C0
    # We only set the first (DC) component
    features_dc = torch.zeros((num_points, 1, 3), device=device)
    features_dc[:, 0, :] = (color_normalized - 0.5) / 0.28209479
    
    # Set all other SH components (features_rest) to zero
    # Assuming 15 other components (degree 3)
    features_rest = torch.zeros((num_points, 15, 3), device=device)
    
    return {
        "_xyz": xyz,
        "_scaling": scales,
        "_rotation": rotations,
        "_opacity": opacity,
        "_features_dc": features_dc,
        "_features_rest": features_rest
    }

def transform_from_pos_quat(pos, quat, device='cuda'):
    # Build the 4x4 transform matrix from pos and quat
    transform = torch.eye(4, device=device, dtype=torch.float32)
    
    # quaternion_to_rot_matrix expects (x, y, z, w) format
    # rot_mat_tensor = torch.tensor(quaternion_to_rot_matrix(quat), device=device).float()

    if not isinstance(quat, torch.Tensor):
        quat = torch.tensor(quat, device=device, dtype=torch.float32)

    # o3.quaternion to matrix takes in (w,x,y,z) format for quaternions
    rot_mat_tensor = o3.quaternion_to_matrix(quat)
    
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, device=device, dtype=torch.float32)
        
    transform[:3, :3] = rot_mat_tensor.to(device, torch.float32)
    transform[:3, 3] = pos.to(device, torch.float32)
    return transform

def transform_to_pos_quat(transform):
    # Extract rotation matrix and position from 4x4 transform
    rot_mat = transform[:3, :3]
    pos = transform[:3, 3].cpu().numpy().tolist()
    quat = get_quaternion_from_matrix(rot_mat.cpu().numpy()).tolist()
    return pos, quat

def build_matrix_from_r_t(r, t, device='cuda'):
    transform = torch.eye(4, device=device, dtype=torch.float32)
    transform[:3, :3] = r.to(device, torch.float32)
    transform[:3, 3] = t.to(device, torch.float32)
    return transform

def transform_object(splatsim_obj, pos=None, quat=None, transform=None, use_base_position=True, inplace=False):
    """
    Transforms all properties of a Gaussian splat object (splatsim_obj)
    using a 4x4 transformation matrix.
    
    This function correctly handles non-uniform scaling by using SVD
    to decompose the transformation into its rotation and scaling components.
    """
    assert (pos is not None and quat is not None) + (transform is not None) == 1, "Provide either (a pos and a quat) or (a 4x4 transform matrix)"

    pc = splatsim_obj.gaussians
    device = pc.get_xyz.device

    # Unify the representation to a 4x4 transform matrix
    if transform is None:
        transform = transform_from_pos_quat(pos, quat, device)

    A = transform[:3, :3] # Scaling + Rotation part
    t = transform[:3, 3] # Translation part

    try:
        # Decompose A = U @ S_diag @ Vh
        # U and Vh are rotation matrices. S_vec is a vector of singular values.
        U, S_vec, Vh = torch.linalg.svd(A)
    except torch._C._LinAlgError:
        # Fallback if SVD fails (e.g., matrix is all zeros)
        U = torch.eye(3, device=device)
        S_vec = torch.ones(3, device=device)
        Vh = torch.eye(3, device=device)

    # --- 3. Find pure rotation R (and fix reflections) ---
    # R = U @ Vh. (Vh is V.T)
    R_mat = U @ Vh
    # SVD can produce a reflection (det(R) = -1) if one scale is negative.
    # We fix this to get a proper rotation matrix.
    if torch.linalg.det(R_mat) < 0:
        U[:, -1] *= -1 # Invert the last column of U
        S_vec[-1] *= -1 # And invert the last scaling factor
        R_mat = U @ Vh

    # --- 4. Get all Gaussian properties ---
    xyz_old = pc.get_xyz
    rot_old = pc.get_rotation     # (N, 4) quaternions
    scales_old = pc.get_scaling  # (N, 3) scales (already exp(log_scales))
    opacity_obj = pc.get_opacity_raw
    
    with torch.no_grad():
        features_dc_obj = pc._features_dc.clone()
        features_rest_obj = pc._features_rest.clone()

    # --- 5. Apply transformations to each property ---
    # 1. Positions: Apply full A matrix and t
    # (N, 3) @ (3, 3) -> (N, 3). Then add translation.
    if use_base_position:
        base_position = torch.tensor(splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0], device='cuda').float()
        rotated_base_position = base_position @ A.T
        # The transform's t includes the base position offset because it was computed in world frame
        # xyz is in the object frame, so we need to remove the base position effect
        xyz_obj = (xyz_old + base_position) @ A.T - rotated_base_position + t
    else:
        xyz_obj = xyz_old @ A.T + t

    # 2. Rotations: Compose with pure rotation R_mat
    # We must convert old rotations to matrices, multiply, then convert back
    rot_old_mat = o3.quaternion_to_matrix(rot_old) # (N, 3, 3)
    # (1, 3, 3) @ (N, 3, 3) -> (N, 3, 3)
    rot_new_mat = R_mat.unsqueeze(0) @ rot_old_mat 
    # import pdb; pdb.set_trace()
    rot_obj = o3.matrix_to_quaternion(rot_new_mat) # (N, 4)

    # 3. Scales: Multiply with pure scaling S_vec
    # (N, 3) * (1, 3) -> (N, 3)
    scales_new = scales_old * S_vec.unsqueeze(0)
    scales_obj = torch.log(scales_new) # Save back in log space

    # 4. SH Features: Transform with pure rotation R_mat
    features_rest_obj = transform_shs(features_rest_obj, R_mat)

    if inplace:
        splatsim_obj.gaussians._xyz = xyz_obj
        splatsim_obj.gaussians._rotation = rot_obj
        splatsim_obj.gaussians._opacity = opacity_obj
        splatsim_obj.gaussians._features_dc = features_dc_obj
        splatsim_obj.gaussians._features_rest = features_rest_obj
        splatsim_obj.gaussians._scaling = scales_obj
        if splatsim_obj.articulation_config is not None:
            # Update the initial link poses after transforming the robot
            splatsim_obj.articulation_config.initial_link_poses = get_curr_link_states(
                splatsim_obj.sim_id
            )

    return xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj


def transform_shs(shs_feat, rotation_matrix):
    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], device=rotation_matrix.device).float() # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    rot_angles = (rot_angles[0].cpu(), rot_angles[1].cpu(), rot_angles[2].cpu())
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # shs_feat: (..., 15, 3)   # [SH-index, RGB]
    # D_1: (..., 3, 3) or (3,3)
    # D_2: (..., 5, 5) or (5,5)
    # D_3: (..., 7, 7) or (7,7)

    device = shs_feat.device
    dtype  = shs_feat.dtype

    # Build block-diagonal Wigner-D. If D_1/2/3 are batched (have leading ...),
    # make a batched block-diagonal; otherwise a single 15x15 is fine.

    if D_1.dim() == 2:  # unbatched (3,3),(5,5),(7,7)
        D = torch.block_diag(D_1.to(device=device, dtype=dtype),
                            D_2.to(device=device, dtype=dtype),
                            D_3.to(device=device, dtype=dtype))                    # (15,15)
    else:
        # batched: D_1: (...,3,3), etc. Build (...,15,15)
        *batch, _, _ = D_1.shape
        D = torch.zeros(*batch, 15, 15, device=device, dtype=dtype)
        D[...,  0: 3,  0: 3] = D_1
        D[...,  3: 8,  3: 8] = D_2
        D[...,  8:15,  8:15] = D_3

    # take l=1..3 bands and move RGB before SH for a single einsum
    sh = einops.rearrange(shs_feat[..., :15, :], '... s r -> ... r s')  # (..., 3, 15)

    # one einsum: (...,i,j) @ (...,r,j) -> (...,r,i)
    sh_rot = torch.einsum('...ij, ...rj -> ...ri', D, sh)               # (..., 3, 15)

    # restore layout and write back
    shs_feat[..., :15, :] = einops.rearrange(sh_rot, '... r i -> ... i r')  # (..., 15, 3)

    return shs_feat


def get_segmented_indices(splatsim_obj: SplatSimObject):
    pc = splatsim_obj.gaussians
    aabb = splatsim_obj.object_config["aabb"]["bounding_box"]

    # Defining a cube in Gaussian space to segment out the robot
    xyz = pc.get_xyz # 3D means shape (N, 3)
    
    segmented_points = []

    # TODO can't this just be a list of xyz points, not the mask and the original points?
    condition = (xyz[:, 0] > aabb[0][0]) & (xyz[:, 0] < aabb[1][0]) & (xyz[:, 1] > aabb[0][1]) & (xyz[:, 1] < aabb[1][1]) & (xyz[:, 2] > aabb[0][2]) & (xyz[:, 2] < aabb[1][2])
    condition = torch.where(condition)[0]
    for i in range(p.getNumJoints(splatsim_obj.sim_id)):
        segmented_points.append(condition[splatsim_obj.articulation_config.segmentation_labels==i])
    
    return segmented_points
