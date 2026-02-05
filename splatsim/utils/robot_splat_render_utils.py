import copy

import torch
from e3nn import o3
import einops
import numpy as np
import pybullet as p

from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from functools import wraps

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
    _cache: dict = field(default_factory=dict)  # Cache for GPU tensors to avoid recreating each step

def high_precision_mode(func):
    """
    Some PyTorch operations use TensorFloat-32 (TF32) on NVIDIA GPUs by default for better performance.
    This can lead to reduced numerical precision in certain computations.
    The rotation matrix math and quaternion conversions are sensitive to precision errors,
    and may throw assertion errors if TF32 is used.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Store original state
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        # Disable TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return func(*args, **kwargs)
        finally:
            # Always restore state
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return wrapper

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


def get_transformation_list(splatsim_obj: SplatSimObject, use_link_centers=True):
    """Get transformation matrices for each joint, using cached tensors to avoid GPU fragmentation.

    Args:
        splatsim_obj: The articulated SplatSimObject (must have articulation_config)
        use_link_centers: Whether to use link center positions

    Returns:
        List of (r_rel, t) tuples for each joint
    """
    assert splatsim_obj.articulation_config is not None, "splatsim_obj must have articulation_config"
    assert splatsim_obj.articulation_config.initial_link_poses is not None, "articulation_config must have initial_link_poses"

    robot_uid = splatsim_obj.sim_id
    initial_link_states = splatsim_obj.articulation_config.initial_link_poses
    num_joints = p.getNumJoints(robot_uid)

    new_joints = get_curr_link_states(robot_uid, use_link_centers=use_link_centers)

    if len(initial_link_states) != num_joints or len(new_joints) != num_joints:
        print(f"Error: Number of joints mismatch. Initial: {len(initial_link_states)}, New: {len(new_joints)}, Expected: {num_joints}")

    # Initialize transform cache if needed
    if 'transform_r_rel' not in splatsim_obj._cache:
        splatsim_obj._cache['transform_r_rel'] = [
            torch.zeros(3, 3, device='cuda', dtype=torch.float32) for _ in range(num_joints)
        ]
        splatsim_obj._cache['transform_t'] = [
            torch.zeros(3, device='cuda', dtype=torch.float32) for _ in range(num_joints)
        ]

    transformations_list = []
    for joint_index in range(num_joints):
        # x, y, z, q
        input_1 = (initial_link_states[joint_index]["pos"][0], initial_link_states[joint_index]["pos"][1], initial_link_states[joint_index]["pos"][2], np.array(initial_link_states[joint_index]["q"]))
        input_2 = (new_joints[joint_index]["pos"][0], new_joints[joint_index]["pos"][1], new_joints[joint_index]["pos"][2], np.array(new_joints[joint_index]["q"]))
        # this takes in q = (x,y,z,w) format for quaternions
        r_rel_np, t_np = compute_transformation(input_1, input_2)
        # Reuse cached tensors - copy data instead of creating new GPU tensors
        splatsim_obj._cache['transform_r_rel'][joint_index].copy_(torch.from_numpy(r_rel_np))
        splatsim_obj._cache['transform_t'][joint_index].copy_(torch.from_numpy(t_np))

        transformations_list.append((
            splatsim_obj._cache['transform_r_rel'][joint_index],
            splatsim_obj._cache['transform_t'][joint_index]
        ))

    return transformations_list


# Keep old function name for backwards compatibility with other files
def get_transfomration_list(robot_uid, initial_link_states, use_link_centers=True):
    """Deprecated: Use get_transformation_list(splatsim_obj) instead for better memory efficiency."""
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

@high_precision_mode
def transform_means(splatsim_obj: SplatSimObject, transformations_list, use_base_position=True, inplace=False, output_slices=None):
    """Transform articulated object gaussians (e.g., robot with multiple joints).

    Args:
        splatsim_obj: The articulated SplatSimObject
        transformations_list: List of (r_rel, t) tuples for each joint
        use_base_position: Whether to apply base position offset
        inplace: If True, save outputs directly into splatsim_obj.gaussians
        output_slices: If provided, a dict of pre-sliced tensor views to write into directly.
            Expected keys: '_xyz', '_rotation', '_opacity', '_scaling', '_features_dc', '_features_rest'
            This avoids creating intermediate tensors and improves memory efficiency.

    Returns:
        If output_slices is None: tuple of (xyz, rot, opacity, scales, features_rest, features_dc)
        If output_slices is provided: None (data written directly to slices)
    """
    # xyz is in global frame. pc is in splat frame
    pc = splatsim_obj.gaussians
    robot_uid = splatsim_obj.sim_id
    segmented_list = splatsim_obj.articulation_config.segmented_list

    # If writing to output_slices, use them as our working buffers
    if output_slices is not None:
        xyz = output_slices['_xyz']
        xyz.copy_(pc.get_xyz)
        rot = output_slices['_rotation']
        rot.copy_(pc.get_rotation)
        shs_featrest = output_slices['_features_rest']
        shs_featrest.copy_(pc._features_rest)
        # These don't need transformation per-segment, just copy once at the end
        scales_obj = pc._scaling
        opacity = pc.get_opacity_raw
        shs_dc = pc._features_dc
    else:
        # Use .clone() instead of copy.deepcopy - much more efficient for GPU tensors
        xyz = pc.get_xyz.clone()
        # Clone rotation since we modify it in-place per segment
        rot = pc.get_rotation.clone()
        # Clone features_rest since transform_shs modifies it in-place
        # features_dc is not modified, so we can reference it directly
        with torch.no_grad():
            shs_dc = pc._features_dc
            shs_featrest = pc._features_rest.clone()
        scales_obj = pc._scaling
        opacity = pc.get_opacity_raw

    # Cache base_position on the splatsim_obj to avoid recreating tensor each step
    if use_base_position:
        if 'base_position' not in splatsim_obj._cache:
            splatsim_obj._cache['base_position'] = torch.tensor(
                splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0],
                device='cuda'
            ).float()
        base_position = splatsim_obj._cache['base_position']

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

    # If output_slices provided, copy the non-transformed data into the slices
    if output_slices is not None:
        output_slices['_opacity'].copy_(opacity)
        output_slices['_scaling'].copy_(scales_obj)
        output_slices['_features_dc'].copy_(shs_dc)
        # xyz, rot, features_rest were already written in-place above
        # Return the slices (which are views into the scene_gaussian buffers)
        return xyz, rot, opacity, scales_obj, shs_featrest, shs_dc

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

@high_precision_mode
def transform_object(splatsim_obj, pos=None, quat=None, transform=None, use_base_position=True, inplace=False, output_slices=None):
    """
    Transforms all properties of a Gaussian splat object using a 4x4 transformation matrix.
    Memory-efficient: writes directly to output_slices if provided.
    """
    assert (pos is not None and quat is not None) + (transform is not None) == 1, \
        "Provide either (a pos and a quat) or (a 4x4 transform matrix)"

    pc = splatsim_obj.gaussians
    device = pc.get_xyz.device

    # --- 1. Unify Representation (4x4 Matrix) ---
    if transform is None:
        if 'transform_matrix' not in splatsim_obj._cache:
            splatsim_obj._cache['transform_matrix'] = torch.eye(4, device=device, dtype=torch.float32)
        transform = splatsim_obj._cache['transform_matrix']

        if not isinstance(quat, torch.Tensor):
            quat = torch.as_tensor(quat, device=device, dtype=torch.float32)
        rot_mat_tensor = o3.quaternion_to_matrix(quat)

        if not isinstance(pos, torch.Tensor):
            pos = torch.as_tensor(pos, device=device, dtype=torch.float32)

        transform[:3, :3] = rot_mat_tensor.to(device, torch.float32)
        transform[:3, 3] = pos.to(device, torch.float32)
        transform[3, :3] = 0
        transform[3, 3] = 1

    A = transform[:3, :3] 
    t = transform[:3, 3]

    # --- 2. SVD Decomposition for Scaling/Rotation ---
    try:
        U, S_vec, Vh = torch.linalg.svd(A)
    except (torch._C._LinAlgError, RuntimeError):
        try:
            U, S_vec, Vh = torch.linalg.svd(A.cpu())
            U, S_vec, Vh = U.to(device), S_vec.to(device), Vh.to(device)
        except Exception as e:
            raise RuntimeError(f"SVD failed for matrix:\n{A}") from e

    R_mat = U @ Vh
    if torch.linalg.det(R_mat) < 0:
        U[:, -1] *= -1
        S_vec[-1] *= -1
        R_mat = U @ Vh

    # --- 3. Compute Transformed Values ---
    
    # 3a. Position
    xyz_old = pc.get_xyz
    if use_base_position:
        if 'base_position' not in splatsim_obj._cache:
            splatsim_obj._cache['base_position'] = torch.tensor(
                splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0],
                device=device
            ).float()
        base_pos = splatsim_obj._cache['base_position']
        rotated_base = base_pos @ A.T
        xyz_final = (xyz_old + base_pos) @ A.T - rotated_base + t
    else:
        xyz_final = xyz_old @ A.T + t

    # 3b. Rotation
    rot_old_mat = o3.quaternion_to_matrix(pc.get_rotation)
    rot_new_mat = R_mat.unsqueeze(0) @ rot_old_mat
    rot_final = o3.matrix_to_quaternion(rot_new_mat)

    # 3c. Scaling
    scales_final = torch.log(pc.get_scaling * S_vec.unsqueeze(0))

    # --- 4. Assign to Final Buffers (Slices or New Tensors) ---
    if output_slices is not None:
        xyz_obj = output_slices['_xyz'].copy_(xyz_final)
        rot_obj = output_slices['_rotation'].copy_(rot_final)
        scales_obj = output_slices['_scaling'].copy_(scales_final)
        opacity_obj = output_slices['_opacity'].copy_(pc.get_opacity_raw)
        features_dc_obj = output_slices['_features_dc'].copy_(pc._features_dc)
        # Copy features_rest to slice first, then transform in-place (avoids clone allocation)
        output_slices['_features_rest'].copy_(pc._features_rest)
        features_rest_obj = transform_shs(output_slices['_features_rest'], R_mat)
    else:
        xyz_obj = xyz_final
        rot_obj = rot_final
        scales_obj = scales_final
        opacity_obj = pc.get_opacity_raw.clone()
        features_dc_obj = pc._features_dc.clone()
        features_rest_obj = transform_shs(pc._features_rest.clone(), R_mat)

    # --- 5. Finalize ---
    if inplace:
        pc._xyz = xyz_obj
        pc._rotation = rot_obj
        pc._opacity = opacity_obj
        pc._features_dc = features_dc_obj
        pc._features_rest = features_rest_obj
        pc._scaling = scales_obj
        if splatsim_obj.articulation_config is not None:
            splatsim_obj.articulation_config.initial_link_poses = get_curr_link_states(splatsim_obj.sim_id)

    return xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj

# Cached permutation matrix for transform_shs (created once, reused)
_P_MATRIX_CACHE = {}

@high_precision_mode
def transform_shs(shs_feat, rotation_matrix):
    ## rotate shs
    device = rotation_matrix.device
    if device not in _P_MATRIX_CACHE:
        _P_MATRIX_CACHE[device] = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], device=device).float()
    P = _P_MATRIX_CACHE[device]
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
