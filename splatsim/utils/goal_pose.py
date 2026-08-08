"""Shared goal-pose generation: one definition of "where should the arm be
to image a target", used by the env server AND by standalone tools.

Everything about how a goal pose is specified lives here — the wrist-camera
convention, the camera-up modes, the position-vs-aim split, the parameter
defaults — so `tune_goal_pose.py` cannot drift from what the env actually
generates. Before this existed the tuner re-derived all of it and quietly
disagreed with the env (different aim axis, different tip offset, no
soft-cost ranking), which is exactly the class of bug a tuning tool must not
have: you tune against one thing and ship another.

The heavy lifting stays in `grape_targets.reachable_approach_pose`; this is
the layer that decides WHAT to ask it for.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

from splatsim.utils import grape_targets

# --------------------------------------------------------------- conventions
# The wrist camera looks down `wrist_camera_link` +Z and its image-up is -Y
# (COLMAP +Y is down), with NO offset between link frame and camera frame.
# Copied from PybulletRobotServerBase.get_wrist_camera_transform /
# _splat_camera_from_gsplat; keep in sync if the robot's camera mounting
# convention ever changes.
CAMERA_FORWARD_AXIS = (0.0, 0.0, 1.0)
CAMERA_UP_AXIS = (0.0, -1.0, 0.0)


class CameraUpMode(str, Enum):
    """Which way the wrist camera's image-up should point in WORLD space."""

    OFF = "off"          # unconstrained — whatever roll IK lands on
    UPRIGHT = "upright"  # image-up toward world +Z
    INVERTED = "inverted"  # image-up toward world -Z (upside down)

    def world_up(self):
        """World direction for `align_up_world`, or None when unconstrained."""
        if self is CameraUpMode.UPRIGHT:
            return (0.0, 0.0, 1.0)
        if self is CameraUpMode.INVERTED:
            return (0.0, 0.0, -1.0)
        return None

    @classmethod
    def from_world_up(cls, vec) -> "CameraUpMode":
        """Inverse of `world_up` — lets an env express its preference as a
        vector (GRIPPER_CAMERA_UP_WORLD) and still round-trip to a mode."""
        if vec is None:
            return cls.OFF
        return cls.INVERTED if float(np.asarray(vec)[2]) < 0 else cls.UPRIGHT


def wrist_camera_pose(client, robot_id: int, ee_link: int):
    """``(eye, forward, up)`` of the wrist camera at the CURRENT joint state,
    in world coordinates, using the convention above."""
    st = client.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    eye = np.asarray(st[4], dtype=np.float64)
    rot = np.asarray(client.getMatrixFromQuaternion(st[5]),
                     dtype=np.float64).reshape(3, 3)
    return eye, rot[:, 2], -rot[:, 1]


def wrist_view_matrix(client, robot_id: int, ee_link: int):
    """PyBullet view matrix looking through the wrist camera right now."""
    eye, forward, up = wrist_camera_pose(client, robot_id, ee_link)
    return client.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=(eye + forward).tolist(),
        cameraUpVector=up.tolist(),
    )


def camera_up_state(quat, client=None):
    """Classify an achieved goal orientation. Returns
    ``(state, tilt_deg, up_world)`` where state is "upright" / "UPSIDE DOWN"
    / "SIDEWAYS".

    The deadband matters: a bare ``up_z < 0`` test labels a HORIZONTAL camera
    upside down, which is what the position-relaxation search often lands on,
    so that test reports success for poses that are nothing of the kind.
    """
    import pybullet as pb

    rot = np.asarray(
        (client.getMatrixFromQuaternion(quat) if client is not None
         else pb.getMatrixFromQuaternion(quat)), dtype=np.float64).reshape(3, 3)
    up = -rot[:, 1]
    uz = float(up[2])
    state = "upright" if uz > 0.5 else "UPSIDE DOWN" if uz < -0.5 else "SIDEWAYS"
    tilt = float(np.degrees(np.arccos(np.clip(uz, -1.0, 1.0))))
    return state, tilt, up


# ------------------------------------------------------------------- the spec
@dataclasses.dataclass
class GoalPoseSpec:
    """Everything that defines an imaging goal pose. Envs own the values;
    tools read them off the env so both ask for the same thing."""

    standoff_m: float = 0.10
    # None = measure from the URDF by FK (grape_targets.tool_tip_vector).
    tip_offset_m: float | None = None
    # Axis pointed AT the target. Defaults to the camera's optical axis, so
    # the fruit lands centred in frame; aiming the tool axis instead puts it
    # off-centre by the angle between the two (~13 deg on this gripper).
    aim_axis_local: tuple = CAMERA_FORWARD_AXIS
    camera_up_axis_local: tuple = CAMERA_UP_AXIS
    camera_up: CameraUpMode = CameraUpMode.OFF
    roll_offset_deg: float = 0.0
    max_aim_error_deg: float = 12.0
    max_up_error_deg: float = 45.0
    position_relax_m: float = 0.08
    position_relax_step_m: float = 0.02
    ik_seed: int = 0
    # Search budget. More seeds = more elbow/wrist branches tried, which is
    # what finds solutions on bunches where the obvious branch collides;
    # ik_enough_candidates stops the sweep once that many pass every hard
    # gate, so the budget is only spent where solutions are scarce.
    ik_random_seeds: int = 12
    ik_enough_candidates: int = 5
    # False (default) = an unmet camera-up request raises instead of silently
    # returning a differently-rolled pose.
    allow_roll_fallback: bool = False
    # Position off the bunch PEDUNCLE (cut-ready) rather than its centre; the
    # camera still aims at the centre.
    aim_at_peduncle: bool = True

    @classmethod
    def from_env_class(cls, env_cls, **overrides) -> "GoalPoseSpec":
        """Build from an env server class's attributes, so the env stays the
        single source of truth and tools inherit its choices."""
        up_world = getattr(env_cls, "GRIPPER_CAMERA_UP_WORLD", None)
        spec = cls(
            standoff_m=float(getattr(env_cls, "GRAPE_STANDOFF_M", 0.10)),
            tip_offset_m=None,
            aim_axis_local=tuple(getattr(env_cls, "CAMERA_FORWARD_AXIS",
                                         CAMERA_FORWARD_AXIS)),
            camera_up=CameraUpMode.from_world_up(up_world),
            roll_offset_deg=float(getattr(env_cls, "GRIPPER_ROLL_OFFSET_DEG", 0.0)),
            aim_at_peduncle=bool(getattr(env_cls, "AIM_AT_PEDUNCLE", True)),
        )
        for k, v in overrides.items():
            setattr(spec, k, v)
        return spec


def resolve_targets(bunch: dict, spec: GoalPoseSpec):
    """``(reach_pt, look_at)`` for a bunch: where the TOOL goes versus what
    the CAMERA centres on.

    The tool sits off the peduncle (the stem a cutter must reach, so a
    straight-ahead nudge after imaging arrives at it) while the camera
    centres on the bunch centre. Falls back to the centre for target files
    written before the peduncle field existed.
    """
    center = np.asarray(bunch["center"], dtype=np.float64)
    ped = bunch.get("peduncle")
    if ped is None or not spec.aim_at_peduncle:
        return center, center
    return np.asarray(ped, dtype=np.float64), center


def solve_goal_pose(client, robot_id: int, ee_link: int, joint_indices,
                    bunch: dict, spec: GoalPoseSpec, from_point=None,
                    collision_fn=None, score_fn=None):
    """Generate the imaging goal pose for ``bunch``.

    Returns ``(pos, quat, q_seed)``; raises ValueError when no pose satisfies
    the spec (including an unmet camera-up request, unless the spec allows
    the roll fallback).
    """
    reach_pt, look_at = resolve_targets(bunch, spec)
    tip = spec.tip_offset_m
    if tip is None:
        _, tip = grape_targets.tool_tip_vector(client, robot_id, ee_link)
    if from_point is None:
        base = np.asarray(client.getBasePositionAndOrientation(robot_id)[0],
                          dtype=np.float64)
        from_point = np.array([base[0], base[1], reach_pt[2]])
    return grape_targets.reachable_approach_pose(
        client, robot_id, ee_link, list(joint_indices), reach_pt,
        look_at=look_at,
        standoff=spec.standoff_m,
        from_point=from_point,
        aim_axis_local=spec.aim_axis_local,
        camera_up_axis_local=spec.camera_up_axis_local,
        align_up_world=spec.camera_up.world_up(),
        roll_offset_deg=spec.roll_offset_deg,
        max_aim_error_deg=spec.max_aim_error_deg,
        max_up_error_deg=spec.max_up_error_deg,
        position_relax_m=spec.position_relax_m,
        position_relax_step_m=spec.position_relax_step_m,
        ik_seed=spec.ik_seed,
        ik_random_seeds=spec.ik_random_seeds,
        ik_enough_candidates=spec.ik_enough_candidates,
        allow_roll_fallback=spec.allow_roll_fallback,
        collision_fn=collision_fn,
        tool_tip_offset=tip,
        score_fn=score_fn,
    )


# ------------------------------------------------------------- projection
def project_points(points_world, splatsim_camera, rectify_zoom: float = 1.0):
    """Project world points into a SplatSimCamera's IMAGE pixels.

    Returns ``(uv, valid)`` — ``uv`` is (N, 2) float pixel coordinates and
    ``valid`` marks points in front of the camera.

    Matches what ``render_image`` actually hands back. That matters for the
    fisheye wrist: the render is RECTIFIED before it is returned
    (``_rectify_fisheye_image`` remaps it to a pinhole at K with fx/fy scaled
    by FISHEYE_RECTIFY_ZOOM), so projecting with the RAW fisheye intrinsics
    and distortion would put markers in the wrong place. Pass the same zoom
    the server uses and this is exact pinhole math against the rectified
    image; pass 1.0 for an already-pinhole camera.

    Duck-types the camera (``.camera.R/.T/.image_width/.image_height``,
    ``.intrinsic_matrix``, ``.camera_model``) so this module needs no import
    from the robot-server package.
    """
    cam = splatsim_camera.camera
    rot = np.asarray(cam.R, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(cam.T, dtype=np.float64).reshape(3)
    eye = -rot @ trans
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    # Camera axes are the columns of R (z forward, y down — COLMAP), so the
    # camera-frame coordinates are just (p - eye) projected onto them.
    p_cam = (pts - eye) @ rot
    z = p_cam[:, 2]

    width, height = float(cam.image_width), float(cam.image_height)
    k = getattr(splatsim_camera, "intrinsic_matrix", None)
    if k is not None:
        k = np.asarray(k.detach().cpu().numpy() if hasattr(k, "detach") else k,
                       dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        if getattr(splatsim_camera, "camera_model", "pinhole") == "fisheye":
            fx *= rectify_zoom
            fy *= rectify_zoom
    else:
        fx = (width / 2.0) / np.tan(float(cam.FoVx) * 0.5)
        fy = (height / 2.0) / np.tan(float(cam.FoVy) * 0.5)
        cx, cy = width / 2.0, height / 2.0

    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    return np.stack([u, v], axis=1), z > 1e-6
