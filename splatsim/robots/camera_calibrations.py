"""Shipped camera calibrations, selectable from an asset.yaml by name
(`model: fisheye_v2`). New cameras don't need an entry here: point
`intrinsics:` at the JSON written by scripts/calibrate_camera_intrinsics.py,
or write the numbers inline — see splatsim.robots.robot_spec.
"""

from typing import Any, Dict

# Wrist camera fisheye calibrations indexed by version (the old
# --wrist_cam_ver). 0 is pinhole (no entry).
#   1 = fisheye, original 2704x2028 GoPro calibration.
#   2 = fisheye, recalibrated 1920x1080 GoPro calibration.
WRIST_CAM_FISHEYE_CALIBRATIONS: Dict[int, Dict[str, Any]] = {
    1: {
        "CAL_W": 2704, "CAL_H": 2028,
        "CAL_FX": 775.5615, "CAL_FY": 778.0103,
        "CAL_CX": 1343.6974, "CAL_CY": 1005.3416,
        "D": [-0.0232652411, -0.0160767049, 0.0, 0.0],
    },
    2: {
        "CAL_W": 1920, "CAL_H": 1080,
        "CAL_FX": 777.86654216, "CAL_FY": 767.71982274,
        "CAL_CX": 973.16480901, "CAL_CY": 524.25398954,
        "D": [0.16369808, -0.15318689, 0.10608916, -0.02891525],
    },
}


def shipped_intrinsics(version: int) -> Dict[str, Any]:
    """The table entry as the normalised intrinsics dict CameraSpec uses."""
    c = WRIST_CAM_FISHEYE_CALIBRATIONS[version]
    return {"width": c["CAL_W"], "height": c["CAL_H"], "fx": c["CAL_FX"], "fy": c["CAL_FY"],
            "cx": c["CAL_CX"], "cy": c["CAL_CY"], "D": list(c["D"]), "source": f"shipped fisheye_v{version}"}
