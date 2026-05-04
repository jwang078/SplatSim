"""
Keyboard Interface Agent
========================
SE(3) end-effector teleop via keyboard. Uses pynput for key capture — works
over remote connections (TeamViewer, VNC) with no window focus required.
A small pygame window is opened for visual feedback only.

Key bindings:
    W / S   – +x / -x  (in EE frame)
    A / D   – -y / +y
    Q / E   – -z / +z
    U / O   – pitch +/−
    I / K   – yaw   +/−
    J / L   – roll  +/−
    Space   – toggle gripper open/close

Each call to act():
  1. Read _held_keys set (maintained by pynput KEYDOWN/KEYUP in background thread)
  2. FK → current EE pose
  3. Apply delta_pos / delta_rot in EE local frame
  4. Solve IK, clip per-joint step to max_joint_delta
  5. Return joint action

Example usage::

    from splatsim.agents.keyboard_interface_agent import KeyboardInterfaceAgent
    agent = KeyboardInterfaceAgent(robot_name="robot_iphone_w_engine_new")
"""

import threading

import numpy as np
import pybullet as p
import pygame
from pynput import keyboard as pynput_kb
from scipy.spatial.transform import Rotation

from splatsim.agents.agent import Agent
from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils.agent_state_utils import AGENT_STATE
from splatsim.utils.paths import resolve_splatsim_path

def _key_id(key):
    """Return a hashable identifier for a pynput key.

    Uses char.lower() for regular keys (consistent regardless of shift/caps),
    and the Key enum for special keys (space, ctrl, …).
    """
    if isinstance(key, pynput_kb.KeyCode) and key.char is not None:
        return key.char.lower()
    return key  # Key enum or KeyCode with no char


def _build_key_maps(p1: float, r1: float):
    KEY_POS = {
        'a': np.array([-1,  0,  0], dtype=float) * p1,
        'd': np.array([ 1,  0,  0], dtype=float) * p1,
        'w': np.array([ 0, -1,  0], dtype=float) * p1,
        's': np.array([ 0,  1,  0], dtype=float) * p1,
        'q': np.array([ 0,  0, -1], dtype=float) * p1,
        'e': np.array([ 0,  0,  1], dtype=float) * p1,
    }
    KEY_ROT = {
        'u': np.array([ 0,  0, -1], dtype=float) * r1,
        'o': np.array([ 0,  0,  1], dtype=float) * r1,
        'i': np.array([ 1,  0,  0], dtype=float) * r1,
        'k': np.array([-1,  0,  0], dtype=float) * r1,
        'j': np.array([ 0, -1,  0], dtype=float) * r1,
        'l': np.array([ 0,  1,  0], dtype=float) * r1,
    }
    return KEY_POS, KEY_ROT


class KeyboardInterfaceAgent(Agent):
    """SE(3) keyboard teleop agent using pynput — no window focus required.

    Parameters
    ----------
    robot_name : str
        Key in configs/object_configs/objects.yaml.
    pos_sensitivity : float
        Metres moved per step when a translation key is held.
    rot_sensitivity : float
        Radians rotated per step when a rotation key is held.
    max_joint_delta : float
        Maximum allowed change in any single joint angle per step (rad).
        Tunable velocity / safety constraint; lower = slower but smoother.
    num_dofs : int
        Number of arm joints, excluding gripper (default 6).
    """

    def __init__(
        self,
        robot_name: str = "robot_iphone_w_engine_new",
        pos_sensitivity: float = 0.02, #0.02, # 0.05,
        rot_sensitivity: float = 0.05, #0.05, # 0.2,
        max_joint_delta: float = 0.1,
        num_dofs: int = 6,
        delta_mode: bool = False,
    ):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.max_joint_delta = max_joint_delta
        self.num_dofs = num_dofs
        self.delta_mode = delta_mode

        self.state = AGENT_STATE.EXECUTING_TRAJ
        self._last_action = None
        self._desired_q = None  # internal desired joint state (IK seed); avoids drift from lagging obs
        self._close_gripper = False
        self._held_keys: set = set()  # pynput key objects; updated by listener thread
        self._release_timers: dict = {}  # kid → threading.Timer (deferred KEYUP)
        self._lock = threading.Lock()

        self._KEY_POS, self._KEY_ROT = _build_key_maps(pos_sensitivity, rot_sensitivity)

        # Private DIRECT pybullet connection for FK + IK
        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name

        self._pb_client = p.connect(p.DIRECT)
        self._robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self._pb_client,
        )
        self._ee_link = self._find_link(ee_link_name)
        self._num_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

        # pynput listener — global, no window focus needed
        self._listener = pynput_kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

        # pygame window for visual feedback only (no key capture)
        self._stop_display = False
        self._display_ready = threading.Event()
        self._display_thread = threading.Thread(target=self._run_display, daemon=True)
        self._display_thread.start()
        self._display_ready.wait()

        self._print_bindings()

    # ---- Helpers ----------------------------------------------------------- #

    def _print_bindings(self):
        print(
            "\n[KeyboardInterfaceAgent] Key bindings (pynput — no window focus needed):\n"
            "  W/S    – -y / +y  (EE frame)\n"
            "  A/D    – -x / +x\n"
            "  Q/E    – -z / +z\n"
            "  U/O    – pitch +/−\n"
            "  I/K    – roll   +/−\n"
            "  J/L    – yaw  +/−\n"
            "  Space  – toggle gripper\n"
        )

    def _find_link(self, link_name: str) -> int:
        for i in range(p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)
            if info[12].decode("utf-8") == link_name:
                return i
        raise ValueError(f"Link '{link_name}' not found in URDF.")

    # ---- pynput callbacks (called from listener thread) -------------------- #

    def _on_press(self, key):
        kid = _key_id(key)
        with self._lock:
            # Cancel any pending repeat-release for this key
            timer = self._release_timers.pop(kid, None)
            if timer is not None:
                timer.cancel()
            self._held_keys.add(kid)
            if kid == pynput_kb.Key.space:
                self._close_gripper = not self._close_gripper
                print(f"[keyboard] gripper {'CLOSED' if self._close_gripper else 'OPEN'}")

    def _on_release(self, key):
        kid = _key_id(key)
        # Defer removal by 100 ms — X11 key-repeat fires KEYUP+KEYDOWN pairs
        # ~30 ms apart; a real release has no following KEYDOWN within 100 ms.
        def _remove():
            with self._lock:
                self._held_keys.discard(kid)
                self._release_timers.pop(kid, None)
        with self._lock:
            old = self._release_timers.pop(kid, None)
            if old is not None:
                old.cancel()
            t = threading.Timer(0.1, _remove)
            self._release_timers[kid] = t
            t.start()

    # ---- pygame display thread (visual feedback only) ---------------------- #

    def _run_display(self):
        pygame.init()
        W, H = 460, 180
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Keyboard Teleop")

        font_key  = pygame.font.SysFont("monospace", 15, bold=True)
        font_label = pygame.font.SysFont("monospace", 11)

        # Two key grids side by side:
        #   left  — translate:  Q W E / A S D
        #   right — rotate:     U I O / J K L
        # Each entry: (label, kid)
        TRANSLATE = [
            [('Q', 'q'), ('W', 'w'), ('E', 'e')],
            [('A', 'a'), ('S', 's'), ('D', 'd')],
        ]
        ROTATE = [
            [('U', 'u'), ('I', 'i'), ('O', 'o')],
            [('J', 'j'), ('K', 'k'), ('L', 'l')],
        ]

        BG      = (30,  30,  46)
        DIM     = (60,  60,  90)   # inactive key
        LIT     = (100, 210, 100)  # held key highlight
        FG_DIM  = (180, 180, 200)
        FG_LIT  = (20,  20,  20)
        HEADING = (140, 140, 170)

        KW, KH, KP = 32, 28, 6   # key width, height, padding

        def draw_grid(grid, origin_x, origin_y, held):
            for r, row in enumerate(grid):
                for c, (label, kid) in enumerate(row):
                    active = kid in held
                    x = origin_x + c * (KW + KP)
                    y = origin_y + r * (KH + KP)
                    pygame.draw.rect(screen, LIT if active else DIM, (x, y, KW, KH), border_radius=4)
                    surf = font_key.render(label, True, FG_LIT if active else FG_DIM)
                    screen.blit(surf, (x + KW//2 - surf.get_width()//2,
                                       y + KH//2 - surf.get_height()//2))

        self._display_ready.set()

        while not self._stop_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_display = True

            with self._lock:
                held = set(self._held_keys)
                gripper_closed = self._close_gripper

            screen.fill(BG)

            # --- section title + centered "y +/-" above col 1 of translate ---
            screen.blit(font_label.render("TRANSLATE", True, HEADING), (14, 8))
            y_surf = font_label.render("y +/-", True, HEADING)
            cx_w = 14 + 1 * (KW + KP) + KW // 2  # col 1 = W/S
            screen.blit(y_surf, (cx_w - y_surf.get_width() // 2, 22))

            # --- section title + centered "roll +/-" above col 1 of rotate ---
            screen.blit(font_label.render("ROTATE", True, HEADING), (250, 8))
            r_surf = font_label.render("roll +/-", True, HEADING)
            cx_i = 250 + 1 * (KW + KP) + KW // 2  # col 1 = I/K
            screen.blit(r_surf, (cx_i - r_surf.get_width() // 2, 22))

            # --- key grids (pushed down a bit to make room for col labels) ---
            draw_grid(TRANSLATE, 14,  36, held)
            draw_grid(ROTATE,    250, 36, held)

            # --- row labels to the right of each grid ---
            for i, txt in enumerate(["z +/-", "x +/-"]):
                screen.blit(font_label.render(txt, True, HEADING),
                            (14 + 3*(KW+KP) + 4, 36 + i*(KH+KP) + KH//2 - font_label.get_height()//2))
            for i, txt in enumerate(["pitch +/-", "yaw +/-"]):
                screen.blit(font_label.render(txt, True, HEADING),
                            (250 + 3*(KW+KP) + 4, 36 + i*(KH+KP) + KH//2 - font_label.get_height()//2))

            # --- bottom row: SPACE gripper toggle ---
            y_bot = 36 + 2*(KH+KP) + 10

            space_active = pynput_kb.Key.space in held
            gripper_color = (210, 100, 100) if gripper_closed else (100, 180, 210)
            sw = 160
            pygame.draw.rect(screen, gripper_color if (space_active or gripper_closed) else DIM,
                             (14, y_bot, sw, KH), border_radius=4)
            gripper_state = "CLOSED" if gripper_closed else "open"
            gripper_label = f"[SPACE] grip: {gripper_state}"
            screen.blit(font_label.render(gripper_label, True, FG_LIT if gripper_closed else FG_DIM),
                        (14 + 4, y_bot + KH//2 - font_label.get_height()//2))

            pygame.display.flip()
            pygame.time.wait(33)  # ~30 Hz refresh

        pygame.quit()

    # ---- FK + IK ----------------------------------------------------------- #

    def _sync_joints(self, q: np.ndarray):
        for i in range(self.num_dofs):
            p.resetJointState(
                self._robot_id, i + 1, q[i], physicsClientId=self._pb_client
            )

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self._robot_id,
            self._ee_link,
            computeForwardKinematics=True,
            physicsClientId=self._pb_client,
        )
        return np.array(state[4]), np.array(state[5])  # pos, quat (xyzw)

    def _compute_next_joints(
        self, q: np.ndarray, delta_pos: np.ndarray, delta_rot: np.ndarray
    ) -> np.ndarray:
        self._sync_joints(q)
        pos, quat = self._get_ee_pose()

        R_current = Rotation.from_quat(quat)
        target_pos = pos + R_current.apply(delta_pos)
        R_delta = Rotation.from_euler("XYZ", delta_rot)
        target_quat = (R_current * R_delta).as_quat()

        # Build rest poses that bias wrist joints away from ±π to prevent
        # wrist-flip singularity. For each joint near ±π, prefer the 0-side.
        rest = list(q)
        for i in range(self.num_dofs):
            if abs(q[i]) > 2.5:  # approaching ±π
                rest[i] = 0.0

        n = self._num_pb_joints
        joint_poses = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link,
            target_pos,
            target_quat,
            restPoses=rest + [0.0] * (n - self.num_dofs),
            jointDamping=[0.1] * n,
            physicsClientId=self._pb_client,
        )
        q_ik = np.array(joint_poses[: self.num_dofs])
        # Reject IK solutions that require a large jump — these indicate the
        # solver converged to a far configuration branch or singularity.
        # if np.max(np.abs(q_ik - q)) > 0.15:
        #     return q
        # delta_q = np.clip(q_ik - q, -self.max_joint_delta, self.max_joint_delta)
        # return q + delta_
        return q_ik

    # ---- Agent interface --------------------------------------------------- #

    def act(self, obs) -> np.ndarray:
        joints = obs["joint_positions"]
        q_arm = joints[:self.num_dofs].astype(float)

        with self._lock:
            held = set(self._held_keys)

        delta_pos = sum(
            (v for k, v in self._KEY_POS.items() if k in held),
            np.zeros(3),
        )
        delta_rot = sum(
            (v for k, v in self._KEY_ROT.items() if k in held),
            np.zeros(3),
        )

        gripper_cmd = 1.0 if self._close_gripper else 0.0

        no_motion = np.linalg.norm(delta_pos) < 1e-9 and np.linalg.norm(delta_rot) < 1e-9

        if self.delta_mode:
            # Return [dx,dy,dz,droll,dpitch,dyaw,gripper] or all-NaN when no key held.
            if no_motion:
                return np.full(7, np.nan, dtype=np.float32)
            return np.concatenate([delta_pos, delta_rot, [gripper_cmd]]).astype(np.float32)

        # Absolute mode: return joint target via IK.
        if self._last_action is None:
            self._last_action = joints.copy().astype(float)

        if self._desired_q is None:
            self._desired_q = q_arm.copy()
        elif np.max(np.abs(self._desired_q - q_arm)) > 0.3:
            self._desired_q = q_arm.copy()

        if no_motion:
            return self._last_action

        q = self._compute_next_joints(self._desired_q, delta_pos, delta_rot)
        self._desired_q = q.copy()
        action = np.concatenate([q, [gripper_cmd]])
        self._last_action = action
        print(action)
        return action
