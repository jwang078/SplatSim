"""
Joystick Interface Agent
========================
Provides a tkinter GUI with two virtual joysticks for Cartesian velocity control:
  - Right joystick: X/Y end-effector velocity
  - Left joystick:  Z end-effector velocity (vertical axis only)

The agent uses Differential Inverse Kinematics (Damped Least Squares) to convert
the desired Cartesian velocity into joint position commands for the next timestep.

Like SliderInterfaceAgent, this agent is self-contained: it creates its own
PyBullet DIRECT connection (no extra window) and loads the robot URDF purely for
Jacobian calculation.

Example usage in run_env_sim.py or a standalone script::

    from splatsim.agents.joystick_interface_agent import JoystickInterfaceAgent
    agent = JoystickInterfaceAgent(robot_name="robot_iphone_w_engine_new")
"""

import threading
import tkinter as tk

import numpy as np
import pybullet as p

from splatsim.agents.agent import Agent
from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils.agent_state_utils import AGENT_STATE
from splatsim.utils.paths import resolve_splatsim_path


# --------------------------------------------------------------------------- #
# Joystick widget                                                              #
# --------------------------------------------------------------------------- #

class _Joystick(tk.Canvas):
    """Draggable circular joystick widget.

    active_axes=(use_x, use_y).  Set use_x=False to lock to vertical only.
    """

    RADIUS = 80
    DOT_R  = 14
    SIZE   = 2 * RADIUS + 10

    def __init__(self, parent, label: str, active_axes=(True, True), **kwargs):
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                         bg="#1e1e2e", highlightthickness=0, **kwargs)
        self._active = active_axes
        cx = cy = self.SIZE // 2
        self._cx, self._cy = cx, cy
        self._dx = self._dy = 0.0
        self._lock = threading.Lock()

        self.create_oval(cx - self.RADIUS, cy - self.RADIUS,
                         cx + self.RADIUS, cy + self.RADIUS,
                         outline="#6c7086", width=2)
        self.create_line(cx - self.RADIUS, cy, cx + self.RADIUS, cy,
                         fill="#313244", width=1)
        self.create_line(cx, cy - self.RADIUS, cx, cy + self.RADIUS,
                         fill="#313244", width=1)
        self.create_text(cx, cy + self.RADIUS + 12, text=label,
                         fill="#cdd6f4", font=("Helvetica", 11, "bold"))
        self._dot = self.create_oval(cx - self.DOT_R, cy - self.DOT_R,
                                     cx + self.DOT_R, cy + self.DOT_R,
                                     fill="#89b4fa", outline="#cba6f7", width=2)

        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _move_dot(self, cx: float, cy: float):
        use_x, use_y = self._active
        dx = (cx - self._cx) if use_x else 0.0
        dy = (cy - self._cy) if use_y else 0.0
        dist = np.hypot(dx, dy)
        if dist > self.RADIUS:
            dx *= self.RADIUS / dist
            dy *= self.RADIUS / dist
        self.coords(self._dot,
                    self._cx + dx - self.DOT_R, self._cy + dy - self.DOT_R,
                    self._cx + dx + self.DOT_R, self._cy + dy + self.DOT_R)
        with self._lock:
            self._dx =  dx / self.RADIUS
            self._dy = -dy / self.RADIUS   # invert canvas-y: up = positive

    def _reset_dot(self):
        self.coords(self._dot,
                    self._cx - self.DOT_R, self._cy - self.DOT_R,
                    self._cx + self.DOT_R, self._cy + self.DOT_R)
        with self._lock:
            self._dx = self._dy = 0.0

    def _on_press(self,   e): self._move_dot(e.x, e.y)
    def _on_drag(self,    e): self._move_dot(e.x, e.y)
    def _on_release(self, _): self._reset_dot()

    def get_axes(self):
        """Return (dx, dy) normalised to [-1, 1], thread-safe."""
        with self._lock:
            return self._dx, self._dy


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #

class JoystickInterfaceAgent(Agent):
    """Cartesian velocity teleop via a tkinter dual-joystick GUI.

    Uses Damped Least Squares (DLS) Differential IK so the robot moves
    smoothly without wrist-flip artefacts and stays stable near singularities.

    Parameters
    ----------
    robot_name : str
        Key in configs/object_configs/objects.yaml (e.g. "robot_iphone_w_engine_new").
        The URDF path and EE link name are read from that entry automatically.
    velocity_scale : float
        Max Cartesian speed (m/s) when a joystick is fully deflected.
    dt : float
        Simulation timestep (seconds). Should match PyBullet's setTimeStep.
    damping : float
        DLS damping factor λ — prevents joint-velocity blow-up near singularities.
    num_dofs : int
        Number of arm joints, excluding gripper (default 6).
    """

    NUM_PAUSE_STEPS_AFTER_SAVE = 10

    def __init__(
        self,
        robot_name: str = "robot_iphone_w_engine_new",
        velocity_scale: float = 0.05,
        dt: float = 1.0 / 240.0,
        damping: float = 0.05,
        num_dofs: int = 6,
    ):
        self.velocity_scale = velocity_scale
        self.dt             = dt
        self.damping        = damping
        self.num_dofs       = num_dofs

        self.state = AGENT_STATE.EXECUTING_TRAJ
        self.settling_countdown = 0
        self._last_action = None

        # Load URDF path and EE link name from objects.yaml via SplatObjectConfig,
        # exactly the same way PybulletRobotServerBase does it.
        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path    = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name

        # Private DIRECT pybullet connection used only for Jacobian math.
        # This does NOT open a window and does NOT interfere with the main sim.
        self._pb_client = p.connect(p.DIRECT)
        self._robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self._pb_client,
        )
        self._ee_link = self._find_link(ee_link_name)

        # GUI runs on a background thread so it never blocks the control loop.
        gui_thread = threading.Thread(target=self._run_gui, daemon=True)
        gui_thread.start()

    # ---- PyBullet helpers ------------------------------------------------- #

    def _find_link(self, link_name: str) -> int:
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        for i in range(num_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)
            if info[12].decode("utf-8") == link_name:
                return i
        raise ValueError(
            f"Link '{link_name}' not found in robot URDF. "
            f"Check wrist_camera_link_name in configs/object_configs/objects.yaml."
        )

    # ---- GUI --------------------------------------------------------------- #

    def _run_gui(self):
        self._root = tk.Tk()
        self._root.title("Joystick Teleop  •  SplatSim")
        self._root.configure(bg="#1e1e2e")
        self._root.resizable(False, False)

        pad = dict(padx=16, pady=12)

        tk.Label(self._root, text="Cartesian Velocity Control",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 14, "bold")
                 ).grid(row=0, column=0, columnspan=2, pady=(14, 4))

        tk.Label(self._root, text="Left stick: Z  •  Right stick: X / Y",
                 bg="#1e1e2e", fg="#a6adc8",
                 font=("Helvetica", 10)
                 ).grid(row=1, column=0, columnspan=2, pady=(0, 8))

        # Left joystick → Z only (drag up = +Z, drag down = −Z)
        self._joy_left = _Joystick(self._root, "Z  (left stick)",
                                   active_axes=(False, True))
        self._joy_left.grid(row=2, column=0, **pad)

        # Right joystick → X / Y
        self._joy_right = _Joystick(self._root, "X / Y  (right stick)",
                                    active_axes=(True, True))
        self._joy_right.grid(row=2, column=1, **pad)

        self._vel_label = tk.Label(
            self._root, text="vx=+0.000  vy=+0.000  vz=+0.000",
            bg="#1e1e2e", fg="#a6e3a1", font=("Courier", 10))
        self._vel_label.grid(row=3, column=0, columnspan=2, pady=4)

        tk.Button(self._root, text="Save Episode",
                  command=self._on_save,
                  bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                  relief="flat", padx=12, pady=6,
                  font=("Helvetica", 11)
                  ).grid(row=4, column=0, columnspan=2, pady=(4, 16))

        self._root.mainloop()

    def _on_save(self):
        self.state = AGENT_STATE.SETTLING
        self.settling_countdown = self.NUM_PAUSE_STEPS_AFTER_SAVE
        print("Saving episode")

    # ---- Jacobian DLS IK -------------------------------------------------- #

    def _compute_next_joints(self, q: np.ndarray, v_cart: np.ndarray) -> np.ndarray:
        """q_next = q + J†(q) · v_cart · dt  via Damped Least Squares.

        Only the translational + rotational velocity is controlled (6-DOF twist).
        Angular velocity is set to zero so the orientation is held constant.
        """
        # Sync the local DIRECT robot to the current real joint positions
        for i in range(self.num_dofs):
            p.resetJointState(self._robot_id, i + 1, q[i],
                              physicsClientId=self._pb_client)

        q_full = list(q)
        zero   = [0.0] * len(q_full)

        jac_t, jac_r = p.calculateJacobian(
            self._robot_id,
            self._ee_link,
            [0.0, 0.0, 0.0],
            q_full, zero, zero,
            physicsClientId=self._pb_client,
        )
        J = np.vstack([np.array(jac_t), np.array(jac_r)])  # (6, n_joints)
        J = J[:, :self.num_dofs]                             # arm joints only

        lam   = self.damping
        q_dot = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6)) @ v_cart
        return q + q_dot * self.dt

    # ---- Agent interface --------------------------------------------------- #

    def act(self, obs) -> np.ndarray:
        joints    = obs["joint_positions"]               # length: num_dofs + 1
        q_arm     = joints[:self.num_dofs].astype(float)
        q_gripper = float(joints[self.num_dofs])

        if self._last_action is None:
            self._last_action = joints.copy().astype(float)

        # Settling countdown (mirrors SliderInterfaceAgent behaviour)
        if self.settling_countdown > 0:
            self.settling_countdown -= 1
            if self.settling_countdown == 0:
                self.state = AGENT_STATE.EXECUTING_TRAJ
                print("back to executing")

        if self.state != AGENT_STATE.EXECUTING_TRAJ:
            return self._last_action

        # Read joystick axes
        _,     vz_raw  = self._joy_left.get_axes()   # only y-axis active
        vx_raw, vy_raw = self._joy_right.get_axes()

        vx = vx_raw * self.velocity_scale
        vy = vy_raw * self.velocity_scale
        vz = vz_raw * self.velocity_scale

        # Update velocity readout in GUI (non-blocking; GUI may not be ready yet)
        try:
            self._vel_label.configure(
                text=f"vx={vx:+.3f}  vy={vy:+.3f}  vz={vz:+.3f}")
        except Exception:
            pass

        v_cart = np.array([vx, vy, vz, 0.0, 0.0, 0.0])
        if np.linalg.norm(v_cart) < 1e-6:
            return self._last_action

        q_next  = self._compute_next_joints(q_arm, v_cart)
        action  = np.concatenate([q_next, [q_gripper]])
        self._last_action = action
        return action
