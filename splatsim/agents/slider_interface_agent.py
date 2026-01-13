import numpy as np
import pybullet as p
import pybullet_data
from splatsim.agents.agent import Agent
from splatsim.utils.agent_state_utils import AGENT_STATE

class SliderInterfaceAgent(Agent):
    NUM_PAUSE_STEPS_AFTER_SAVE = 10

    def __init__(self):
        self.robot = None
        self.joint_signs = [1] * 6
        self.default_joint = [1.57, -1.57, 1.57, -1.57, -1.57, 0, 1]
        self.num_joints = len(self.default_joint)
        self.last_action = np.array(self.default_joint)
        self.state = AGENT_STATE.EXECUTING_TRAJ
        self.slider_ids = []
        self.settling_button_id = None
        self.settling_countdown = 0
        self.last_button_state = 0
        p.connect(p.GUI)
        self._init_sliders()

    def _init_sliders(self):
        # Create sliders for each joint
        self.slider_ids = []
        for i, val in enumerate(self.default_joint):
            slider_id = p.addUserDebugParameter(f"Joint {i+1}", -3.14, 3.14, val)
            self.slider_ids.append(slider_id)

        # Create settling button
        self.settling_button_id = p.addUserDebugParameter("Save Episode", 1, 0, 0)

    def act(self, obs):
        # Check settling button state
        button_state = p.readUserDebugParameter(self.settling_button_id)

        # Detect button press (transition from 0 to non-zero)
        if button_state != self.last_button_state and button_state > 0:
            self.state = AGENT_STATE.SETTLING
            self.settling_countdown = self.NUM_PAUSE_STEPS_AFTER_SAVE
            print("Saving episode")
        self.last_button_state = button_state

        # Handle settling countdown
        if self.settling_countdown > 0:
            self.settling_countdown -= 1
            if self.settling_countdown == 0:
                self.state = AGENT_STATE.EXECUTING_TRAJ
                print('back to executing')

        # Read joint angles from sliders
        if self.state == AGENT_STATE.EXECUTING_TRAJ:
            angles = []
            for slider_id in self.slider_ids:
                angle = p.readUserDebugParameter(slider_id)
                angles.append(angle)
            angles = np.array(angles)
            self.last_action = angles
        else:
            angles = self.last_action
        return angles