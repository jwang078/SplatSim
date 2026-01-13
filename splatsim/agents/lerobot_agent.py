from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import numpy as np
import torch
from transformers import AutoTokenizer
from splatsim.utils.image_utils import letterbox

class LeRobotAgent:
    def __init__(self, checkpoint_path: str, device="cuda", task_description: str = "obstacle avoidance in scenario_0000"):
        self.policy = PI05Policy.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()
        self.device = device

        # Tokenize the task description once during initialization
        # PI05 uses the Qwen2-VL processor for language encoding
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.language_inputs = self.tokenizer(
            task_description,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Move tokens to device
        self.language_inputs = {k: v.to(device) for k, v in self.language_inputs.items()}

    def act(self, obs: dict) -> np.ndarray:
        # 1. Prepare observations (LeRobot expects dict of tensors with batch dim)
        # Assuming your ZMQ obs already has "base_rgb" in (3, H, W)
        base_rgb_resized = letterbox(obs["base_rgb"].detach().cpu().numpy(), output_size=(224, 224)) # 224x224 is the usual size for paligemma / pi0 (224x224, 448x448, and 896x896 pixels)
        lerobot_obs = {
            "observation.images.base_rgb": torch.from_numpy(base_rgb_resized).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(obs["joint_positions"]).unsqueeze(0).to(self.device),
            "observation.language.tokens": self.language_inputs["input_ids"],
            "observation.language.attention_mask": self.language_inputs["attention_mask"].bool(),
        }

        # 2. Inference
        with torch.no_grad():
            action_tensor = self.policy.select_action(lerobot_obs)


        # 3. Return to sim
        return np.concatenate([action_tensor.squeeze(0).cpu().numpy()[:6], [0.0]])  # Append gripper open command