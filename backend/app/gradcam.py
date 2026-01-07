import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


class GradCAM:
    """
    Grad-CAM with hook fallback for older torch versions.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)

        # Torch compatibility: full backward hook if available, else fallback
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            # Deprecated in newer torch, but useful for older versions
            self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        A = self.activations  # (1,K,h,w)
        dA = self.gradients   # (1,K,h,w)

        if A is None or dA is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Check target layer / torch version.")

        weights = dA.mean(dim=(2, 3), keepdim=True)      # (1,K,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)     # (1,1,h,w)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def _denorm_imagenet(chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = chw * std + mean
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))  # HWC RGB uint8


def cam_overlay_base64(
    orig_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.35
) -> str:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed.")

    # ORIGINAL image (true pixels)
    orig = np.array(orig_pil)  # HWC RGB uint8
    H, W = orig.shape[:2]

    # Resize CAM to ORIGINAL image size
    cam_resized = cv2.resize(cam, (W, H))

    heat = (cam_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heat + (1 - alpha) * orig).astype(np.uint8)

    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.layer4[-1]
