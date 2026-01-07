import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import torch.nn as nn
from torchvision import models

# Grad-CAM helpers
from app.gradcam import GradCAM, cam_overlay_base64, get_resnet_target_layer


class InferenceService:
    def __init__(self, weights_path: str, meta_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        meta = json.loads(Path(meta_path).read_text())
        self.meta = meta
        self.img_size = int(meta["img_size"])

        # Convert class keys to int if needed
        self.idx_to_class = {int(k): v for k, v in meta["classes"].items()}

        self.model = build_model(
            num_classes=len(self.idx_to_class),
            backbone=meta["backbone"],
            pretrained=False,  # we load trained weights
        )

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Init Grad-CAM once
        self.gradcam = GradCAM(self.model, get_resnet_target_layer(self.model))

        self.tfm = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, pil_img: Image.Image):
        """
        Returns:
          label (str),
          confidence (float),
          probs_named (dict),
          overlay_b64 (str)  # base64 PNG for Grad-CAM overlay
        """
        # Keep the ORIGINAL image for overlay (correct alignment)
        img = pil_img.convert("RGB")

        # Model input (resized + normalized)
        x = self.tfm(img).unsqueeze(0).to(self.device)

        # 1) prediction (no gradients)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])

        probs_named = {
            self.idx_to_class[i]: float(probs[i])
            for i in range(len(probs))
        }

        # 2) Grad-CAM (needs gradients)
        cam = self.gradcam.generate(x, class_idx=pred_idx)

        # IMPORTANT FIX:
        # overlay on ORIGINAL image (not on normalized tensor)
        overlay_b64 = cam_overlay_base64(img, cam)

        return label, confidence, probs_named, overlay_b64


# -------------------------------------------------
# Model definition (backend-local, NO ml/ imports)
# -------------------------------------------------
def build_model(num_classes: int, backbone: str, pretrained: bool = False) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")
