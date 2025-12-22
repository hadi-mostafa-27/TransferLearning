import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def freeze_backbone(model: nn.Module) -> None:
    # Freeze everything except final classifier
    for name, p in model.named_parameters():
        p.requires_grad = False
    # Unfreeze classifier head
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_last_blocks_resnet(model: nn.Module, n_blocks: int = 1) -> None:
    """
    Unfreeze last n ResNet blocks for fine-tuning.
    n_blocks=1 -> layer4
    n_blocks=2 -> layer3 + layer4
    """
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for l in layers[-n_blocks:]:
        for p in l.parameters():
            p.requires_grad = True
