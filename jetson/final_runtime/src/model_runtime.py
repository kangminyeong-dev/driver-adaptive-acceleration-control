from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


def load_runtime_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(config: dict) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config["image_height"], config["image_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(config_path: Path, device_arg: str | None = None) -> tuple[nn.Module, dict, torch.device]:
    config = load_runtime_config(config_path)
    config_dir = config_path.parent.parent
    weights_path = config_dir / config["weights_path"]
    device = resolve_device(device_arg)

    model = build_model(config["model_name"], len(config["class_names"]))
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, device


@torch.inference_mode()
def predict_image(model: nn.Module, config: dict, device: torch.device, image) -> dict[str, float | str]:
    transform = build_transform(config)
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(torch.argmax(probs).item())
    prediction = config["class_names"][pred_idx]
    e_mapping = config.get("e_mapping", {})

    off_score = float(probs[0].item())
    on_score = float(probs[1].item())

    return {
        "prediction": prediction,
        "e_raw": int(e_mapping.get(prediction, pred_idx)),
        "confidence": float(probs[pred_idx].item()),
        "off_score": off_score,
        "on_score": on_score,
    }
