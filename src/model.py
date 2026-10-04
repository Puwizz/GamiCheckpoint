"""
model.py — Model Definition
============================
กำหนดโครงสร้าง model สำหรับจำแนกขั้นตอน Origami
ใช้ Transfer Learning จาก MobileNetV2 (pre-trained บน ImageNet)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


class StepModel(nn.Module):
    """
    CNN Classifier สำหรับจำแนกขั้นตอน Origami
    
    ใช้ MobileNetV2 เป็น backbone (pre-trained บน ImageNet)
    แก้ classifier layer สุดท้ายให้ตรงกับจำนวนขั้นตอน
    
    Args:
        num_classes: จำนวน class (= จำนวนขั้นตอน origami)
        pretrained: ใช้ weights ที่ train มาแล้วหรือไม่
        freeze_backbone: freeze backbone layers หรือไม่
    """

    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.classes: Optional[List[str]] = None

        # โหลด MobileNetV2 pre-trained
        if pretrained:
            self.backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze backbone layers (ไม่ต้อง train ใหม่)
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # แก้ classifier layer สุดท้าย
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True):
        """
        Override train mode to ensure BatchNorm layers inside backbone features
        remain in eval mode when freeze_backbone is True (prevents stats drift).
        """
        super().train(mode)
        if self.freeze_backbone and mode:
            for module in self.backbone.features.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        expected_classes: Optional[List[str]] = None,
    ) -> "StepModel":
        """
        Load a trained checkpoint and return ready-to-use model in eval mode.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: torch.device (defaults to cpu if not provided)
            expected_classes: list of expected class names (optional validation)
            
        Returns:
            model: StepModel instance in eval mode
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        task = checkpoint.get("task")
        if task is not None and task != "step-classification":
            raise ValueError(f"Checkpoint task is not 'step-classification': {task}")

        classes = checkpoint.get("classes")
        if expected_classes is not None and classes is not None and classes != expected_classes:
            raise ValueError(
                f"Checkpoint classes mismatch: expected {expected_classes}, got {classes}"
            )

        target_classes = classes if classes is not None else expected_classes
        if target_classes is None:
            raise ValueError(
                f"Checkpoint does not contain 'classes' and expected_classes was not provided: {checkpoint_path}"
            )

        num_classes = len(target_classes)
        model = cls(num_classes=num_classes, pretrained=False, freeze_backbone=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        model.classes = target_classes
        return model

    def save_checkpoint(
        self,
        path: Union[str, Path],
        classes: List[str],
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model weights and metadata to a checkpoint file.
        Automatically creates parent directories if needed.
        
        Args:
            path: Destination file path
            classes: List of class names
            epoch: Training epoch (optional)
            metadata: Additional metadata dictionary (optional)
        """
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "task": "step-classification",
            "schema_version": 2,
            "model_state_dict": self.state_dict(),
            "classes": list(classes),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "backbone": "mobilenet_v2",
            "num_classes": len(classes),
            "epoch": epoch,
        }
        if metadata:
            # Merge extra metadata without overwriting core schema keys
            protected_keys = set(checkpoint.keys())
            for key, val in metadata.items():
                if key not in protected_keys:
                    checkpoint[key] = val

        torch.save(checkpoint, dest_path)

    def load_backbone_weights(self, checkpoint_path: Union[str, Path]):
        """
        Load pretrained backbone features from a domain checkpoint.
        Supports keys starting with 'backbone.features.' or 'features.'.
        
        Args:
            checkpoint_path: Path to source checkpoint containing backbone weights
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        feature_weights = {}
        for key, value in state_dict.items():
            if key.startswith("backbone.features."):
                feature_weights[key] = value
            elif key.startswith("features."):
                feature_weights[f"backbone.{key}"] = value

        if not feature_weights:
            raise ValueError(
                f"No backbone feature weights found in checkpoint: {checkpoint_path}. "
                "Expected keys starting with 'backbone.features.' or 'features.'"
            )

        return self.load_state_dict(feature_weights, strict=False)

    def get_trainable_params(self) -> Tuple[int, int]:
        """นับจำนวน parameters ที่ train ได้"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# Backward compatibility alias
OrigamiClassifier = StepModel


def create_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> StepModel:
    """
    สร้าง model พร้อมใช้งาน
    
    Args:
        num_classes: จำนวน class
        pretrained: ใช้ pre-trained weights
        freeze_backbone: freeze backbone
    
    Returns:
        model: StepModel instance
    """
    model = StepModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    trainable, total = model.get_trainable_params()
    print(f"[MODEL] StepModel created")
    print(f"  - Classes: {num_classes}")
    print(f"  - Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    return model
