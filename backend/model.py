import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small CNN that trains quickly on CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes: int = 10):
    """ResNet-18 adjusted for CIFAR-10 (32x32 inputs).

    - 3x3 conv stride=1, padding=1
    - remove the initial maxpool
    """

    try:
        from torchvision.models import resnet18
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for ResNet-18. Install torchvision."
        ) from e

    model = resnet18(weights=None)

    # CIFAR-friendly stem
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    name = name.lower().strip()
    if name in {"resnet18", "resnet-18", "resnet"}:
        return get_resnet18(num_classes=num_classes)
    if name in {"cnn", "simplecnn", "simple"}:
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Use 'resnet18' or 'cnn'.")
