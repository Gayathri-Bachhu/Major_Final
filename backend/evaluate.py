from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attacks.fgsm import Normalization, fgsm_attack
from model import get_model


DEFAULT_NORM = Normalization(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))


def get_test_loader(batch_size: int, num_workers: int, data_dir: str, norm: Normalization) -> DataLoader:
    from torchvision import datasets, transforms

    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm.mean, norm.std),
        ]
    )
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def accuracy_fgsm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
    norm: Normalization,
) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = fgsm_attack(model=model, x=x, y=y, epsilon=epsilon, loss_fn=loss_fn, norm=norm)
        pred = model(x_adv).argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(1, total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "cifar10_model.pt"),
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--epsilon", type=float, default=8 / 255)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model", "resnet18")

    norm_dict = ckpt.get("cifar10_norm")
    norm = DEFAULT_NORM
    if isinstance(norm_dict, dict) and "mean" in norm_dict and "std" in norm_dict:
        norm = Normalization(mean=tuple(norm_dict["mean"]), std=tuple(norm_dict["std"]))

    model = get_model(model_name, num_classes=10)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    loader = get_test_loader(args.batch_size, args.num_workers, args.data_dir, norm)

    clean_acc = accuracy(model, loader, device)
    adv_acc = accuracy_fgsm(model, loader, device, epsilon=args.epsilon, norm=norm)

    print(f"checkpoint={args.ckpt}")
    print(f"clean_acc={clean_acc:.4f}")
    print(f"fgsm_eps={args.epsilon} adv_acc={adv_acc:.4f}")


if __name__ == "__main__":
    main()
