from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from attacks.fgsm import Normalization
from defenses.adversarial_training import AdvTrainConfig, adversarial_training_step
from model import get_model


# CIFAR-10 normalization
CIFAR10_NORM = Normalization(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: str,
) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORM.mean, CIFAR10_NORM.std),
        ]
    )
    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORM.mean, CIFAR10_NORM.std),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=tf_train
    )
    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=tf_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,   # 🔧 FIX: CPU-safe
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,   # 🔧 FIX: CPU-safe
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(1, total)


def train_one_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
    }


def train_one_epoch_adv(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: AdvTrainConfig,
) -> dict:
    total_loss = 0.0
    total_clean_acc = 0.0
    total_adv_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        metrics = adversarial_training_step(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            loss_fn=loss_fn,
            norm=CIFAR10_NORM,
            cfg=cfg,
        )

        bs = y.size(0)
        total_loss += metrics["loss"] * bs
        total_clean_acc += metrics["acc_clean"] * bs
        total_adv_acc += metrics["acc_adv"] * bs
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "acc_clean": total_clean_acc / max(1, n),
        "acc_adv": total_adv_acc / max(1, n),
    }


def main() -> None:
    p = argparse.ArgumentParser()

    # Model & training
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
    p.add_argument("--seed", type=int, default=42)

    # Defense (FGSM)
    p.add_argument("--adv-train", action="store_true", help="Enable FGSM adversarial training")
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--adv-ratio", type=float, default=0.5)

    # Output
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "checkpoints", "cifar10_model.pt"
        ),
    )

    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    model = get_model(args.model, num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        gamma=0.1,
    )

    best_acc = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"adv_train={args.adv_train} epsilon={args.epsilon} adv_ratio={args.adv_ratio}")

    for epoch in range(1, args.epochs + 1):
        if args.adv_train:
            cfg = AdvTrainConfig(epsilon=args.epsilon, adv_ratio=args.adv_ratio)
            train_metrics = train_one_epoch_adv(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                cfg=cfg,
            )
            msg = (
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc_clean={train_metrics['acc_clean']:.4f} "
                f"train_acc_adv={train_metrics['acc_adv']:.4f}"
            )
        else:
            train_metrics = train_one_epoch_standard(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
            msg = (
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f}"
            )

        test_acc = evaluate_clean(model, test_loader, device)
        msg += f" test_acc={test_acc:.4f} lr={optimizer.param_groups[0]['lr']:.5f}"
        print(msg)

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model": args.model,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
                "cifar10_norm": {
                    "mean": list(CIFAR10_NORM.mean),
                    "std": list(CIFAR10_NORM.std),
                },
                "adv_train": args.adv_train,
                "adv_train_cfg": asdict(
                    AdvTrainConfig(epsilon=args.epsilon, adv_ratio=args.adv_ratio)
                ),
                "seed": args.seed,
            }
            torch.save(ckpt, args.out)

        scheduler.step()

    print(f"done. best_test_acc={best_acc:.4f}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
