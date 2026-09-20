
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eng_dataset import ECGDataset
from model import ECGCNN


SEED = 42
BATCH_SIZE = 16
EPOCHS = 18
LEARNING_RATE = 0.00257
FOCAL_GAMMA = 2.0
SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8
NUM_CLASSES = 94
SIGNAL_LENGTH = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
LATEST_DIR = EXPERIMENT_ROOT / "residual_focal_latest"
BEST_DIR = EXPERIMENT_ROOT / "residual_focal_best"
HISTORY_PATH = EXPERIMENT_ROOT / "residual_focal_training_history.csv"
CONFIG_PATH = EXPERIMENT_ROOT / "residual_focal_config.json"
ARCHITECTURE_PATH = EXPERIMENT_ROOT / "residual_focal_architecture.txt"


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probabilities = torch.sigmoid(inputs)
        pt = (
            probabilities * targets
            + (1 - probabilities) * (1 - targets)
        )
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def verify_label_schema(datasets):
    reference_name, reference = datasets[0]
    for name, dataset in datasets[1:]:
        if dataset.label_columns != reference.label_columns:
            raise ValueError(
                f"Label columns differ between {reference_name} and {name}."
            )


def clear_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def verify_model(model):
    test_input = torch.randn(4, 12, SIGNAL_LENGTH, device=DEVICE)
    test_output = model(test_input)
    expected_input = (4, 12, SIGNAL_LENGTH)
    expected_output = (4, NUM_CLASSES)

    if tuple(test_input.shape) != expected_input:
        raise ValueError(f"Unexpected input shape: {test_input.shape}")
    if tuple(test_output.shape) != expected_output:
        raise ValueError(f"Unexpected output shape: {test_output.shape}")
    if not torch.isfinite(test_output).all():
        raise ValueError("Model forward pass produced NaN or Inf values.")

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Input shape: {tuple(test_input.shape)}")
    print(f"Output shape: {tuple(test_output.shape)}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_dataset = ECGDataset(PROJECT_ROOT / "train_split.csv")
    val_dataset = ECGDataset(PROJECT_ROOT / "val_split.csv")
    verify_label_schema([
        ("train", train_dataset),
        ("validation", val_dataset),
    ])

    if len(train_dataset.label_columns) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} labels, found "
            f"{len(train_dataset.label_columns)}"
        )

    positive_counts = torch.tensor(
        train_dataset.df[train_dataset.label_columns].sum(axis=0).to_numpy(),
        dtype=torch.float32,
    )
    negative_counts = len(train_dataset) - positive_counts
    pos_weight = torch.where(
        positive_counts > 0,
        negative_counts / positive_counts,
        torch.ones_like(positive_counts),
    ).clamp(max=20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)
    verify_model(model)
    model_architecture = str(model)

    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        pos_weight=pos_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
    )

    clear_directory(LATEST_DIR)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = BEST_DIR / "model.pth"
    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "focal_gamma": FOCAL_GAMMA,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "gradient_clip_max_norm": 1.0,
        "num_classes": NUM_CLASSES,
        "signal_shape": [12, SIGNAL_LENGTH],
        "device": str(DEVICE),
        "loss": "FocalLoss + BCEWithLogitsLoss(pos_weight)",
        "focal_loss_formulation": (
            "bce = BCEWithLogitsLoss(pos_weight, reduction='none'); "
            "p = sigmoid(logits); pt = p*y + (1-p)*(1-y); "
            "loss = mean((1-pt)^gamma * bce)"
        ),
        "train_dataset": "train_split.csv",
        "validation_dataset": "val_split.csv",
        "experiment": "residual_focal",
    }
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    ARCHITECTURE_PATH.write_text(model_architecture)

    with HISTORY_PATH.open("w", newline="") as history_file:
        csv.writer(history_file).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "mean_gradient_norm",
                "max_gradient_norm",
                "clipped_batches",
            ]
        )

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        clipped_batches = 0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
        )
        for signals, labels in train_progress:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(signals)
            if not torch.isfinite(outputs).all():
                raise ValueError("Training forward pass produced NaN or Inf.")
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became NaN or Inf.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            gradient_norm_value = float(gradient_norm.item())
            if not np.isfinite(gradient_norm_value):
                raise ValueError("Gradient norm became NaN or Inf.")
            gradient_norm_sum += gradient_norm_value
            gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
            clipped_batches += gradient_norm_value > 1.0
            optimizer.step()
            train_loss += loss.item() * signals.size(0)
            train_progress.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                grad_norm=f"{gradient_norm_value:.3f}",
            )

        train_loss /= len(train_loader.dataset)
        mean_gradient_norm = gradient_norm_sum / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
            )
            for signals, labels in val_progress:
                signals = signals.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(signals)
                if not torch.isfinite(outputs).all():
                    raise ValueError("Validation output produced NaN or Inf.")
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Validation loss became NaN or Inf.")
                val_loss += loss.item() * signals.size(0)
                val_progress.set_postfix(batch_loss=f"{loss.item():.4f}")

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "model_architecture": model_architecture,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "learning_rate": current_lr,
            "seed": SEED,
            "completed_epochs": epoch + 1,
            "mean_gradient_norm": mean_gradient_norm,
            "max_gradient_norm": gradient_norm_max,
            "clipped_batches": clipped_batches,
        }

        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint["best_epoch"] = best_epoch
        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["completed_epochs"] = epoch + 1
        checkpoint["learning_rate"] = current_lr
        torch.save(checkpoint, LATEST_DIR / "model.pth")
        if is_new_best:
            torch.save(checkpoint, best_checkpoint_path)

        with HISTORY_PATH.open("a", newline="") as history_file:
            csv.writer(history_file).writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    current_lr,
                    mean_gradient_norm,
                    gradient_norm_max,
                    clipped_batches,
                ]
            )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Grad Norm: {mean_gradient_norm:.3f} | "
            f"Clipped: {clipped_batches}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining complete!")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint: {LATEST_DIR / 'model.pth'}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
