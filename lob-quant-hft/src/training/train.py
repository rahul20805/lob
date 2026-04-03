"""
Training Loop
=============
Handles:
  - Mixed-precision training (torch.cuda.amp)
  - Gradient clipping
  - Early stopping
  - LR scheduling (CosineAnnealingLR / OneCycleLR)
  - Checkpoint saving / loading
  - TensorBoard-compatible logging (via dict)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best: bool = True,
    ) -> None:
        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.restore_best = restore_best

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_state: dict | None = None
        self.counter      = 0
        self.stopped      = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        self.stopped = self.counter >= self.patience
        return self.stopped

    def restore(self, model: nn.Module) -> None:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Restored best model weights.")


# ---------------------------------------------------------------------------
# Optimiser / scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adamw").lower()
    lr   = float(cfg.get("learning_rate", 1e-4))
    wd   = float(cfg.get("weight_decay", 1e-5))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name   = cfg.get("lr_scheduler", "cosine").lower()
    epochs = int(cfg.get("epochs", 50))

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("learning_rate", 1e-4)) * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    if name == "none" or name is None:
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip: float = 1.0,
    train: bool = True,
    scheduler_step_batch: bool = False,
    scheduler: object = None,
) -> dict[str, float]:
    """Run one training or validation epoch."""

    model.train(train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(X)
                loss   = criterion(logits, y)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if scheduler_step_batch and scheduler is not None:
                    scheduler.step()

            preds   = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            total_loss += loss.item() * y.size(0)

    return {
        "loss":     total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    cfg: dict,
    device: torch.device,
    checkpoint_dir: str | Path = "checkpoints/",
    experiment_name: str = "run",
    callback: Callable[[dict], None] | None = None,
) -> dict[str, list]:
    """
    Full training loop.

    Parameters
    ----------
    model           : the neural network
    train_loader    : DataLoader for training set
    val_loader      : DataLoader for validation set
    criterion       : loss function
    cfg             : training config dict
    device          : torch.device
    checkpoint_dir  : directory to save checkpoints
    experiment_name : used for checkpoint filenames
    callback        : optional per-epoch callback; receives metrics dict

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer  = build_optimizer(model, cfg)
    steps_pe   = len(train_loader)
    scheduler  = build_scheduler(optimizer, cfg, steps_pe)
    use_onecycle = cfg.get("lr_scheduler", "cosine").lower() == "onecycle"

    scaler: torch.cuda.amp.GradScaler | None = (
        torch.cuda.amp.GradScaler() if cfg.get("mixed_precision", True)
        and device.type == "cuda"
        else None
    )

    early_stopper = EarlyStopping(
        patience=int(cfg.get("patience", 10)),
        min_delta=float(cfg.get("min_delta", 1e-4)),
        mode="min",
    )

    grad_clip = float(cfg.get("gradient_clip", 1.0))
    epochs    = int(cfg.get("epochs", 50))

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    model.to(device)
    logger.info("Starting training for %d epochs", epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            grad_clip=grad_clip, train=True,
            scheduler_step_batch=use_onecycle, scheduler=scheduler,
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, None, device, scaler=None,
            train=False,
        )

        # Step epoch-level schedulers
        if scheduler is not None and not use_onecycle:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.3f | "
            "val_loss=%.4f acc=%.3f | lr=%.2e | %.1fs",
            epoch, epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"],   val_metrics["accuracy"],
            current_lr, elapsed,
        )

        if callback:
            callback({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
                "lr": current_lr,
            })

        # Early stopping
        if early_stopper(val_metrics["loss"], model):
            logger.info("Early stopping at epoch %d", epoch)
            break

        # Periodic checkpoint
        if epoch % max(1, epochs // 10) == 0:
            ckpt_path = checkpoint_dir / f"{experiment_name}_epoch{epoch:03d}.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)

    early_stopper.restore(model)

    # Save final model
    final_path = checkpoint_dir / f"{experiment_name}_best.pt"
    save_checkpoint(model, optimizer, epoch, val_metrics, final_path)
    logger.info("Training complete. Best val_loss=%.4f", early_stopper.best_score)

    return history


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":        metrics,
    }, path)
    logger.debug("Checkpoint saved: %s", path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    path = Path(path)
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return ckpt
