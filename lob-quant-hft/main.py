#!/usr/bin/env python3
"""
LOB-Quant-HFT — Main Entry Point
==================================
Usage examples:

  # Train DeepLOB on synthetic data
  python main.py --mode train --model deeplob

  # Train transformer on synthetic data
  python main.py --mode train --model transformer

  # Evaluate a saved checkpoint
  python main.py --mode eval --checkpoint checkpoints/run_best.pt

  # Run backtest on synthetic data
  python main.py --mode backtest --checkpoint checkpoints/run_best.pt

  # Full pipeline (train → eval → backtest)
  python main.py --mode all --model deeplob
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from src.utils.config import load_config, set_seed, get_device, ensure_dirs
from src.data.preprocess import generate_synthetic_lob, preprocess_pipeline
from src.data.loader import split_dataset, make_dataloaders
from src.models.transformer import build_model
from src.models.loss import build_loss
from src.training.train import train as train_model, load_checkpoint
from src.training.evaluate import evaluate, print_confusion_matrix, print_report
from src.strategy.market_maker import AvellanedaStoikovMM
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_data(cfg, args):
    """Load or generate LOB data and return split datasets + raw arrays."""
    logger.info("=== Data Stage ===")

    data_dir = Path(cfg.get("paths", {}).get("data_raw", "data/raw/"))
    fi2010_files = list(data_dir.glob("*.txt"))

    if fi2010_files:
        from src.data.loader import load_fi2010
        logger.info("Loading FI-2010 dataset from %s", data_dir)
        X, y = load_fi2010(
            data_dir,
            horizon=cfg.get("data", {}).get("horizon", 10),
            n_levels=cfg.get("data", {}).get("levels", 10),
            normalize=False,   # we normalise in pipeline
        )
    else:
        logger.info("No FI-2010 data found — generating synthetic LOB data")
        X, y = generate_synthetic_lob(
            n_ticks=50_000,
            n_levels=cfg.get("features", {}).get("n_levels", 10),
            seed=cfg.get("experiment", {}).get("seed", 42),
        )

    norm_method = cfg.get("data", {}).get("normalization", "zscore")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), normalizer = (
        preprocess_pipeline(X, y, norm_method=norm_method)
    )

    window  = cfg.get("data", {}).get("window_size", 100)
    bs      = cfg.get("training", {}).get("batch_size", 64)
    workers = cfg.get("training", {}).get("num_workers", 0)

    train_ds, val_ds, test_ds = split_dataset(
        np.concatenate([X_train, X_val, X_test]),
        np.concatenate([y_train, y_val, y_test]),
        train_ratio=0.7, val_ratio=0.1,
        window_size=window,
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=bs, num_workers=workers, pin_memory=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "X_raw": X,
        "y_raw": y,
        "X_test_raw": X_test,
        "n_features": X.shape[-1],
    }


def stage_train(cfg, data, device, args):
    """Build model, train, return trained model."""
    logger.info("=== Training Stage ===")

    model_cfg   = load_config("configs/model.yaml")
    model_type  = args.model
    model = build_model(dict(model_cfg), model_type=model_type)
    model.to(device)

    logger.info("Model: %s  Parameters: %d", model_type, model.count_parameters())

    # Loss
    train_labels = data["train_loader"].dataset.y[
        torch.tensor(data["train_loader"].dataset._indices)
    ].numpy()
    samples_per_class = list(np.bincount(train_labels, minlength=3))

    loss_cfg = cfg.get("training", {})
    criterion = build_loss(
        loss_type=loss_cfg.get("loss", "focal"),
        num_classes=3,
        alpha=float(model_cfg.get("training", {}).get("alpha", 0.5)),
        gamma=float(model_cfg.get("training", {}).get("gamma", 2.0)),
        smoothing=float(model_cfg.get("training", {}).get("label_smoothing", 0.1)),
        samples_per_class=samples_per_class,
    )

    ensure_dirs("checkpoints/", "logs/")

    history = train_model(
        model=model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        criterion=criterion,
        cfg=dict(cfg.get("training", {})),
        device=device,
        checkpoint_dir="checkpoints/",
        experiment_name=args.exp_name,
    )

    return model, criterion, history


def stage_eval(cfg, data, model, criterion, device):
    """Evaluate on test set and print results."""
    logger.info("=== Evaluation Stage ===")

    results = evaluate(
        model=model,
        loader=data["test_loader"],
        device=device,
        criterion=criterion,
        return_probs=True,
    )

    print("\n── Classification Report ──────────────────")
    print_report(results["report"])
    print("\n── Confusion Matrix ───────────────────────")
    print_confusion_matrix(results["confusion_matrix"])

    return results


def stage_backtest(cfg, data, model, device):
    """Run backtest and print metrics."""
    logger.info("=== Backtest Stage ===")

    bt_cfg  = cfg.get("backtest", {})
    str_cfg = cfg.get("strategy", {})

    strategy = AvellanedaStoikovMM(
        gamma=float(str_cfg.get("gamma", 0.1)),
        kappa=float(str_cfg.get("kappa", 1.5)),
        sigma=float(str_cfg.get("sigma", 0.02)),
        T=float(str_cfg.get("T", 1.0)),
        dt=float(str_cfg.get("dt", 0.001)),
        max_inventory=float(bt_cfg.get("max_position", 1000)),
        order_size=100.0,
        signal_weight=0.3,
    )

    engine = BacktestEngine(
        model=model,
        strategy=strategy,
        transaction_cost=float(bt_cfg.get("transaction_cost", 0.0001)),
        slippage=float(bt_cfg.get("slippage", 0.00005)),
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000)),
        device=device,
    )

    X_bt = data["X_test_raw"]
    result = engine.run(X_bt, window_size=100, step=1)
    metrics = compute_metrics(
        result,
        risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.05)),
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000)),
    )
    return result, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LOB-Quant-HFT")
    p.add_argument("--mode",       default="all",
                   choices=["train", "eval", "backtest", "all"])
    p.add_argument("--model",      default="deeplob",
                   choices=["deeplob", "transformer"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint for eval/backtest modes")
    p.add_argument("--exp-name",   default="run", dest="exp_name")
    p.add_argument("--device",     default=None,
                   help="Override device (cuda/cpu/mps)")
    p.add_argument("--config",     nargs="+",
                   default=["configs/model.yaml", "configs/train.yaml"])
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(*args.config)
    device = get_device(args.device or cfg.get("experiment", {}).get("device", "cpu"))
    seed   = int(cfg.get("experiment", {}).get("seed", 42))

    set_seed(seed)
    logger.info("Device: %s  Seed: %d  Mode: %s  Model: %s",
                device, seed, args.mode, args.model)

    ensure_dirs("data/raw", "data/processed", "checkpoints", "logs", "results")

    # ── Data ─────────────────────────────────────────────────────────────────
    data = stage_data(cfg, args)

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = load_config("configs/model.yaml")
    model = build_model(dict(model_cfg), model_type=args.model)
    model.to(device)

    criterion = build_loss(loss_type="focal", num_classes=3)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, device=device)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── Run stages ───────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        model, criterion, history = stage_train(cfg, data, device, args)
        logger.info("Final train_loss=%.4f  val_loss=%.4f",
                    history["train_loss"][-1], history["val_loss"][-1])

    if args.mode in ("eval", "all"):
        stage_eval(cfg, data, model, criterion, device)

    if args.mode in ("backtest", "all"):
        stage_backtest(cfg, data, model, device)

    logger.info("Done.")


if __name__ == "__main__":
    main()
