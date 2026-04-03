"""
Configuration utilities — loads and merges YAML configs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config(dict):
    """A dictionary subclass that supports dot-access for nested keys."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __repr__(self) -> str:
        import json
        return json.dumps(self, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(*paths: str | Path, overrides: dict | None = None) -> Config:
    """
    Load one or more YAML config files and merge them left-to-right.
    Optional *overrides* dict is applied last.

    Example
    -------
    >>> cfg = load_config("configs/model.yaml", "configs/train.yaml")
    >>> cfg.training.batch_size
    64
    """
    merged: dict = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)

    if overrides:
        merged = _deep_merge(merged, overrides)

    return Config(merged)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "cuda") -> torch.device:
    """Return the best available device given a preference string."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_config(cfg: dict, path: str | Path) -> None:
    """Persist a config dict back to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI-style override parser
# ---------------------------------------------------------------------------

def parse_overrides(args: list[str]) -> dict:
    """
    Parse key=value pairs from a list of strings into a nested dict.

    Example
    -------
    >>> parse_overrides(["training.lr=0.001", "data.batch_size=32"])
    {'training': {'lr': 0.001}, 'data': {'batch_size': 32}}
    """
    overrides: dict = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, raw_val = arg.split("=", 1)
        val = _cast(raw_val)
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return overrides


def _cast(val: str) -> Any:
    """Try to cast a string to int, float, bool, or leave as str."""
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
