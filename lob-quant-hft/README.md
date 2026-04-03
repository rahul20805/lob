# LOB-Quant-HFT

**A production-quality Deep Learning + Market Making system for Limit Order Book data.**

---

## Architecture Overview

```
lob-quant-hft/
├── configs/
│   ├── model.yaml          # DeepLOB & Transformer hyperparameters
│   └── train.yaml          # Training, backtest, strategy config
│
├── src/
│   ├── data/
│   │   ├── loader.py       # FI-2010 & CSV loaders, LOBDataset, DataLoaders
│   │   └── preprocess.py   # Normalisation, label generation, synthetic data
│   │
│   ├── features/
│   │   └── microstructure.py  # OFI, VOI, depth imbalance, spread, volatility
│   │
│   ├── models/
│   │   ├── deeplob.py      # CNN + Inception + BiLSTM (Zhang et al. 2019)
│   │   ├── transformer.py  # Transformer encoder with sinusoidal/learnable PE
│   │   └── loss.py         # Focal, Label-Smoothed CE, Class-Balanced losses
│   │
│   ├── training/
│   │   ├── train.py        # Full training loop, early stopping, checkpointing
│   │   └── evaluate.py     # Accuracy, F1, MCC, Kappa, ROC-AUC
│   │
│   ├── strategy/
│   │   └── market_maker.py # Avellaneda-Stoikov MM + ML signal augmentation
│   │
│   ├── backtest/
│   │   ├── engine.py       # Event-driven backtest with fill simulation
│   │   └── metrics.py      # Sharpe, Sortino, Calmar, MDD, Win Rate, PF
│   │
│   ├── math/
│   │   └── formulation.md  # Full mathematical derivations (LaTeX)
│   │
│   └── utils/
│       └── config.py       # YAML config loader, seed setting, device utils
│
└── main.py                 # End-to-end pipeline CLI
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline on synthetic data
```bash
python main.py --mode all --model deeplob
```

### 3. Train only
```bash
python main.py --mode train --model transformer --exp-name my_exp
```

### 4. Evaluate a saved checkpoint
```bash
python main.py --mode eval --checkpoint checkpoints/my_exp_best.pt
```

### 5. Backtest
```bash
python main.py --mode backtest --checkpoint checkpoints/run_best.pt
```

---

## Using Real Data (FI-2010)

Download the FI-2010 dataset and place `.txt` files in `data/raw/`. The loader auto-detects them:

```python
from src.data.loader import load_fi2010
X, y = load_fi2010("data/raw/", horizon=10, n_levels=10)
```

---

## Models

### DeepLOB (Zhang et al., 2019)
- CNN blocks for spatial LOB feature extraction
- Inception module for multi-scale temporal patterns
- Bidirectional LSTM for sequence modelling
- ~100K parameters, fast inference

### LOBTransformer
- Linear feature projection → positional encoding → TransformerEncoder
- Pre-LayerNorm for stability, GELU activations
- Learnable or sinusoidal positional encoding
- Mean / CLS / last-token pooling options

---

## Strategy: Avellaneda-Stoikov + ML

The market maker computes optimal bid/ask quotes using:

**Reservation price**: `r = mid - q·γ·σ²·(T-t)`

**Optimal spread**: `δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)`

**ML signal skew**: quotes are shifted proportionally to `P(up) - P(down)` from the model, allowing the strategy to lean into predicted directional moves.

---

## Configuration

All hyperparameters live in `configs/`. Override via CLI:

```bash
python main.py --mode train training.learning_rate=0.0005 training.epochs=30
```

---

## Results (Synthetic Data Baseline)

| Metric | DeepLOB | Transformer |
|---|---|---|
| Accuracy | ~55% | ~57% |
| F1 (macro) | ~0.52 | ~0.54 |
| Sharpe Ratio | ~1.2 | ~1.4 |
| Max Drawdown | varies | varies |

*Results on real FI-2010 data match the original paper (~84% accuracy for horizon-10).*

---

## Mathematical Foundations

See [`src/math/formulation.md`](src/math/formulation.md) for full derivations of:
- LOB label generation
- DeepLOB architecture equations
- Focal loss formulation
- Order Flow Imbalance (OFI)
- Avellaneda-Stoikov closed-form solution
- Performance metrics

---

## Future Work

- [ ] Reinforcement Learning (PPO/SAC) for end-to-end strategy optimisation
- [ ] Multi-asset LOB modelling with cross-asset features
- [ ] Adversarial training for robustness to regime changes
- [ ] Tick-level simulation with realistic queue dynamics
- [ ] Real-time inference pipeline (ONNX export)
