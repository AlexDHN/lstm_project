# LSTM Stock Return Prediction & Trading Strategy

> Academic project — Université Paris-Dauphine  
> LSTM-based return forecasting on the OMXN40 index with systematic weight initialization benchmarking and full trading backtest.

---

## Overview

This project builds an end-to-end pipeline for **equity return prediction using a Long Short-Term Memory (LSTM) neural network**, applied to the Nordic large-cap index **OMXN40**. The US 10-year Treasury yield (`10yUS.csv`) is used as an additional macro feature.

The central research question is practical: **does the choice of weight initialization function materially affect the out-of-sample predictive performance and trading profitability of an LSTM?** Eleven initialization strategies are benchmarked systematically, from standard schemes (Glorot, orthogonal) to degenerate ones (zeros, ones, constant), to quantify their impact on convergence and final strategy returns.

---

## Repository Structure

```
.
├── classes/                    # Core model definition (LSTM architecture in PyTorch)
├── utils/
│   └── strategy_trading.py     # Full pipeline: train → predict → backtest → compare
├── models/                     # Saved model weights per initialization function
│
├── dataset.csv                 # Processed feature dataset (returns, macro features)
├── raw_dataset.csv             # Raw price/yield data before feature engineering
├── OMXN40.csv                  # OMXN40 index historical prices
├── 10yUS.csv                   # US 10-year Treasury yield historical data
│
├── main.py                     # Entry point: runs the full strategy across all inits
├── example_train.ipynb         # Step-by-step walkthrough of model training
├── example_backtest.ipynb      # Standalone backtest visualization and P&L analysis
├── Statistics.ipynb            # Statistical analysis of predictions and strategy returns
└── README.md
```

---

## Model & Pipeline

### LSTM architecture (`classes/`)

A configurable LSTM network implemented in PyTorch with:

- `window_size` lookback period (default: 30 days) to construct input sequences
- `hidden_number` stacked LSTM layers with `hidden_size` hidden units
- A linear output head predicting the next-period return

### Training (`utils/strategy_trading.py`)

The `strategy()` function orchestrates the complete workflow:

1. **Sequence construction** — sliding window over the feature set
2. **Train/test split** — strict temporal split (no lookahead)
3. **Training** — Adam optimizer with L2 regularization (`weight_decay`), `num_epochs` epochs, `batch_size` samples per step
4. **Multi-init sweep** — for each initialization function in the list, a fresh model is trained, saved to `models/`, and its predictions recorded
5. **Signal generation** — predicted return sign → long/flat position
6. **Backtest** — cumulative P&L, Sharpe ratio, drawdown computed per initialization

### Weight initialization benchmark

Eleven schemes are evaluated:

| Category | Functions |
|---|---|
| Standard random | `random_normal`, `random_uniform`, `truncated_normal` |
| Variance-preserving | `glorot_normal`, `glorot_uniform`, `variance_scaling` |
| Structure-preserving | `orthogonal`, `identity` |
| Degenerate | `zeros`, `ones`, `constant_` |

### Hyperparameters (defaults in `main.py`)

| Parameter | Value |
|---|---|
| Learning rate | 0.0075 |
| Batch size | 1000 |
| Epochs | 100 |
| Window size | 30 days |
| LSTM layers | 1 |
| Hidden units | 3 |
| Weight decay | 1e-5 |

---

## Data

| File | Description |
|---|---|
| `OMXN40.csv` | Nordic 40 large-cap index — daily closing prices |
| `10yUS.csv` | US 10-year Treasury yield — daily series |
| `raw_dataset.csv` | Merged raw data before feature engineering |
| `dataset.csv` | Final feature matrix used for training (log-returns, normalized yield) |

---

## Getting Started

### Prerequisites

```bash
pip install torch pandas numpy matplotlib
```

### Run the full strategy sweep

```python
# main.py
import pandas as pd
from utils.strategy_trading import strategy

data = pd.read_csv("dataset.csv", index_col=0)
data.index = pd.to_datetime(data.index)

kwargs = dict(
    learning_rate=0.0075,
    batch_size=1000,
    num_epochs=100,
    window_size=30,
    hidden_number=1,
    hidden_size=3,
    weight_decay=1e-5,
    override=False,   # Set True to retrain even if saved models exist
    path="models/"
)

init_functions = [
    "random_normal", "random_uniform", "truncated_normal",
    "zeros", "ones", "glorot_normal", "glorot_uniform",
    "identity", "orthogonal", "constant_", "variance_scaling",
]

df, df_pred = strategy(data, init_functions, **kwargs)
```

`df` contains per-initialization strategy performance metrics. `df_pred` contains the raw return predictions for each initialization, aligned on the test set dates.

Setting `override=False` skips re-training if a saved model already exists in `models/` for a given initialization — useful for iterating on the backtest without re-running training.

### Explore interactively

Open the notebooks in order:

1. **`example_train.ipynb`** — understand how a single LSTM is built, trained, and its loss curve inspected
2. **`example_backtest.ipynb`** — visualize the P&L curve and trading signals for a given initialization
3. **`Statistics.ipynb`** — cross-initialization comparison: prediction accuracy, Sharpe ratios, drawdown analysis

---

## Key Design Decisions

**Strict temporal train/test split.** No shuffling of sequences across time, preventing any form of lookahead bias in the backtest.

**Model caching via `override` flag.** Once trained, models are persisted to `models/` and reloaded on subsequent runs unless `override=True`. This makes the initialization comparison fully reproducible without re-running 11 training loops each time.

**Signal simplicity.** The trading rule is deliberately minimal — long when predicted return > 0, flat otherwise — to isolate the effect of initialization quality on raw predictive signal, without confounding it with position sizing or execution assumptions.
