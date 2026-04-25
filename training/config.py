import torch
import os

# ── DEVICE CONFIGURATION ─────────────────────────────────────────────────────
def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_best_device()

# ── DATA PATHS ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(DATA_DIR, "grid_dataset_neurips2020.jsonl")

# ── HYPERPARAMETERS ──────────────────────────────────────────────────────────
# NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]

# Auto-scale config based on device
if DEVICE.type == "cuda":
    # Research PC — full scale
    TRAIN_CONFIG = {
        "epochs": 100,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "hidden_channels": [128, 256, 512],
        "heads": [8, 8, 1],
        "loc_loss_weight": 0.5
    }
else:
    # Personal PC — smoke test only
    TRAIN_CONFIG = {
        "epochs": 3,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "hidden_channels": [64, 128, 256],
        "heads": [4, 4, 1],
        "loc_loss_weight": 0.5
    }

# ── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
# These dimensions are fixed by the GridDataset implementation in pyg_data.py
NODE_FEATURES = 4  # load_p, gen_p, mean_v_or, max_rho
EDGE_FEATURES = 3  # rho, p_or, q_or

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
