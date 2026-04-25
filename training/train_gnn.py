import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
import argparse
from tqdm import tqdm
try:
    from training.config import DEVICE, DATA_FILE, DATA_DIR, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED
except ImportError:
    from config import DEVICE, DATA_FILE, DATA_DIR, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED

from scripts.pyg_data import PreloadedGridDataset, GridEnvMetadata, LABEL_MAP
from scripts.split import get_splits, compute_class_weights


class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, n_classes, hidden_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_channels[0], heads=heads[0],
                             edge_dim=edge_features, dropout=dropout)
        self.conv2 = GATConv(hidden_channels[0] * heads[0], hidden_channels[1],
                             heads=heads[1], edge_dim=edge_features, dropout=dropout)
        self.conv3 = GATConv(hidden_channels[1] * heads[1], hidden_channels[2],
                             heads=heads[2], edge_dim=edge_features, dropout=dropout)

        last_dim = hidden_channels[2] * heads[2]
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(last_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        loc_logits   = self.localizer(x).squeeze(-1)
        graph_emb    = global_mean_pool(x, batch)
        class_logits = self.classifier(graph_emb)
        return class_logits, loc_logits


def build_loc_targets_fast(batch):
    """
    Vectorized, no Python loop, no .item() calls — stays on device.

    Strategy:
      Each graph i has n_nodes_i nodes. fault_loc[i] is the target node
      index *within* graph i (substation index). We need the global node
      index = cumulative node offset for graph i + fault_loc[i].

      batch.ptr gives cumulative node counts: ptr[i] is the start index
      of graph i in the flattened node list. So global_idx = ptr[i] + fl[i].
      We only set the target for graphs where fault_loc >= 0.
    """
    n_nodes  = batch.x.size(0)
    device   = batch.x.device
    targets  = torch.zeros(n_nodes, dtype=torch.float, device=device)

    fault_loc = batch.fault_loc          # (B,) — already on device after batch.to(device)
    ptr       = batch.ptr                # (B+1,) cumulative node offsets

    valid_mask   = fault_loc >= 0                          # (B,) bool
    valid_graphs = valid_mask.nonzero(as_tuple=True)[0]    # indices of graphs with a fault

    if valid_graphs.numel() > 0:
        # global node index = start of graph + local fault node
        global_idx = ptr[valid_graphs] + fault_loc[valid_graphs]
        # clamp to avoid rare edge case where fault_loc >= n_nodes in that graph
        n_nodes_per_graph = ptr[1:] - ptr[:-1]            # (B,)
        max_idx = ptr[valid_graphs] + n_nodes_per_graph[valid_graphs] - 1
        global_idx = torch.min(global_idx, max_idx)
        targets.scatter_(0, global_idx, 1.0)

    return targets


def make_dataloader(dataset, batch_size, shuffle):
    """
    num_workers=0 on Windows (PyG + multiprocessing is broken there).
    """
    is_accelerator = str(DEVICE).startswith("cuda") or str(DEVICE).startswith("xpu")
    is_win         = sys.platform == "win32"

    num_workers = 0 if is_win else 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=is_accelerator,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device, non_blocking=True)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total   += batch.num_graphs
    return correct / (total + 1e-9)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    args = parser.parse_args()

    epochs     = args.epochs     if args.epochs     is not None else TRAIN_CONFIG["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else TRAIN_CONFIG["batch_size"]
    lr         = args.lr         if args.lr         is not None else TRAIN_CONFIG["lr"]

    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")
    print("Environment: NeurIPS 2020 Track 1 Small")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta_dict = json.load(f)
        meta = GridEnvMetadata(meta_dict)
    else:
        print("Warning: meta JSON not found — falling back to Grid2Op init (slow).")
        meta = GridEnvMetadata()

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found.")
        sys.exit(1)

    # ── Splits ────────────────────────────────────────────────────────────────
    split_prefix   = "split_neurips2020"
    train_idx_path = os.path.join(DATA_DIR, f"{split_prefix}_train_idx.npy")
    val_idx_path   = os.path.join(DATA_DIR, f"{split_prefix}_val_idx.npy")

    if not (os.path.exists(train_idx_path) and os.path.exists(val_idx_path)):
        print(f"Error: Split files not found. Run: python scripts/split.py")
        sys.exit(1)
        
    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)

    pt_data_path = os.path.join(DATA_DIR, "processed_grid_data.pt")
    
    # 1. Load the ENTIRE dataset onto the GPU once
    full_dataset = PreloadedGridDataset(pt_data_path, device=DEVICE)

    # 2. PyG natively slices the dataset using your numpy arrays!
    train_ds = full_dataset[train_idx]
    val_ds   = full_dataset[val_idx]

    # ── Class weights ─────────────────────────────────────────────────────────
    from scripts.split import load_labels
    all_labels   = load_labels(DATA_FILE)
    train_labels = [all_labels[i] for i in train_idx]
    weights      = compute_class_weights(train_labels, LABEL_MAP)
    class_weights = torch.tensor(weights, dtype=torch.float, device=DEVICE)
    print(f"Class weights: {class_weights}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GridGNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        n_classes=len(LABEL_MAP),
        hidden_channels=TRAIN_CONFIG["hidden_channels"],
        heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)

    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs)
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loc_loss_fn = nn.BCEWithLogitsLoss()

    scaler = GradScaler(device=DEVICE.type)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # 1. Cast forward pass to mixed precision
            with autocast(device_type=DEVICE.type):
                class_logits, loc_logits = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                cls_loss = cls_loss_fn(class_logits, batch.y)

                has_fault = (batch.fault_loc >= 0).any()
                if has_fault:
                    loc_targets = build_loc_targets_fast(batch)
                    loc_loss = loc_loss_fn(loc_logits, loc_targets)
                else:
                    loc_loss = torch.tensor(0.0, device=DEVICE)

                loss = cls_loss + TRAIN_CONFIG["loc_loss_weight"] * loc_loss
            
            # 2. Scale the loss and call backward
            scaler.scale(loss).backward()
            
            # 3. Step the optimizer and update the scaler
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "gnn_checkpoint_best.pt")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()