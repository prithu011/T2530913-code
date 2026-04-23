import sys
import os

# Ensure we can import from root (for scripts.* and training.*)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse

# Import centralized configuration
try:
    from training.config import DEVICE, DATA_FILE, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED
except ImportError:
    from config import DEVICE, DATA_FILE, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED

from scripts.pyg_data import GridDataset, GridEnvMetadata, LABEL_MAP
from scripts.split import get_splits, compute_class_weights

class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, n_classes, hidden_channels, heads, dropout):
        super().__init__()
        # hidden_channels: [64, 128, 256]
        # heads: [4, 4, 1]
        
        self.conv1 = GATConv(node_features, hidden_channels[0], heads=heads[0], 
                             edge_dim=edge_features, dropout=dropout)
        
        # Input to conv2 is hidden_channels[0] * heads[0]
        self.conv2 = GATConv(hidden_channels[0] * heads[0], hidden_channels[1], 
                             heads=heads[1], edge_dim=edge_features, dropout=dropout)
        
        # Input to conv3 is hidden_channels[1] * heads[1]
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

def build_loc_targets(batch):
    targets = torch.zeros(batch.x.size(0), dtype=torch.float, device=batch.x.device)
    node_offset = 0
    # Use batch.batch to find node counts per graph
    for i in range(batch.num_graphs):
        fl = batch.fault_loc[i].item()
        n_nodes = (batch.batch == i).sum().item()
        if fl >= 0:
            # Clamp fault_loc to node range to avoid index errors
            # fl is substation index, which matches our node index
            targets[node_offset + fl] = 1.0
        node_offset += n_nodes
    return targets

def train():
    parser = argparse.ArgumentParser()
    # Optional overrides for config
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    # Load config (now fixed to NeurIPS)
    epochs     = args.epochs if args.epochs is not None else TRAIN_CONFIG["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else TRAIN_CONFIG["batch_size"]
    lr         = args.lr if args.lr is not None else TRAIN_CONFIG["lr"]

    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")
    print("Environment: NeurIPS 2020 Track 1 Small")

    # Load metadata and determine data file
    meta = GridEnvMetadata()
    data_file = DATA_FILE

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        sys.exit(1)

    print(f"Loading data from {data_file}...")
    
    # Check for pre-computed splits
    split_prefix = "split_neurips2020"
    train_idx_path = f"{split_prefix}_train_idx.npy"
    val_idx_path = f"{split_prefix}_val_idx.npy"
    
    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path):
        print("Using saved split indices.")
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
    else:
        print("Computing new splits (this may take a while for large files)...")
        train_idx, val_idx, test_idx = get_splits(data_file)
        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)
        np.save(f"{split_prefix}_test_idx.npy", test_idx)

    train_ds = GridDataset(data_file, train_idx, meta)
    val_ds   = GridDataset(data_file, val_idx, meta)

    # Compute class weights for loss
    print("Computing class weights...")
    from scripts.split import load_labels
    all_labels = load_labels(data_file)
    train_labels = [all_labels[i] for i in train_idx]
    weights = compute_class_weights(train_labels, LABEL_MAP)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Class weights: {class_weights}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = GridGNN(
        node_features=NODE_FEATURES, 
        edge_features=EDGE_FEATURES, 
        n_classes=len(LABEL_MAP),
        hidden_channels=TRAIN_CONFIG["hidden_channels"],
        heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"]
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loc_loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            class_logits, loc_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            cls_loss = cls_loss_fn(class_logits, batch.y)

            has_fault = (batch.fault_loc >= 0)
            if has_fault.any():
                loc_targets = build_loc_targets(batch)
                loc_loss = loc_loss_fn(loc_logits, loc_targets)
            else:
                loc_loss = torch.tensor(0.0, device=DEVICE)

            loss = cls_loss + TRAIN_CONFIG["loc_loss_weight"] * loc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                correct += (logits.argmax(dim=1) == batch.y).sum().item()
                total   += batch.num_graphs

        val_acc = correct / (total + 1e-9)
        print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "gnn_checkpoint_best.pt")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
