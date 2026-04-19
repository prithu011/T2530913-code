import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, n_classes):
        super().__init__()
        self.conv1 = GATConv(node_features, 64,  heads=4, edge_dim=edge_features, dropout=0.2)
        self.conv2 = GATConv(256,          128,  heads=4, edge_dim=edge_features, dropout=0.2)
        self.conv3 = GATConv(512,          256,  heads=1, edge_dim=edge_features, dropout=0.2)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
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
    """Build binary node-level localization target. -1 fault_loc = skip."""
    targets = torch.zeros(batch.x.size(0), dtype=torch.float, device=batch.x.device)
    node_offset = 0
    for i in range(batch.num_graphs):
        fl = batch.fault_loc[i].item()
        n_nodes = (batch.batch == i).sum().item()
        if fl >= 0:
            targets[node_offset + fl] = 1.0
        node_offset += n_nodes
    return targets


# ── Training ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = train_ds.n_classes

model     = GridGNN(node_features=4, edge_features=3, n_classes=N_CLASSES).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
loc_loss_fn = nn.BCEWithLogitsLoss()

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        class_logits, loc_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        cls_loss = cls_loss_fn(class_logits, batch.y)

        # Loc loss only on samples with a fault location — Issue 7 fix
        has_fault = (batch.fault_loc >= 0)
        if has_fault.any():
            loc_targets = build_loc_targets(batch)
            loc_loss = loc_loss_fn(loc_logits, loc_targets)
        else:
            loc_loss = torch.tensor(0.0, device=DEVICE)

        loss = cls_loss + 0.5 * loc_loss
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

    print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_acc={correct/total:.4f}")

torch.save(model.state_dict(), "gnn_checkpoint.pt")
print("Checkpoint saved.")