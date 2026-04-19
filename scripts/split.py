from sklearn.model_selection import train_test_split
import numpy as np

with open("grid_dataset_neurips_small.json") as f:
    records = json.load(f)

labels = [r["label"] for r in records]

# Stratified split by timestep — Issue 1 fix
idx = np.arange(len(records))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)

train_records = [records[i] for i in train_idx]
val_records   = [records[i] for i in val_idx]
test_records  = [records[i] for i in test_idx]

train_ds = GridDataset(train_records)
val_ds   = GridDataset(val_records)
test_ds  = GridDataset(test_records)


def compute_class_weights(dataset):
    counts = np.zeros(dataset.n_classes)
    for r in dataset.records:
        counts[dataset.local_label_map[r["label"]]] += 1
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(weights / weights.sum(), dtype=torch.float)

class_weights = compute_class_weights(train_ds)
print("Class weights:", class_weights)