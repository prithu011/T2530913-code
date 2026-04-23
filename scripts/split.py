import json
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split

def load_labels(file_path):
    """Load only labels to save memory for splitting."""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            labels.append(record["label"])
    return labels

def get_splits(file_path, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42):
    labels = load_labels(file_path)
    idx = np.arange(len(labels))
    
    # First split: train vs temp (val + test)
    train_idx, temp_idx = train_test_split(
        idx, 
        test_size=(val_size + test_size), 
        stratify=labels, 
        random_state=random_seed
    )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_idx]
    val_rel_size = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=(1.0 - val_rel_size), 
        stratify=temp_labels, 
        random_state=random_seed
    )
    
    return train_idx, val_idx, test_idx

def compute_class_weights(labels, label_map):
    counts = np.zeros(len(label_map))
    for l in labels:
        if l in label_map:
            counts[label_map[l]] += 1
    weights = 1.0 / (counts + 1e-6)
    return weights / weights.sum()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/grid_dataset_neurips2020.jsonl', help='Path to .jsonl dataset')
    parser.add_argument('--output_prefix', type=str, default='split_neurips2020', help='Prefix for output files')
    args = parser.parse_args()

    import os
    if not os.path.exists(args.input):
        print(f"Error: Dataset {args.input} not found.")
        sys.exit(1)

    print(f"Loading labels from {args.input}...")
    train_idx, val_idx, test_idx = get_splits(args.input)
    
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Save indices in the data directory
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, f"{args.output_prefix}_train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, f"{args.output_prefix}_val_idx.npy"), val_idx)
    np.save(os.path.join(out_dir, f"{args.output_prefix}_test_idx.npy"), test_idx)
    print(f"Split indices saved in {out_dir}/: {args.output_prefix}_*.npy")
