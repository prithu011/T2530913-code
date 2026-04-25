import os
import json
import sys
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import your newly vectorized functions
from pyg_data import GridEnvMetadata, build_node_features, build_edges, LABEL_MAP
from training.config import DATA_FILE, DATA_DIR

def preprocess_data():
    print(f"Loading metadata...")
    meta = GridEnvMetadata()
    
    out_file = os.path.join(DATA_DIR, "processed_grid_data.pt")
    
    data_list = []
    print(f"Processing {DATA_FILE}...")
    
    with open(DATA_FILE, 'r') as f:
        for line in tqdm(f):
            r = json.loads(line)
            
            node_feats = build_node_features(r, meta)
            edge_index, edge_attr = build_edges(r, meta)
            
            label = LABEL_MAP.get(r["label"], 0)
            fault_loc = r["fault_loc"]

            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor(label, dtype=torch.long),
                fault_loc=torch.tensor(fault_loc if fault_loc is not None else -1, dtype=torch.long),
            )
            data_list.append(data)

    print(f"Collating {len(data_list)} graphs to save memory...")
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_file)
    print("Done!")

if __name__ == "__main__":
    preprocess_data()