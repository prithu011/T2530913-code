import torch
import numpy as np
import json
import grid2op
from torch_geometric.data import Data, Dataset
import os
import linecache

def get_env_config(env_name):
    """Get environment configuration based on name."""
    env_configs = {
        "neurips": {
            "name": "l2rpn_neurips_2020_track1_small",
            "tag": "neurips2020",
            "desc": "NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]",
            "n_classes": 5
        },
        "wcci": {
            "name": "l2rpn_wcci_2022",
            "tag": "wcci2022",
            "desc": "WCCI 2022 L2RPN — 118 subs, 186 lines [STRETCH]",
            "n_classes": 5
        }
    }

    if env_name not in env_configs:
        raise ValueError(f"Unknown environment: {env_name}. Choose from {list(env_configs.keys())}")

    return env_configs[env_name]

class GridEnvMetadata:
    def __init__(self, env_name):
        cfg = get_env_config(env_name)
        self.env_name = cfg["name"]
        
        # Grid2Op takes time to initialize; we'll cache the results
        env = grid2op.make(self.env_name)
        self.n_sub = env.n_sub
        self.n_line = env.n_line
        self.n_load = env.n_load
        self.n_gen = env.n_gen
        
        self.line_or_bus = env.line_or_to_subid.copy()
        self.line_ex_bus = env.line_ex_to_subid.copy()
        self.load_to_sub = env.load_to_subid.copy()
        self.gen_to_sub = env.gen_to_subid.copy()
        
        env.close()

LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3, "maintenance": 4}
RHO_CLIP  = 2.0

def build_node_features(r, meta: GridEnvMetadata):
    load_p = np.array(r["load_p"], dtype=np.float32)
    gen_p  = np.array(r["gen_p"],  dtype=np.float32)
    v_or   = np.array(r["v_or"],   dtype=np.float32)
    rho    = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)

    v_or = np.nan_to_num(v_or, nan=0.0, posinf=0.0, neginf=0.0)
    rho  = np.nan_to_num(rho,  nan=0.0, posinf=0.0, neginf=0.0)

    node_load = np.zeros(meta.n_sub, dtype=np.float32)
    node_gen  = np.zeros(meta.n_sub, dtype=np.float32)
    node_v    = np.zeros(meta.n_sub, dtype=np.float32)
    node_rho  = np.zeros(meta.n_sub, dtype=np.float32)
    v_count   = np.zeros(meta.n_sub, dtype=np.float32)

    for i, sub in enumerate(meta.load_to_sub):
        node_load[sub] += load_p[i]
    for i, sub in enumerate(meta.gen_to_sub):
        node_gen[sub] += gen_p[i]
    for i, (s_or, s_ex) in enumerate(zip(meta.line_or_bus, meta.line_ex_bus)):
        node_v[s_or] += v_or[i]; v_count[s_or] += 1
        node_v[s_ex] += v_or[i]; v_count[s_ex] += 1
        node_rho[s_or] = max(node_rho[s_or], rho[i])
        node_rho[s_ex] = max(node_rho[s_ex], rho[i])

    node_v = np.divide(node_v, v_count, out=np.zeros_like(node_v), where=v_count > 0)

    return np.stack([node_load, node_gen, node_v, node_rho], axis=1)

def build_edges(r, meta: GridEnvMetadata):
    rho  = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)
    p_or = np.array(r["p_or"], dtype=np.float32)
    q_or = np.array(r["q_or"], dtype=np.float32)

    for arr in [rho, p_or, q_or]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    src = np.concatenate([meta.line_or_bus, meta.line_ex_bus])
    dst = np.concatenate([meta.line_ex_bus, meta.line_or_bus])
    edge_index = np.stack([src, dst], axis=0)

    feats = np.stack([rho, p_or, q_or], axis=1)
    edge_attr = np.concatenate([feats, feats], axis=0)

    return edge_index, edge_attr

class GridDataset(Dataset):
    """
    Lazy-loading dataset that reads from a .jsonl file on demand.
    indices: list of 0-based indices in the .jsonl file.
    file_path: path to the .jsonl file.
    meta: GridEnvMetadata instance.
    """
    def __init__(self, file_path, indices, meta: GridEnvMetadata):
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.indices = indices
        self.meta = meta
        self.n_classes = len(LABEL_MAP)

    def len(self):
        return len(self.indices)

    def get(self, idx):
        # linecache line numbers are 1-based
        line_num = self.indices[idx] + 1
        line = linecache.getline(self.file_path, line_num)
        if not line:
            raise IndexError(f"Line {line_num} not found in {self.file_path}")
        
        r = json.loads(line)
        node_feats            = build_node_features(r, self.meta)
        edge_index, edge_attr = build_edges(r, self.meta)

        label     = LABEL_MAP.get(r["label"], 0)
        fault_loc = r["fault_loc"]

        return Data(
            x          = torch.tensor(node_feats, dtype=torch.float),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float),
            y          = torch.tensor(label,       dtype=torch.long),
            fault_loc  = torch.tensor(fault_loc if fault_loc is not None else -1, dtype=torch.long),
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='neurips', choices=['neurips', 'wcci'])
    args = parser.parse_args()
    
    meta = GridEnvMetadata(args.env)
    print(f"Metadata loaded for {args.env}: n_sub={meta.n_sub}, n_line={meta.n_line}")
