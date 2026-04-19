import torch
import numpy as np
import json
import grid2op
from torch_geometric.data import Data, Dataset

# ── Config — derived from environment, not hardcoded ──────────────────────────
ENV_NAME = "l2rpn_neurips_2020_track1_small"
env = grid2op.make(ENV_NAME)

N_SUB  = env.n_sub       # 36
N_LINE = env.n_line      # 59
N_LOAD = env.n_load      # 37
N_GEN  = env.n_gen       # 22

# Build line → bus mapping from env (used for edge_index)
LINE_OR_BUS = env.line_or_to_subid   # (59,) — origin substation per line
LINE_EX_BUS = env.line_ex_to_subid   # (59,) — extremity substation per line

# Bus → load mapping (for aggregating load_p to substation)
LOAD_TO_SUB = env.load_to_subid      # (37,)
GEN_TO_SUB  = env.gen_to_subid       # (22,)

LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3, "maintenance": 4}
RHO_CLIP  = 2.0   # Issue 3 fix

env.close()


def build_node_features(r):
    """
    Per-substation aggregation.
    Features: [sum_load_p, sum_gen_p, mean_v_or, max_rho_connected]
    Shape: (N_SUB, 4)
    """
    load_p = np.array(r["load_p"], dtype=np.float32)    # (37,)
    gen_p  = np.array(r["gen_p"],  dtype=np.float32)    # (22,)
    v_or   = np.array(r["v_or"],   dtype=np.float32)    # (59,)
    rho    = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)  # (59,) — Issue 3

    # Replace NaN/inf (disconnected lines → 0) — Issue 6
    v_or = np.nan_to_num(v_or, nan=0.0, posinf=0.0, neginf=0.0)
    rho  = np.nan_to_num(rho,  nan=0.0, posinf=0.0, neginf=0.0)

    node_load = np.zeros(N_SUB, dtype=np.float32)
    node_gen  = np.zeros(N_SUB, dtype=np.float32)
    node_v    = np.zeros(N_SUB, dtype=np.float32)
    node_rho  = np.zeros(N_SUB, dtype=np.float32)
    v_count   = np.zeros(N_SUB, dtype=np.float32)

    for i, sub in enumerate(LOAD_TO_SUB):
        node_load[sub] += load_p[i]
    for i, sub in enumerate(GEN_TO_SUB):
        node_gen[sub] += gen_p[i]
    for i, (s_or, s_ex) in enumerate(zip(LINE_OR_BUS, LINE_EX_BUS)):
        node_v[s_or] += v_or[i]; v_count[s_or] += 1
        node_v[s_ex] += v_or[i]; v_count[s_ex] += 1
        node_rho[s_or] = max(node_rho[s_or], rho[i])
        node_rho[s_ex] = max(node_rho[s_ex], rho[i])

    # Mean voltage — avoid div-by-zero
    node_v = np.divide(node_v, v_count, out=np.zeros_like(node_v), where=v_count > 0)

    return np.stack([node_load, node_gen, node_v, node_rho], axis=1)  # (36, 4)


def build_edges(r):
    """
    Bidirectional edges from line connectivity.
    edge_index: (2, 118) — 59 lines × 2 directions
    edge_attr:  (118, 3) — [rho, p_or, q_or] per direction
    """
    rho  = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)
    p_or = np.array(r["p_or"], dtype=np.float32)
    q_or = np.array(r["q_or"], dtype=np.float32)

    # NaN fill — Issue 6
    for arr in [rho, p_or, q_or]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    src = np.concatenate([LINE_OR_BUS, LINE_EX_BUS])  # (118,)
    dst = np.concatenate([LINE_EX_BUS, LINE_OR_BUS])  # (118,)
    edge_index = np.stack([src, dst], axis=0)          # (2, 118)

    feats = np.stack([rho, p_or, q_or], axis=1)        # (59, 3)
    edge_attr = np.concatenate([feats, feats], axis=0)  # (118, 3) — same attrs both directions

    return edge_index, edge_attr


class GridDataset(Dataset):
    def __init__(self, records):
        super().__init__()
        self.records = records
        # Derive n_classes from actual data — Issue 2 fix
        self.label_set = sorted(set(r["label"] for r in records))
        self.local_label_map = {l: i for i, l in enumerate(self.label_set)}
        self.n_classes = len(self.label_set)
        print(f"Labels present: {self.label_set} → n_classes={self.n_classes}")

    def len(self):
        return len(self.records)

    def get(self, idx):
        r = self.records[idx]

        node_feats           = build_node_features(r)          # (36, 4)
        edge_index, edge_attr = build_edges(r)                  # (2,118), (118,3)

        label     = self.local_label_map[r["label"]]
        fault_loc = r["fault_loc"]                             # int or None — Issue 7

        return Data(
            x          = torch.tensor(node_feats, dtype=torch.float),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float),
            y          = torch.tensor(label,       dtype=torch.long),
            fault_loc  = torch.tensor(fault_loc if fault_loc is not None else -1,dtype=torch.long),  # -1 = no fault loc
        )