"""
Grid2Op Dual-Environment Data Generation Pipeline
==================================================
Supports both thesis environments from a single script.
Run via CLI:

    python generate_dataset.py --env neurips        # 36 subs, 59 lines  [PRIMARY]
    python generate_dataset.py --env wcci           # 118 subs, 186 lines [STRETCH]
    python generate_dataset.py --env neurips --smoke  # 3 chronics, 200 steps — quick sanity check
    python generate_dataset.py --env wcci   --smoke

Outputs (per run):
    grid_dataset_<env_tag>.jsonl      — one JSON record per line (streamable)
    grid_dataset_<env_tag>_meta.json  — env dims + label distribution (for GNN constructor)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
import numpy as np
from collections import Counter
from pathlib import Path

import grid2op
from grid2op.Parameters import Parameters
from tqdm.auto import tqdm

# ── ENV REGISTRY ──────────────────────────────────────────────────────────────
# Add entries here if you want to support more environments later.
ENV_REGISTRY = {
    "neurips": {
        "name":    "l2rpn_neurips_2020_track1_small",
        "tag":     "neurips2020",
        "desc":    "NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]",
    },
    "wcci": {
        "name":    "l2rpn_wcci_2022",
        "tag":     "wcci2022",
        "desc":    "WCCI 2022 L2RPN — 118 subs, 186 lines [STRETCH]",
    },
    "toy": {
        "name":    "rte_case5_example",
        "tag":     "toy",
        "desc":    "Toy 5-bus env — integration testing only",
    },
}

# ── FIXED CONFIG ──────────────────────────────────────────────────────────────
FAULT_PROB  = 0.08   # line-trip injection probability per step
SEED        = 42
RHO_CLIP    = 2.0    # clip rho before saving; must match GridDataset normalisation

# Smoke-test overrides (--smoke flag)
SMOKE_MAX_CHRONICS = 3
SMOKE_MAX_STEPS    = 200

LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3, "maintenance": 4}

LINE_KEYS = ("rho", "p_or", "q_or", "p_ex", "q_ex", "v_or", "v_ex")
BUS_KEYS  = ("load_p", "load_q", "gen_p", "gen_q", "topo_vect")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Grid2Op dataset generator")
    parser.add_argument(
        "--env", choices=list(ENV_REGISTRY.keys()), required=True,
        help="Environment to run: neurips | wcci | toy"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Quick sanity check: {SMOKE_MAX_CHRONICS} chronics × {SMOKE_MAX_STEPS} steps"
    )
    parser.add_argument(
        "--max-chronics", type=int, default=None,
        help="Override: max number of chronics to run (default: all)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override: max steps per episode (default: full episode)"
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="Output directory (default: current directory)"
    )
    return parser.parse_args()


def load_backend():
    try:
        from lightsim2grid import LightSimBackend
        b = LightSimBackend()
        print("[backend] LightSimBackend loaded (~10x faster)")
        return b
    except Exception:
        print("[backend] LightSimBackend unavailable — falling back to PandaPower (slow)")
        return None


def safe_tolist(arr, fill=0.0):
    """Convert array to list, replacing NaN/inf with fill value."""
    arr = np.array(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr.tolist()


def extract_features(obs):
    feats = {}
    for key in LINE_KEYS:
        arr = safe_tolist(getattr(obs, key))
        if key == "rho":
            arr = [min(v, RHO_CLIP) for v in arr]
        feats[key] = arr
    feats["line_status"] = obs.line_status.tolist()
    for key in BUS_KEYS:
        feats[key] = safe_tolist(getattr(obs, key))
    return feats


def derive_label(obs, prev_line_status, injected_label, injected_loc):
    # Priority: maintenance > overload > cascade > injected
    if hasattr(obs, "time_next_maintenance") and hasattr(obs, "duration_next_maintenance"):
        under_maint = (obs.time_next_maintenance == 0) & (obs.duration_next_maintenance > 0)
        if under_maint.any():
            return "maintenance", int(np.where(under_maint)[0][0])

    if obs.rho.max() > 1.0:
        return "overload", int(obs.rho.argmax())

    new_trips = (~obs.line_status) & prev_line_status
    if new_trips.any() and injected_label == "normal":
        return "cascade", int(np.where(new_trips)[0][0])

    return injected_label, injected_loc


def validate_record(record):
    """Raise immediately if any numeric field contains NaN/inf."""
    for key in LINE_KEYS + BUS_KEYS:
        for v in record.get(key, []):
            if not np.isfinite(v):
                raise ValueError(f"Non-finite value in '{key}': {v}")


def build_meta(env, env_key, label_counts, total_records, total_time, smoke):
    """Serialisable metadata dict — consumed by GridDataset and GNN constructor."""
    present_labels = {k: v for k, v in label_counts.items() if v > 0}
    return {
        "env_key":       env_key,
        "env_name":      ENV_REGISTRY[env_key]["name"],
        "smoke_run":     smoke,
        "n_sub":         int(env.n_sub),
        "n_line":        int(env.n_line),
        "n_load":        int(env.n_load),
        "n_gen":         int(env.n_gen),
        "n_classes":     len(present_labels),           # dynamic — use this in GNN
        "label_map":     {k: LABEL_MAP[k] for k in present_labels},
        "rho_clip":      RHO_CLIP,
        "total_records": total_records,
        "label_counts":  dict(label_counts),
        "label_pct":     {k: round(100 * v / max(total_records, 1), 2)
                          for k, v in label_counts.items()},
        "throughput_steps_per_sec": round(total_records / max(total_time, 1e-9)),
        "total_time_sec": round(total_time, 1),
        # Shapes for GridDataset / GNN constructor — no magic numbers needed
        "node_feature_dim": 4,   # load_p, gen_p, mean_v_or, max_rho — built in GridDataset
        "edge_feature_dim": 3,   # rho, p_or, q_or — per line, bidirectional
    }


def print_summary(meta, out_jsonl, out_meta):
    total   = meta["total_records"]
    print(f"\n{'='*56}")
    print(f"  Env      : {meta['env_name']}")
    print(f"  Records  : {total:,}")
    print(f"  Time     : {meta['total_time_sec']:.1f}s  "
          f"({meta['throughput_steps_per_sec']} steps/sec)")
    print(f"  n_classes: {meta['n_classes']}  →  {list(meta['label_map'].keys())}")
    print(f"\n  Label distribution:")
    for lbl, count in sorted(meta["label_counts"].items(), key=lambda x: -x[1]):
        pct = meta["label_pct"][lbl]
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<12} {count:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Data   → {out_jsonl}")
    print(f"  Meta   → {out_meta}")
    print(f"{'='*56}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    env_cfg = ENV_REGISTRY[args.env]
    smoke   = args.smoke

    max_chronics = SMOKE_MAX_CHRONICS if smoke else args.max_chronics
    max_steps    = SMOKE_MAX_STEPS    if smoke else args.max_steps

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag      = env_cfg["tag"] + ("_smoke" if smoke else "")
    out_jsonl = out_dir / f"grid_dataset_{tag}.jsonl"
    out_meta  = out_dir / f"grid_dataset_{tag}_meta.json"

    print(f"\n[run]  {env_cfg['desc']}")
    if smoke:
        print(f"[run]  SMOKE MODE — {max_chronics} chronics × {max_steps} steps\n")

    backend = load_backend()
    make_kwargs = {"backend": backend} if backend else {}
    env = grid2op.make(env_cfg["name"], **make_kwargs)

    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    env.change_parameters(params)
    env.seed(SEED)
    np.random.seed(SEED)

    n_chronics = len(env.chronics_handler.subpaths)
    if max_chronics:
        n_chronics = min(max_chronics, n_chronics)

    print(f"[env]  n_sub={env.n_sub}  n_line={env.n_line}  "
          f"n_load={env.n_load}  n_gen={env.n_gen}")
    print(f"       chronics: {len(env.chronics_handler.subpaths)} available "
          f"→ {n_chronics} running\n")

    do_nothing   = env.action_space({})
    label_counts = Counter({k: 0 for k in LABEL_MAP})
    total_written = 0
    t_start = time.time()

    with out_jsonl.open("w") as out_f:
        for chronic_id in tqdm(range(n_chronics), desc="Chronics", unit="chronic"):
            obs   = env.reset()
            steps = min(env.max_episode_duration(),
                        max_steps if max_steps else int(1e9))

            prev_line_status = obs.line_status.copy()

            for t in tqdm(range(steps), desc=f"  Chronic {chronic_id+1}", leave=False, unit="step"):
                action      = do_nothing
                fault_label = "normal"
                fault_loc   = None

                if np.random.rand() < FAULT_PROB:
                    connected = np.where(obs.line_status)[0]
                    if len(connected) > 0:
                        line_id     = int(np.random.choice(connected))
                        action      = env.action_space({"set_line_status": [(line_id, -1)]})
                        fault_label = "line_trip"
                        fault_loc   = line_id

                obs, reward, done, _info = env.step(action)
                fault_label, fault_loc   = derive_label(
                    obs, prev_line_status, fault_label, fault_loc
                )
                prev_line_status = obs.line_status.copy()

                record = {
                    **extract_features(obs),
                    "label":      fault_label,
                    "label_int":  LABEL_MAP[fault_label],
                    "fault_loc":  fault_loc,
                    "timestep":   t,
                    "chronic_id": chronic_id,
                    "reward":     float(reward),
                }

                validate_record(record)
                out_f.write(json.dumps(record) + "\n")
                label_counts[fault_label] += 1
                total_written += 1

                if done:
                    break

    total_time = time.time() - t_start
    meta = build_meta(env, args.env, label_counts, total_written, total_time, smoke)

    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print_summary(meta, out_jsonl, out_meta)


if __name__ == "__main__":
    main()