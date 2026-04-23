"""
audit_datasets.py
─────────────────
Data quality audit for generated Grid2Op datasets.
Fixed to l2rpn_neurips_2020_track1_small.

Produces: audit_report.json  (machine-readable)
          audit_report.html  (human-readable, self-contained)

Run:
    python audit_datasets.py --input data/grid_dataset_neurips2020.jsonl
"""

import json
import argparse
import math
import numpy as np
import os
from collections import Counter, defaultdict
from datetime import datetime

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="grid_dataset_neurips2020.jsonl",
                    help="Path to NeurIPS 2020 dataset (.json or .jsonl)")
parser.add_argument("--out",     default="audit_report",
                    help="Output filename stem (no extension)")
args = parser.parse_args()

# ── Expected shapes for NeurIPS 2020 ───────────────────────────────────────────
SPEC = {
    "n_line": 59, "n_load": 37, "n_gen": 22, "n_sub": 36,
    "label": "l2rpn_neurips_2020_track1_small"
}

EXPECTED_LABELS = {"normal", "overload", "line_trip", "cascade", "maintenance"}
RHO_CLIP_THRESHOLD = 2.0
ARRAY_FIELDS = ["rho", "p_or", "q_or", "v_or", "load_p", "gen_p", "line_status"]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE AUDIT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def audit_dataset(records: list) -> dict:
    spec   = SPEC
    issues = []
    stats  = {}
    n      = len(records)

    if n == 0:
        return {"error": "Empty dataset", "issues": [], "stats": {}}

    # ── 1. TOTAL RECORD COUNT ──────────────────────────────────────────────────
    stats["total_records"] = n

    # ── 2. LABEL DISTRIBUTION ─────────────────────────────────────────────────
    labels = [r["label"] for r in records]
    label_counts = dict(Counter(labels))
    stats["label_distribution"] = label_counts
    stats["n_classes"] = len(label_counts)

    missing_labels = EXPECTED_LABELS - set(label_counts.keys())
    if missing_labels:
        issues.append({
            "id":       "MISSING_LABELS",
            "severity": "medium" if missing_labels == {"cascade", "maintenance"} else "high",
            "title":    f"Missing label classes: {sorted(missing_labels)}",
            "detail":   f"Expected 5 classes. Present: {sorted(label_counts.keys())}.",
            "fix":      "Derive n_classes dynamically from set(labels)."
        })

    # Class imbalance ratio
    counts_arr = np.array(list(label_counts.values()))
    imbalance_ratio = float(counts_arr.max() / (counts_arr.min() + 1e-9))
    stats["class_imbalance_ratio"] = round(imbalance_ratio, 2)
    if imbalance_ratio > 10:
        issues.append({
            "id":       "CLASS_IMBALANCE",
            "severity": "high",
            "title":    f"Severe class imbalance (ratio {imbalance_ratio:.1f}×)",
            "detail":   f"Dominant class: {max(label_counts, key=label_counts.get)}. Rarest: {min(label_counts, key=label_counts.get)}.",
            "fix":      "Use weighted CrossEntropyLoss."
        })

    # ── 3. ARRAY SHAPE VALIDATION ─────────────────────────────────────────────
    shape_map = {
        "rho":         spec["n_line"],
        "p_or":        spec["n_line"],
        "q_or":        spec["n_line"],
        "v_or":        spec["n_line"],
        "load_p":      spec["n_load"],
        "gen_p":       spec["n_gen"],
        "line_status": spec["n_line"],
    }
    shape_errors = defaultdict(int)
    for r in records:
        for field, expected_len in shape_map.items():
            if field in r and len(r[field]) != expected_len:
                shape_errors[field] += 1

    stats["shape_errors"] = dict(shape_errors)
    if shape_errors:
        issues.append({
            "id":       "SHAPE_MISMATCH",
            "severity": "high",
            "title":    f"Array shape mismatches in fields: {list(shape_errors.keys())}",
            "detail":   str(dict(shape_errors)),
            "fix":      "Verify data generation script used NeurIPS 2020 environment."
        })

    # ── 4. NaN / Inf DETECTION ────────────────────────────────────────────────
    nan_counts  = defaultdict(int)
    inf_counts  = defaultdict(int)
    for r in records:
        for field in ARRAY_FIELDS:
            if field not in r:
                continue
            arr = r[field]
            for v in arr:
                if isinstance(v, float):
                    if math.isnan(v):
                        nan_counts[field] += 1
                    elif math.isinf(v):
                        inf_counts[field] += 1

    stats["nan_counts"] = dict(nan_counts)
    stats["inf_counts"] = dict(inf_counts)
    total_nan = sum(nan_counts.values())
    total_inf = sum(inf_counts.values())

    if total_nan > 0:
        issues.append({
            "id":       "NAN_VALUES",
            "severity": "high",
            "title":    f"NaN values found ({total_nan} total)",
            "fix":      "Apply np.nan_to_num(arr, nan=0.0) in GridDataset.get()."
        })
    if total_inf > 0:
        issues.append({
            "id":       "INF_VALUES",
            "severity": "high",
            "title":    f"Inf values found ({total_inf} total)",
            "fix":      "Apply np.nan_to_num(arr, posinf=0.0, neginf=0.0) in GridDataset.get()."
        })

    # ── 5. RHO STATISTICS ─────────────────────────────────────────────────────
    all_rho = []
    for r in records:
        for v in r.get("rho", []):
            if isinstance(v, float) and not math.isnan(v) and not math.isinf(v):
                all_rho.append(v)

    if all_rho:
        rho_arr = np.array(all_rho)
        rho_stats = {
            "min":    round(float(rho_arr.min()), 4),
            "max":    round(float(rho_arr.max()), 4),
            "mean":   round(float(rho_arr.mean()), 4),
            "p95":    round(float(np.percentile(rho_arr, 95)), 4),
            "p99":    round(float(np.percentile(rho_arr, 99)), 4),
        }
        stats["rho"] = rho_stats

    # ── 6. EPISODE (CHRONIC) LENGTH ANALYSIS ──────────────────────────────────
    chronic_steps = defaultdict(int)
    for r in records:
        cid = r.get("chronic_id", r.get("chronic"))
        chronic_steps[cid] += 1

    step_counts    = list(chronic_steps.values())
    stats["episode_lengths"] = {
        "n_chronics":     len(chronic_steps),
        "min_steps":      int(min(step_counts)) if step_counts else 0,
        "max_steps":      int(max(step_counts)) if step_counts else 0,
    }

    return {"issues": issues, "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════

records = []
try:
    print(f"Loading {args.input} ...", flush=True)
    if args.input.endswith(".jsonl"):
        with open(args.input) as f:
            records = [json.loads(line) for line in f]
    else:
        with open(args.input) as f:
            records = json.load(f)
    print(f"  → {len(records)} records")
except Exception as e:
    print(f"  ✗ Error loading {args.input}: {e}")
    records = None

# ── Run audit ────────────────────────────────────────────────────────────────
if records is not None:
    print(f"\nAuditing NeurIPS 2020 Dataset...")
    result = audit_dataset(records)
    n_issues = len(result["issues"])
    print(f"  → {n_issues} issue(s) found")
else:
    result = {"error": "File not found", "issues": [], "stats": {}}

# ── Save JSON report ──────────────────────────────────────────────────────────
report_data = {
    "generated_at": datetime.now().isoformat(),
    "dataset": {
        "path": args.input,
        **result
    }
}

json_path = f"{args.out}.json"
with open(json_path, "w") as f:
    json.dump(report_data, f, indent=2)
print(f"\nJSON report → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_CONFIG = {
    "high":           {"color": "#e05252", "bg": "#fdf2f2", "label": "HIGH"},
    "medium":         {"color": "#d4841a", "bg": "#fdf6ec", "label": "MEDIUM"},
    "informational":  {"color": "#3b82c4", "bg": "#eff6ff", "label": "INFO"},
}

def severity_badge(sev):
    cfg = SEVERITY_CONFIG.get(sev, SEVERITY_CONFIG["medium"])
    return (f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
            f'border:1px solid {cfg["color"]}33;padding:2px 8px;border-radius:4px;'
            f'font-size:11px;font-weight:600;">'
            f'{cfg["label"]}</span>')

def stat_card(label, value):
    return f"""
    <div style="background:#f8f8f7;border:1px solid #e8e6e0;border-radius:8px;padding:14px 18px;min-width:120px">
      <div style="font-size:11px;color:#888;margin-bottom:4px;">{label}</div>
      <div style="font-size:22px;font-weight:600;color:#1a1a18;line-height:1">{value}</div>
    </div>"""

def render_report(result):
    if "error" in result and result["error"]:
        return f"<div>Error: {result['error']}</div>"

    stats = result["stats"]
    issues = result["issues"]

    cards_html = "".join([
        stat_card("TOTAL RECORDS",   f"{stats.get('total_records',0):,}"),
        stat_card("CLASSES",         stats.get("n_classes", "—")),
        stat_card("RHO MAX",         stats.get("rho", {}).get("max", "—")),
        stat_card("CHRONICS",        stats.get("episode_lengths", {}).get("n_chronics","—")),
    ])

    issues_html = ""
    for issue in issues:
        issues_html += f"""
        <div style="border:1px solid #e8e6e0;border-radius:8px;padding:16px;margin-bottom:10px">
          <div style="display:flex;gap:10px;">
            {severity_badge(issue['severity'])}
            <strong>{issue['id']}</strong>: {issue['title']}
          </div>
          <div style="font-size:12px;margin-top:5px">{issue.get('detail', '')}</div>
          <div style="font-size:12px;color:#3b82c4;margin-top:5px">Fix: {issue['fix']}</div>
        </div>"""

    return f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px">{cards_html}</div>
    <h3>Issues ({len(issues)})</h3>
    <div>{issues_html if issues else 'No issues found.'}</div>
    """

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
html_content = render_report(result)

html = f"""<!DOCTYPE html>
<html>
<head><title>Audit Report</title></head>
<body style="font-family:sans-serif; padding:40px; max-width:800px; margin:auto;">
  <h1>Dataset Audit: NeurIPS 2020</h1>
  <p>Generated: {ts}</p>
  {html_content}
</body>
</html>"""

html_path = f"{args.out}.html"
with open(html_path, "w") as f:
    f.write(html)
print(f"HTML report  → {html_path}")
