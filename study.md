# Methodology & Implementation Guide

> Everything here is decided. This is not a discussion document — it is a reference for how we are building the system, why we made each choice, and how to collect the data we need.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Component A — Graph Neural Network (Neural Layer)](#2-component-a--graph-neural-network-neural-layer)
3. [Component B — LLM Knowledge Extraction Pipeline](#3-component-b--llm-knowledge-extraction-pipeline)
4. [Component C — Knowledge Graph (Symbolic Rule Store)](#4-component-c--knowledge-graph-symbolic-rule-store)
5. [Component D — Symbolic Validation Shield](#5-component-d--symbolic-validation-shield)
6. [Data Collection Strategy](#6-data-collection-strategy)
7. [GNN Training Pipeline](#7-gnn-training-pipeline)
8. [End-to-End Toy Scenario Testing](#8-end-to-end-toy-scenario-testing)
9. [Technology Stack](#9-technology-stack)
10. [Hardware & Compute Allocation](#10-hardware--compute-allocation)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Out of Scope](#12-out-of-scope)
13. [Glossary](#13-glossary)
14. [Disclaimer — Living Document](#14-disclaimer--living-document)

---

## 1. System Architecture

### The Full Pipeline

```
[Domain Documents]                        [Grid2Op Simulation]
(IEEE standards, grid                     (l2rpn_neurips_2020_track1:
 manuals, textbooks)                       36 substations, 59 lines,
                                           48 years of chronic data)
        │                                           │
        ▼                                           ▼
[LLM Knowledge Extraction]              [Dataset Generation]
 Qwen2.5-32B parses documents →          Observations logged per step:
 outputs structured JSON rules           rho, voltages, flows, topology,
 → loaded into Knowledge Graph           fault labels
        │                                           │
        ▼                                           ▼
[Knowledge Graph]                       [GNN Training]
 Rules + grid topology stored            PyTorch Geometric model
 as queryable graph                      learns fault detection &
 (NetworkX / Neo4j)                      action recommendation
        │                                           │
        └──────────────────┬────────────────────────┘
                           │
                           ▼
              [Symbolic Validation — THE SHIELD]
              Every GNN prediction is checked against KG rules.
              PASS → output forwarded
              FAIL → prediction blocked + explanation generated
                           │
                           ▼
                    [Final Output]
              Validated recommendation + traceable explanation
              (cites rule ID and source document)
```

### The Non-Negotiable Principle

**GNN outputs are never final without symbolic validation.** The GNN is the perception engine. The shield is the safety officer. They are always both active. There is no mode where the GNN output bypasses the shield.

---

## 2. Component A — Graph Neural Network (Neural Layer)

### What It Does

Takes a snapshot of the grid state as a graph and outputs one of:
- Fault classification (normal / overload / undervoltage / line trip)
- Fault location (which bus or line)
- Recommended action (load shed, line switch, generator redispatch)

### Why GNN and Not Something Else

A power grid **is** a graph. Buses are nodes. Lines are edges. Electrical quantities (voltage, current, power) are node and edge features. Every other architecture ignores this:

- **MLP/DNN:** Flattens the grid into a vector. Loses all topology information. A fault on Line 3-4 looks identical whether Bus 3 has 2 neighbors or 10.
- **LSTM:** Models time sequences well but is completely blind to graph structure. Cannot localize a fault spatially.
- **CNN:** Designed for grid-like spatial data (images). Power grids are irregular, sparse graphs — CNN kernels have no meaning here.

A GNN propagates information along edges, so it natively understands that a fault on Line 3-4 affects Bus 3 and Bus 4 differently depending on their local topology. This gives us accurate fault localization, topology-aware predictions, and generalization when lines are switched.

### Architecture Decision

We use **Graph Convolutional Network (GCN)** as the baseline, with **Graph Attention Network (GAT)** as the comparison. GAT is preferred if the attention weights improve interpretability (each edge gets an importance score, which is useful for explainability). Final architecture decided after initial experiments.

### Library

**PyTorch Geometric (PyG)** — the standard library for GNN research. Direct integration with PyTorch, supports GCN, GAT, GraphSAGE, and all standard message-passing architectures out of the box.

### Input/Output Specification

**Node features (per bus):**
- Voltage magnitude (pu)
- Active load (kW)
- Reactive load (kVAR)
- Active generation (kW)
- Bus type (slack / PQ / PV — one-hot)

**Edge features (per line):**
- Active power flow (kW)
- Line loading percentage (%)
- Line resistance and reactance (pu)

**Output:**
- Node-level: fault probability per bus
- Graph-level: fault type classification
- Action head: recommended control action (multi-class)

---

## 3. Component B — LLM Knowledge Extraction Pipeline

### What It Does

An LLM reads domain-authoritative documents — IEEE standards, grid operation manuals, safety protocols — and extracts symbolic rules in structured JSON format. These rules are then loaded into the knowledge graph.

This replaces manual rule encoding, which is the core knowledge bottleneck in all existing Neuro-Symbolic systems for power grids.

### Why LLM for This

The alternative is having a domain expert read every document and manually translate prose into code. That is:
- Slow (weeks per document set)
- Error-prone (rules buried in dense technical prose are easy to miss)
- Not scalable (adding a new standard means starting over)

The closest validated precedent is **Chen et al. (2025, 2026)**, who used an LLM to extract rules from USGS geology textbooks into a Knowledge Graph, then used those rules to constrain a Random Forest — achieving 99.06% accuracy vs an 84.3% baseline. We apply the same pipeline to power grid documents.

### Why Local LLM (Not OpenAI API)

- Grid operational documents may be institution-sensitive
- API calls introduce non-determinism across runs (model updates, rate limits)
- Local inference is fully reproducible
- The Research PC (RTX 4080 Super, 16GB VRAM + 64GB RAM) handles Qwen2.5-32B at 4-bit quantization comfortably

### Model

**Qwen2.5-72B (4-bit quantized)** via `llama-cpp-python`, run on the Research PC. This is the Extractor stage of the multi-LLM pipeline. At 4-bit quantization a 72B model requires ~40–44GB — this exceeds the 16GB VRAM of the RTX 4080 Super and relies heavily on CPU offloading into the 64GB system RAM. Inference will be slower than a fully GPU-resident model (~1–3 tokens/sec on offloaded layers), which is acceptable for a one-time offline extraction run.

**Why Qwen2.5-72B over the previously noted 32B:**
- The extraction task requires the highest possible recall on dense IEEE standards prose — 72B models materially outperform 32B on complex technical comprehension and constraint boundary detection
- The Research PC's 64GB RAM absorbs the majority of layers that spill from VRAM; no model sharding or second machine required
- Extraction is a one-time offline run — throughput is not a constraint

**Fallback:** Qwen2.5-32B (4-bit) if 72B inference proves unacceptably slow — it fits within ~18–20GB with only ~4GB RAM offload and runs at ~5–8 tokens/sec.

### Multi-LLM Hybrid Pipeline

A single-model extraction pipeline places competing demands on one model simultaneously: reading comprehension over dense technical prose, strict schema compliance, and precise constraint boundary detection. A two-LLM architecture separates these concerns across models optimized for each role.

Four patterns were evaluated (Extractor+Validator, Dual Extraction+Consensus, Extractor+Formalizer, Self-Consistency). The chosen approach combines Pattern 1 and Pattern 2.

#### Chosen Architecture: Extractor + Validator with Consensus Flagging

```
[Document Chunk]
        │
        ▼
[Qwen2.5-72B — Extractor]       ← high recall; extract everything that might be a rule
        │ JSON rule candidates
        ▼
[Llama 3.3-70B — Validator]     ← high precision; verify each rule against the source chunk
  For each rule:
    - Is this constraint actually present in the text?
    - Is the condition boundary correctly parsed?
    - Is the entity correctly identified?
  Output: CONFIRM / REJECT / CORRECT per rule
        │
        ▼
[Conflict cases] ──► flagged for manual review
        │
        ▼
[Pydantic + GBNF schema enforcement]
        │
        ▼
[Knowledge Graph]
```

**Why this split:**
- Qwen2.5-72B for extraction: best technical comprehension and recall on dense IEEE prose — optimizes for not missing rules
- Llama 3.3-70B for validation: different developer, different pretraining corpus, different tokenization — independent failure modes. Agreement between architecturally distinct models is stronger evidence of a correct extraction than two Qwen models agreeing
- Validation is a strictly easier task than extraction; the 70B validator receives the original chunk plus the candidate rules, so it only needs to verify, not discover
- The validation step produces a reportable precision metric for the thesis: % of extracted rules confirmed by the validator

**Why not run both simultaneously:**
Sequential execution on the Research PC avoids simultaneous multi-model VRAM pressure. Qwen2.5-72B at 4-bit quantization runs first across the full document set; results are saved. Llama 3.3-70B is then loaded for the validation pass. No concurrent loading required.

**Confidence tiers produced:**

| Outcome | Meaning | Action |
|---|---|---|
| Validator: CONFIRM | Rule verified in source text | Load into KG |
| Validator: CORRECT | Rule present but condition/entity adjusted | Load corrected version |
| Validator: REJECT | Rule not supported by source text | Discard |
| Validator: CONFLICT (score 0.5–0.85) | Ambiguous — partial match | Flag for manual review |

#### Implementation Skeleton

```python
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, ValidationError
import json

EXTRACTOR_MODEL = "qwen2.5-72b-q4_k_m.gguf"
VALIDATOR_MODEL  = "llama-3.3-70b-q4_k_m.gguf"

EXTRACT_PROMPT = """You are a power systems engineer extracting safety rules.
From the text below, extract ALL operational constraints as a JSON array.
Each rule must have: rule_id, source, entity, condition, action, severity, explanation.
If no rule is present, return an empty list. Output ONLY a JSON array.

Text:
{chunk}"""

VALIDATE_PROMPT = """You are a power systems safety auditor.
Below is a source text and a list of rules extracted from it.
For each rule, output: rule_id, verdict (CONFIRM/REJECT/CORRECT), and if CORRECT provide the corrected fields.
Output ONLY a JSON array.

Source text:
{chunk}

Extracted rules:
{rules}"""

def run_extraction(chunk: str) -> list:
    llm = Llama(model_path=EXTRACTOR_MODEL, n_gpu_layers=-1, verbose=False)
    grammar = LlamaGrammar.from_file("array.gbnf")  # enforces JSON array output
    out = llm(EXTRACT_PROMPT.format(chunk=chunk), grammar=grammar, max_tokens=2048)
    del llm  # release VRAM before loading validator
    return json.loads(out["choices"][0]["text"])

def run_validation(chunk: str, candidates: list) -> list:
    llm = Llama(model_path=VALIDATOR_MODEL, n_gpu_layers=-1, verbose=False)
    grammar = LlamaGrammar.from_file("array.gbnf")
    out = llm(
        VALIDATE_PROMPT.format(chunk=chunk, rules=json.dumps(candidates, indent=2)),
        grammar=grammar, max_tokens=2048
    )
    del llm
    return json.loads(out["choices"][0]["text"])

def process_chunk(chunk: str) -> dict:
    candidates  = run_extraction(chunk)
    verdicts    = run_validation(chunk, candidates)

    verdict_map = {v["rule_id"]: v for v in verdicts}
    confirmed, flagged = [], []

    for rule in candidates:
        v = verdict_map.get(rule["rule_id"], {})
        if v.get("verdict") == "CONFIRM":
            confirmed.append(rule)
        elif v.get("verdict") == "CORRECT":
            confirmed.append({**rule, **v.get("corrected_fields", {})})
        elif v.get("verdict") == "REJECT":
            pass  # discard
        else:
            flagged.append({"rule": rule, "verdict": v})  # ambiguous — manual review

    return {"confirmed": confirmed, "flagged": flagged}
```

### Extraction Pipeline (Step by Step)

**Step 1 — Document ingestion**
LangChain's document loaders parse PDFs/text into chunks. Each chunk is a few paragraphs.

**Step 2 — Prompted extraction**
Each chunk is passed to the LLM with a structured prompt:

```
You are a power systems engineer extracting safety rules from grid documentation.
From the text below, extract ALL operational constraints as JSON objects.
Each rule must have: rule_id, source, entity, condition, action, severity, explanation.
If no rule is present, return an empty list.
Respond ONLY with a JSON array. No preamble.

Text:
{chunk}
```

**Step 3 — Parsing and validation**
JSON output is parsed and validated against a Pydantic schema. Malformed outputs are retried once, then logged as failures for manual review.

**Step 4 — Deduplication**
Rules with identical conditions across overlapping chunks are merged. Source references are preserved.

### Output Schema

```json
{
  "rule_id": "R_042",
  "source": "IEEE Std 1547-2018, Section 7.4",
  "entity": "Bus",
  "condition": "voltage_pu > 1.05 OR voltage_pu < 0.95",
  "action": "BLOCK",
  "severity": "critical",
  "explanation": "Voltage deviation beyond ±5% nominal violates IEEE 1547 interconnection standards."
}
```

### Source Documents (Planned)

- IEEE Std 1547-2018 (Interconnection and Interoperability Standards)
- IEEE Std C37.2 (Standard for Electrical Power System Device Function Numbers)
- Pandapower documentation (line ratings, transformer limits)
- Any grid operation manual available in the public domain

---

## 4. Component C — Knowledge Graph (Symbolic Rule Store)

### What It Does

Stores the extracted rules and the grid topology as a queryable graph. During inference, the shield queries this graph to evaluate a GNN prediction.

### Graph Schema

**Nodes (Entities):**

| Node Type | Attributes |
|---|---|
| `Bus` | bus_id, voltage_nominal, bus_type |
| `Line` | line_id, max_current_A, max_loading_pct |
| `Transformer` | trafo_id, max_loading_pct, tap_ratio |
| `Generator` | gen_id, p_min_kw, p_max_kw |
| `Load` | load_id, p_nominal_kw |
| `ProtectionDevice` | device_id, trip_threshold |
| `Rule` | rule_id, condition, action, severity, source |

**Edges (Relations):**

| Edge | Meaning |
|---|---|
| `connected_to` | Bus ↔ Line, Bus ↔ Transformer |
| `feeds` | Generator → Bus, Line → Bus |
| `protected_by` | Line/Bus → ProtectionDevice |
| `has_rule` | Entity → Rule |
| `triggers` | Rule → ProtectionDevice |

### Why a Graph and Not a Rule Database

Rules in power grids are **relational**. The rule "if Line 3-4 is overloaded, trip the breaker protecting Bus 3" requires knowing that Line 3-4 connects to Bus 3, which is protected by Breaker B-12. A flat table or a simple list cannot traverse these relationships. A graph can resolve the full chain in a single traversal.

### Why NetworkX (Not Neo4j) for Now

Neo4j is production-grade but requires a running server process and Cypher queries. NetworkX is a pure Python library — zero setup, in-process, and sufficient for our IEEE 33-bus scale (33 nodes, 32 edges + rule nodes). We switch to Neo4j only if the graph grows beyond what NetworkX handles comfortably.

### Graph Construction Code (Conceptual)

```python
import networkx as nx

G = nx.DiGraph()

# Add topology from Pandapower network
for bus in net.bus.itertuples():
    G.add_node(f"Bus_{bus.Index}", type="Bus", v_nom=bus.vn_kv)

for line in net.line.itertuples():
    G.add_edge(f"Bus_{line.from_bus}", f"Bus_{line.to_bus}",
               type="connected_to", line_id=line.Index,
               max_loading_pct=line.max_loading_percent)

# Add extracted rules
for rule in extracted_rules:
    G.add_node(rule["rule_id"], type="Rule", **rule)
    G.add_edge(rule["entity_id"], rule["rule_id"], type="has_rule")
```

---

## 5. Component D — Symbolic Validation Shield

### What It Does

Intercepts every GNN prediction. Runs it against all applicable rules in the knowledge graph. Returns either a PASS (with the validated prediction) or a BLOCK (with a structured explanation).

### Why "Shield" and Not a Loss Function Penalty

This is the distinction between Generation 2 and Generation 3 NeSy systems:

- **Gen 2 (PINNs):** Loss function penalizes rule violations. The model is discouraged from breaking physics but not prevented. It can still output a physically impossible voltage.
- **Gen 3 (Shield):** The rule violation is detected at inference time, after the model outputs. The output is structurally blocked from reaching execution. This is a **hard constraint**, not a soft suggestion.

Directly inspired by **Younesi et al. (2026)**: their two-step microgrid system validated actions against a fixed rule set and achieved 91.7% safe power restoration. Our improvement: their rules were manually written. Ours are LLM-generated.

### Validation Logic

```python
def validate(prediction, knowledge_graph):
    context = prediction.context  # current grid state snapshot
    violated = []

    for rule_node in knowledge_graph.get_rules_for(prediction.entity):
        rule = knowledge_graph.nodes[rule_node]
        if evaluate_condition(rule["condition"], context, prediction):
            violated.append(rule)

    if not violated:
        return {"status": "PASS", "output": prediction}

    explanation = build_explanation(violated)
    return {"status": "BLOCK", "violated_rules": violated, "explanation": explanation}


def build_explanation(violated_rules):
    lines = []
    for rule in violated_rules:
        lines.append(
            f"Rule {rule['rule_id']} violated: {rule['explanation']} "
            f"(Source: {rule['source']})"
        )
    return " | ".join(lines)
```

### Example Output (Blocked Prediction)

```
ACTION BLOCKED: Switch OFF Line 3-4

Reason:
  Rule R_042 violated: Voltage at Bus 5 would drop to 0.91 pu.
  Minimum permissible voltage is 0.95 pu per IEEE Std 1547-2018, Section 7.4.

  Rule R_019 violated: Disconnecting Line 3-4 leaves Bus 7 with no alternate
  feed path, violating N-1 security constraint per grid operations manual, p.34.
```

Every blocked decision cites a specific rule and a specific source document — this is the explainability output.

---

## 6. Data Collection Strategy

### Why Simulation, Not Real Data

Real operational data from utilities is proprietary, unavailable without NDAs, and typically anonymized in ways that remove the topology information a GNN needs. This is a known limitation across the entire field — Ahmadi et al. (2026) explicitly identifies this as the "simulation-to-reality gap."

We use **Grid2Op** — a Python framework developed by RTE (the French transmission system operator) specifically for sequential decision-making and AI research on power grids. It implements full power flow equations via a backend (PandaPower by default), enforces thermal limits and N-1 security constraints natively, and is the environment used in the L2RPN (Learn to Run a Power Network) challenge — the most prominent AI-for-grids benchmark in the literature. Simulation on standard benchmarks is the accepted methodology for this research domain.

### The Environments: l2rpn_neurips_2020_track1 (Primary) + l2rpn_wcci_2022 (Scale)

Grid2Op ships several competition environments. Environment choice directly determines the scale and complexity of the problem the GNN must solve. `l2rpn_case14_sandbox` — which we were previously using — is explicitly a development/sandbox environment with only 14 buses and 20 lines. That is not a serious research benchmark; it leaves the majority of the Research PC's compute idle.

**Primary: `l2rpn_neurips_2020_track1`**
- 36 substations, 59 powerlines, 22 generators, 37 loads
- Subset of the IEEE 118-bus grid (the standard large-scale benchmark)
- Used in the NeurIPS 2020 L2RPN competition robustness track
- Ships with two data tiers: `_small` (900MB, ~48 years of 5-min data) and `_large` (4.5GB, ~240 years)
- We use `_small` for development, `_large` for final training runs
- Includes stochastic line disconnections and maintenance events — realistic fault conditions out of the box

**Stretch / Scalability: `l2rpn_wcci_2022`**
- 118 substations, 186 powerlines, 91 loads, 62 generators — full IEEE 118 scale
- Supports `chronix2grid` for infinite synthetic data generation
- Used if we want to demonstrate scalability beyond the primary environment
- Secondary priority; only pursued if the primary benchmark is fully working

### What We Simulate and Log

**Normal operation:** Grid2Op's chronics already provide 48–240 years of realistic load and generation time-series at 5-minute resolution — including renewable variability and demand peaks. No synthetic load profiles needed.

**Fault injection:** Grid2Op natively supports stochastic line disconnections and scheduled maintenance. We additionally inject targeted faults:

| Fault Type | How Injected in Grid2Op | Label |
|---|---|---|
| Line overload | Reduce line thermal limit temporarily; monitor `obs.rho > 1.0` | `overload` |
| Line trip | `action_space({"set_line_status": [(line_id, -1)]})` | `line_trip` |
| Cascading failure | Trip a high-flow line; adjacent lines exceed thermal limit within 1–3 steps | `cascade` |
| Maintenance | Use Grid2Op's built-in maintenance schedule from chronics | `maintenance` |

**Per-step observation (from `obs` returned by `env.step()`):**

| Feature Group | Grid2Op Attribute | Shape |
|---|---|---|
| Line loading ratio | `obs.rho` | `(59,)` — current / thermal limit; >1.0 = overload |
| Active power flow | `obs.p_or`, `obs.p_ex` | `(59,)` each |
| Reactive power flow | `obs.q_or`, `obs.q_ex` | `(59,)` each |
| Voltage at buses | `obs.v_or`, `obs.v_ex` | `(59,)` each |
| Load values | `obs.load_p`, `obs.load_q` | `(37,)` each |
| Generation | `obs.gen_p`, `obs.gen_q` | `(22,)` each |
| Topology vector | `obs.topo_vect` | `(n_sub_elements,)` — busbar assignment per element |
| Line status | `obs.line_status` | `(59,)` — bool |

**Label:** derived from `obs.rho.max() > 1.0` (overload), `obs.line_status` changes (trip/maintenance), and step-over-step cascading detection.

### Data Generation Script (Skeleton)

```python
import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend  # faster backend — use this always
import numpy as np
import json

# Use the NeurIPS 2020 track1 small environment (36 subs, 59 lines)
env = grid2op.make(
    "l2rpn_neurips_2020_track1_small",
    backend=LightSimBackend()  # ~10x faster than default PandaPowerBackend
)

params = Parameters()
params.NO_OVERFLOW_DISCONNECTION = True  # keep overloads alive so we can label them
env.change_parameters(params)

do_nothing = env.action_space({})
records = []

for chronic_id in range(len(env.chronics_handler.subpaths)):
    obs = env.reset()

    for t in range(env.max_episode_duration()):
        action = do_nothing
        fault_label = "normal"
        fault_loc = None

        # Inject line trip with 8% probability
        if np.random.rand() < 0.08:
            line_id = np.random.randint(0, env.n_line)
            action = env.action_space({"set_line_status": [(line_id, -1)]})
            fault_label = "line_trip"
            fault_loc = int(line_id)

        obs, reward, done, info = env.step(action)

        # Overload detection from observation
        if obs.rho.max() > 1.0:
            fault_label = "overload"
            fault_loc = int(obs.rho.argmax())

        records.append({
            "rho":         obs.rho.tolist(),        # (59,) line loading
            "p_or":        obs.p_or.tolist(),        # (59,) active power origin
            "q_or":        obs.q_or.tolist(),
            "v_or":        obs.v_or.tolist(),        # (59,) voltage origin bus
            "load_p":      obs.load_p.tolist(),      # (37,) load active power
            "gen_p":       obs.gen_p.tolist(),       # (22,) gen active power
            "topo_vect":   obs.topo_vect.tolist(),   # topology
            "line_status": obs.line_status.tolist(), # (59,) bool
            "label":       fault_label,
            "fault_loc":   fault_loc,
            "timestep":    t,
            "chronic":     chronic_id
        })

        if done:
            break

with open("grid_dataset.json", "w") as f:
    json.dump(records, f)
```

> **Note:** `LightSimBackend` from the `lightsim2grid` package replaces the default PandaPower backend. It is the recommended backend for any serious data collection — approximately 10x faster for power flow computation, which matters when iterating over thousands of chronic steps.

### Why Not These Alternatives

| Alternative | Why Rejected |
|---|---|
| **Real utility datasets** | Proprietary; topology usually stripped; unavailable |
| **Pandapower (standalone)** | Excellent power flow solver but no built-in episode/chronic management, no RL-compatible step API, and no native fault/maintenance injection framework — we would have to rebuild what Grid2Op already provides |
| **PSCAD / MATLAB** | Commercial license required; not Python-native; no direct PyTorch integration |
| **Random synthetic graphs** | Not comparable to published benchmarks; unreproducible by reviewers |

### Dataset Targets

- **~200,000+ timestep samples** — the `_small` tier alone covers ~48 years at 5-min resolution; we use a sampled subset covering normal, overload, trip, cascade, and maintenance conditions
- **Split:** 70% train / 15% validation / 15% test
- **Stratified** by fault type to prevent class imbalance
- Final training runs use `_large` (240 years of data) if validation metrics plateau on `_small`

---

## 7. GNN Training Pipeline

### From Raw Records to PyG Graph Objects

The JSON records produced by the data collection script are raw arrays — they need to be converted into PyTorch Geometric `Data` objects before training. Each timestep becomes one graph: nodes are substations, edges are powerlines, and labels are the fault classification target.

```python
import torch
from torch_geometric.data import Data, Dataset
import json
import numpy as np

class GridDataset(Dataset):
    def __init__(self, records):
        super().__init__()
        self.records = records

    def len(self):
        return len(self.records)

    def get(self, idx):
        r = self.records[idx]

        # Node features: one row per substation (36 subs)
        # We aggregate line-end features to bus level
        node_feats = build_node_features(r)  # shape: (36, n_node_features)

        # Edge index: (2, 59*2) — bidirectional
        edge_index, edge_attr = build_edge_index_and_features(r)

        # Label encoding: normal=0, overload=1, line_trip=2, cascade=3, maintenance=4
        label = LABEL_MAP[r["label"]]

        return Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(label, dtype=torch.long),
            fault_loc=r["fault_loc"]
        )


def build_node_features(r):
    # Per substation: aggregate load_p, gen_p, and mean voltage of connected lines
    # Shape: (36, 4) — [load_p, gen_p, mean_v_or, max_rho_connected]
    ...  # implemented during coding phase

def build_edge_index_and_features(r):
    # Lines become bidirectional edges
    # Edge features: [rho, p_or, q_or] per line
    # Shape: edge_index (2, 118), edge_attr (118, 3)
    ...
```

### Model Architecture

We implement a two-headed GNN:
- **Classification head:** graph-level output — what type of fault is this? (normal / overload / line\_trip / cascade / maintenance)
- **Localization head:** node-level output — which substation/line is the fault at?

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, n_classes, n_nodes):
        super().__init__()

        # GAT layers — attention weights give us interpretability per edge
        self.conv1 = GATConv(node_features, 64, heads=4, edge_dim=edge_features, dropout=0.2)
        self.conv2 = GATConv(64 * 4, 128, heads=4, edge_dim=edge_features, dropout=0.2)
        self.conv3 = GATConv(128 * 4, 256, heads=1, edge_dim=edge_features, dropout=0.2)

        # Classification head (graph-level)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        # Localization head (node-level)
        self.localizer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # fault probability per node
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Message passing
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        # Node-level: localization logits
        loc_logits = self.localizer(x).squeeze(-1)

        # Graph-level: pool then classify
        graph_emb = global_mean_pool(x, batch)
        class_logits = self.classifier(graph_emb)

        return class_logits, loc_logits
```

### Training Loop

```python
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Split dataset
train_set, val_set, test_set = stratified_split(dataset, ratios=(0.70, 0.15, 0.15))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64)

model = GridGNN(node_features=4, edge_features=3, n_classes=5, n_nodes=36).cuda()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Class weights to handle imbalance (normal >> fault samples)
class_weights = compute_class_weights(train_set).cuda()
cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loc_loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        batch = batch.cuda()
        class_logits, loc_logits = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        cls_loss = cls_loss_fn(class_logits, batch.y)

        # Localization loss only on fault samples (loc target = 1 at fault node, 0 elsewhere)
        if batch.fault_loc is not None:
            loc_targets = build_loc_targets(batch)  # (total_nodes,) binary
            loc_loss = loc_loss_fn(loc_logits, loc_targets)
        else:
            loc_loss = torch.tensor(0.0).cuda()

        loss = cls_loss + 0.5 * loc_loss   # weighted sum; tune alpha if needed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    evaluate(model, val_loader)  # logs to W&B
```

### What Gets Logged to W&B

Every epoch logs: train loss (classification + localization), validation macro F1, per-class F1, localization accuracy (correct fault node in top-3 predictions), and attention weight entropy (a proxy for how focused the model's attention is — useful for explainability analysis).

---

## 8. End-to-End Toy Scenario Testing

This section describes how all four components are wired together and run against hand-crafted scenarios to verify the full pipeline before any formal evaluation.

### The Toy Scenario Setup

We use `rte_case5_example` — Grid2Op's 5-bus, 8-line minimal environment — specifically for toy testing. It is small enough that every state can be manually verified by inspection, but it runs through the exact same code path as the full environment. This is the integration test harness, not a benchmark.

Three toy scenarios are defined manually:

**Scenario 1 — Clean pass (should not be blocked)**
- Grid state: all lines operating at 60–80% load, voltages nominal
- GNN prediction: "normal — no action required"
- Expected shield output: PASS

**Scenario 2 — Overload (should be blocked)**
- Grid state: Line 3 at 115% thermal limit (`obs.rho[3] = 1.15`)
- GNN prediction: "switch off Line 3"
- LLM-extracted rule R_011: "IF rho > 1.0 on any line THEN action must reduce loading, not disconnect without alternate path"
- Expected shield output: BLOCK + explanation citing R_011

**Scenario 3 — Cascading risk (should be blocked with multi-rule explanation)**
- Grid state: Lines 2 and 5 at 90% load
- GNN prediction: "disconnect Line 2 to reduce load"
- LLM-extracted rules: R_011 (thermal limit), R_019 (N-1 security — no bus left isolated)
- Expected shield output: BLOCK + explanation citing both R_011 and R_019

### The Full Wiring Code

```python
import grid2op
from lightsim2grid import LightSimBackend
import torch
import networkx as nx

# ── 1. Load environment ──────────────────────────────────────────────────────
env = grid2op.make("rte_case5_example", backend=LightSimBackend())

# ── 2. Load trained GNN ─────────────────────────────────────────────────────
model = GridGNN(node_features=4, edge_features=3, n_classes=5, n_nodes=5)
model.load_state_dict(torch.load("gnn_checkpoint.pt"))
model.eval()

# ── 3. Load knowledge graph (built from LLM extraction run) ─────────────────
KG = nx.read_gpickle("knowledge_graph.gpickle")

# ── 4. Define toy scenarios ──────────────────────────────────────────────────
scenarios = [
    {"name": "clean_pass",      "inject": None},
    {"name": "overload",        "inject": {"set_line_status": [(3, -1)]}},  # trip line 3
    {"name": "cascade_risk",    "inject": {"set_line_status": [(2, -1)]}},  # trip line 2
]

for scenario in scenarios:
    obs = env.reset()

    # Apply scenario setup if needed
    if scenario["inject"]:
        action = env.action_space(scenario["inject"])
        obs, _, _, _ = env.step(action)

    # ── 5. Convert observation to PyG graph ──────────────────────────────────
    graph = obs_to_pyg(obs)  # same function used during training

    # ── 6. GNN inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        class_logits, loc_logits = model(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )
    predicted_class = class_logits.argmax().item()
    predicted_action = CLASS_TO_ACTION[predicted_class]

    # ── 7. Build prediction context for the shield ───────────────────────────
    prediction = Prediction(
        action=predicted_action,
        context={
            "rho":         obs.rho.tolist(),
            "line_status": obs.line_status.tolist(),
            "v_or":        obs.v_or.tolist(),
        },
        entity="grid"
    )

    # ── 8. Shield validation ──────────────────────────────────────────────────
    result = validate(prediction, KG)

    # ── 9. Print result ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Scenario : {scenario['name']}")
    print(f"GNN says : {predicted_action}")
    print(f"Shield   : {result['status']}")
    if result["status"] == "BLOCK":
        print(f"Reason   : {result['explanation']}")
```

### Expected Console Output

```
============================================================
Scenario : clean_pass
GNN says : no_action
Shield   : PASS

============================================================
Scenario : overload
GNN says : disconnect_line_3
Shield   : BLOCK
Reason   : Rule R_011 violated: Action would disconnect Line 3 without a verified
           alternate feed path for Bus 4. N-1 security constraint not satisfied.
           (Source: IEEE Std C37.2, Section 5.2)

============================================================
Scenario : cascade_risk
GNN says : disconnect_line_2
Shield   : BLOCK
Reason   : Rule R_011 violated: Line 2 loading at 90% — disconnection triggers
           thermal cascade on Line 5 (projected rho: 1.21).
           (Source: IEEE Std 1547-2018, Section 7.4) |
           Rule R_019 violated: Bus 3 has no alternate feed after Line 2 removal.
           (Source: Grid Operations Manual, Section 4.2)
```

### What This Validates

Running all three scenarios successfully confirms:
1. The GNN produces outputs in the expected format and can be converted from an observation without errors
2. The knowledge graph was correctly built from LLM extraction and is queryable
3. The shield correctly passes valid states and blocks invalid ones
4. Explanations are traceable to specific rule IDs and source documents
5. The full pipeline runs end-to-end without integration failures

Only after all three toy scenarios pass do we move to formal evaluation on held-out chronics from `l2rpn_neurips_2020_track1_small`.

---

## 9. Technology Stack

| Component | Tool | Justification |
|---|---|---|
| Grid simulation | **Grid2Op** | Built by RTE (French TSO); native chronic/episode management; thermal limit enforcement; L2RPN benchmark environment; RL-compatible step API |
| Grid2Op backend | **lightsim2grid** | ~10x faster power flow solver than Grid2Op's default PandaPower backend; mandatory for any serious data generation run |
| GNN framework | **PyTorch + PyTorch Geometric** | De facto standard for GNN research; full architecture flexibility |
| LLM inference | **llama-cpp-python + Qwen2.5-72B / Llama 3.3-70B (4-bit)** | Local, reproducible; sequential two-stage pipeline — Extractor then Validator; GBNF grammar enforcement on output |
| LLM orchestration | **LangChain** | PDF document loaders, chunking, prompt chaining |
| Knowledge graph | **NetworkX** → **Neo4j** if needed | Zero-setup for research scale; Neo4j only if Cypher querying becomes necessary |
| Shield logic | **Python (custom)** | Fully auditable, no external dependencies, easy to unit test |
| Data validation | **Pydantic** | Schema enforcement on LLM JSON outputs |
| Experiment tracking | **Weights & Biases** | Training metrics, hyperparameter logging, run comparison |
| Version control | **Git + GitHub** | Standard |

---

## 10. Hardware & Compute Allocation

### Research PC — Primary Compute (All Heavy Workloads)
- **CPU:** Intel Core i7-14700K (20 cores / 28 threads)
- **GPU:** NVIDIA RTX 4080 Super — **16GB VRAM**
- **RAM:** DDR5 64GB
- **Runs:** GNN training, large-scale dataset generation, LLM knowledge extraction (Qwen2.5-72B Extractor + Llama 3.3-70B Validator, sequential), hyperparameter sweeps

### Personal PC — Development & Light Testing Only
- **CPU:** AMD Ryzen 5 7500F
- **GPU:** Intel Arc B580 — **12GB VRAM**
- **RAM:** DDR5 16GB
- **Runs:** Code development, unit tests, shield logic, small-scale smoke tests

### Implications for Implementation

- All production runs (data generation, LLM extraction, GNN training) execute on the Research PC
- Personal PC is for writing and testing code only — do not run the full pipeline there
- LLM extraction uses two sequential passes: Qwen2.5-72B (Extractor, ~40–44GB total with majority CPU offload) then Llama 3.3-70B (Validator, ~40–44GB). Both run one at a time; 64GB system RAM handles the offload for either model
- GNN must train in under 2 hours per run on the RTX 4080 Super — if it doesn't, reduce model size or dataset batch size first
- No multi-GPU; keep everything single-device

---

## 11. Evaluation Plan

### GNN (Neural Component)

| Metric | Description |
|---|---|
| Accuracy | Overall correct fault classification |
| Macro F1-score | F1 averaged across all fault classes equally |
| Precision / Recall | Per fault class — especially for rare fault types |

Baseline comparisons: flat MLP on same data, LSTM on time-series version of same data. GNN should outperform both due to topology awareness.

### Shield (Symbolic Validation)

| Metric | Description |
|---|---|
| Rule compliance rate | % of passed predictions that satisfy all applicable rules |
| False block rate | % of valid predictions incorrectly blocked |
| Block precision | % of blocked predictions that were genuinely rule-violating |

Test: Feed a set of deliberately crafted rule-violating predictions. Shield must block 100% of them. Feed valid predictions. False block rate should be near zero.

### LLM Rule Extraction

| Metric | Description |
|---|---|
| Rule coverage | % of known ground-truth rules successfully extracted |
| Rule precision | % of extracted rules that are correct (manually verified) |
| Parse success rate | % of LLM outputs that are valid JSON matching the schema |

Ground truth: a manually compiled reference set of rules from IEEE 1547 and one grid operations manual, independently verified by the team.

### End-to-End System

- Run held-out chronics from `l2rpn_neurips_2020_track1_small` with injected faults
- Compare: GNN-only output vs. GNN + Shield output
- Metrics: safe action rate, blocked action rate, explanation quality (human-rated 1–5)
- Reference target: Younesi et al. (2026) achieved 91.7% safe restoration with manually written rules — our system targets matching or exceeding that with automated rule generation

---

## 12. Out of Scope

These are decided exclusions. Do not prototype or propose these.

- Real utility deployment or hardware-in-the-loop
- Training any foundation model (LLMs are used pre-trained, quantized only)
- Transmission-level grids (we work on distribution level via Grid2Op's built-in environments only)
- Multi-agent / multi-microgrid coordination
- Blockchain, digital twins, or metaverse integration
- Reinforcement Learning agent (future work only)

---

## 13. Glossary

| Term | Definition |
|---|---|
| **GNN** | Graph Neural Network — processes graph-structured input natively |
| **GCN** | Graph Convolutional Network — a specific GNN architecture using spectral convolution |
| **GAT** | Graph Attention Network — GNN variant where edges have learned attention weights |
| **Knowledge Graph (KG)** | Graph of entities, relationships, and rules — our symbolic rule store |
| **Shield** | The symbolic validation layer — hard-blocks rule-violating GNN outputs |
| **Grid2Op** | Python power grid simulation framework by RTE; manages episodes, chronics, and fault injection for AI research |
| **Chronic** | A time-series of load and generation values used as one simulation episode in Grid2Op |
| **`obs.rho`** | Grid2Op observation attribute — ratio of current flow to thermal limit per line; >1.0 means overload |
| **`rte_case5_example`** | Grid2Op's 5-bus, 8-line minimal environment — used exclusively for toy scenario integration testing |
| **`l2rpn_neurips_2020_track1`** | Grid2Op's NeurIPS 2020 competition environment — 36 substations, 59 lines, subset of IEEE 118. Our primary benchmark |
| **`l2rpn_wcci_2022`** | Grid2Op's WCCI 2022 environment — full IEEE 118 scale (118 subs, 186 lines). Used for scalability testing |
| **LightSimBackend** | Fast power flow backend for Grid2Op (~10x faster than default); from the `lightsim2grid` package |
| **Power flow** | Mathematical solution for voltage/current/power at every grid node |
| **Fault injection** | Programmatically forcing a fault condition (overload, undervoltage, line trip) in simulation |
| **Voltage pu** | Voltage in per-unit — normalized so 1.0 = nominal voltage |
| **Knowledge bottleneck** | The problem that symbolic rules in NeSy systems must be written by hand — LLM extraction solves this |
| **LangChain** | Python framework for orchestrating LLM pipelines with structured outputs |
| **NetworkX** | Python graph library used as our initial knowledge graph backend |
| **4-bit quantization** | Model compression that reduces LLM memory ~4×; allows Qwen2.5-32B (~18–20GB) to run on the Research PC with minor system RAM offload |

---

## 14. Disclaimer — Living Document

> **Nothing in this document is final.**
>
> All model names, dataset choices, simulation environments, library selections, and architectural decisions recorded here reflect the best available information at the time of writing. Any of these may change as new findings, benchmarks, hardware constraints, or implementation realities emerge during the course of the thesis.
>
> Specific items subject to change without notice:
> - LLM model selection (e.g. Qwen2.5-32B, Mistral-22B, formatter models)
> - GNN architecture choice (GCN vs GAT vs alternatives)
> - Grid2Op environments and their version-specific behavior
> - Knowledge graph backend (NetworkX vs Neo4j)
> - Dataset size targets and split ratios
> - Evaluation metrics and baseline comparisons
>
> When a decision changes, the relevant section of this document is updated to reflect the new choice and the reasoning behind the change. Previous decisions are not preserved — this document describes the current state, not the history.

---

*Last updated: April 2026.*
