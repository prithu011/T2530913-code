# GEMINI.md

## ⚡ Project Overview: Neuro-Symbolic Grid

This project implements a **Neuro-Symbolic AI system** designed for power grid fault detection and action validation. It bridges the gap between neural perception (GNN) and symbolic reasoning (Knowledge Graph) to ensure safety-critical decisions in power systems.

### 🏗️ Core Architecture
The system follows a four-component "Shield" architecture:
1.  **Component A: GNN (Neural Layer)** - Uses PyTorch Geometric (GAT/GCN) to perceive grid states, classify faults, and recommend actions.
2.  **Component B: LLM Extraction Pipeline** - Uses Ollama (Qwen2.5-32B) and LangChain to extract symbolic rules from grid standards (IEEE, manuals).
3.  **Component C: Knowledge Graph** - A NetworkX-based store that maps grid topology to extracted symbolic rules.
4.  **Component D: Symbolic Validation Shield** - A hard constraint layer that intercepts GNN outputs and blocks any that violate the symbolic rules.

**Non-Negotiable Principle:** GNN outputs are *never* final without symbolic validation.

---

## 🚀 Building and Running

### 📦 Environment Setup
The project requires a specific installation order for CUDA-enabled PyTorch and PyTorch Geometric.

```bash
# 1. Install PyTorch (optimized for Research PC with RTX 4080 Super / CUDA 12.8)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Install PyTorch Geometric
pip install torch_geometric

# 3. Install PyG Dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 4. Install remaining requirements
pip install -r requirements.txt
```

### 🛠️ Key Commands

| Task | Command |
| :--- | :--- |
| **Verification** | `python sanity/verify_grid2op.py` or `python sanity/verify_stack.py` |
| **Data Generation** | `python scripts/generate_dataset.py` |
| **GNN Training** | `python training/train_gnn.py` |
| **LLM Extraction** | `ollama run qwen2.5:32b` (configured via scripts) |
| **End-to-End Test** | `python -c "import grid2op; ..."` (refer to `study.md` section 8) |

---

## 📂 Project Structure

-   `scripts/`: Data handling and processing.
    -   `generate_dataset.py`: Grid2Op simulation runner for data collection.
    -   `pyg_data.py`: Dataset and loader definitions for PyTorch Geometric.
    -   `split.py`: Train/Val/Test splitting logic.
-   `training/`: GNN model definitions and training loops.
    -   `train_gnn.py`: Main training entry point.
-   `sanity/`: Lightweight verification scripts to check CUDA, Grid2Op, and stack integrity.
-   `data/`: (Ignored/External) Storage for generated datasets and simulation chronics.
-   `study.md`: **The Primary Reference.** Contains full methodology, implementation specs, and logic snippets.
-   `CLAUDE.md`: Operational guide for AI assistants in this workspace.

---

## ⚖️ Development Conventions

1.  **Hardware Awareness**: Heavy workloads (training, extraction, large-scale generation) MUST run on the **Research PC** (RTX 4080 Super). The Development PC is for code editing and smoke tests only.
2.  **Safety First**: The "Validation Shield" logic (Component D) is the highest priority. Any modification to the GNN must be accompanied by a check on how it interacts with the Shield.
3.  **Local-First LLM**: All rule extractions must use local Ollama instances for reproducibility and privacy. Do not use external API calls (OpenAI/Claude) for the core extraction pipeline.
4.  **Schema Enforcement**: Use **Pydantic** for all data structures, especially those handling LLM outputs or grid state snapshots.

---

## 📚 Documentation Reference
For deep dives into the math, architecture decisions, or specific training parameters, refer to the following sections in `study.md`:
-   **Section 2**: GNN Architecture & Features
-   **Section 3-5**: LLM Pipeline & Knowledge Graph
-   **Section 8**: Toy Scenario Validation (The "Shield" logic)
