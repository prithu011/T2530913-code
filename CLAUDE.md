# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support (RTX 4080 Super)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch Geometric dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install remaining dependencies
pip install -r requirements.txt
```

### Data Inspection
```bash
# Inspect dataset statistics
python scripts/inspect.py [data_file_path]
# Default: data/grid_dataset_neurips2020.jsonl
```

### Model Training
```bash
# Train GNN model (default: neurips environment)
python training/train_gnn.py

# Train with specific environment
python training/train_gnn.py --env wcci

# Customize training parameters
python training/train_gnn.py --epochs 100 --batch_size 64 --lr 0.001
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports
isort .

# Lint with ruff
ruff check .

# Type checking (if configured)
# mypy .  # or pyright
```

## Project Architecture

### Core Components
1. **Data Layer** (`data/`)
   - JSONL format datasets: `grid_dataset_neurips2020.jsonl`, `grid_dataset_wcci2022.jsonl`
   - Contains grid observations (rho, load_p, gen_p) and fault labels

2. **Processing Scripts** (`scripts/`)
   - `inspect.py`: Dataset analysis and statistics
   - `pyg_data.py`: PyTorch Geometric dataset wrapper and preprocessing
   - `split.py`: Train/validation/test splitting and class weight computation

3. **Training Pipeline** (`training/`)
   - `train_gnn.py`: Main training script implementing:
     - GridGNN model (3-layer GAT with global mean pooling)
     - Dual-output architecture (classification + localization)
     - Training loop with validation and checkpointing
     - Environment-specific configuration (neurips/wcci)

4. **Model Architecture**
   - **GridGNN**: Graph Attention Network (GAT) based
     - 3 GATConv layers with increasing hidden dimensions (64→256→256)
     - Edge feature integration throughout
     - Classification head: MLP for fault type prediction
     - Localization head: MLP for fault node localization
   - **Loss Function**: Weighted CrossEntropy + BCEWithLogits (0.5 weight)

### Key Data Structures
- **GridDataset**: Wraps JSONL data for PyTorch Geometric
- **GridEnvMetadata**: Environment-specific metadata (node/edge counts)
- **LABEL_MAP**: Mapping from fault types to integer labels

### Dependencies
- PyTorch 2.8.0 + CUDA 12.8
- PyTorch Geometric
- grid2op, lightsim2grid (grid simulation)
- langchain, ollama (LLM pipeline)
- networkx (knowledge graph)
- pydantic, numpy, scikit-learn (data/validation)
- wandb (experiment tracking)
- tqdm (utilities)

### Notes
- Uses Windows-compatible DataLoader settings (num_workers=0)
- Implements early stopping via best validation accuracy tracking
- Saves model checkpoints as `gnn_checkpoint_{env}_best.pt`
- Fault localization uses binary cross-entropy with node-level targets