# RAL-Opt: Retrieval-Augmented Learning for Post-ECO Slack Prediction

A demonstration of the RAL-Opt framework applied to predicting post-ECO timing slack in VLSI 
physical design. Due to confidentiality constraints, all EDA tool outputs are sanitized.

## Overview

RAL-Opt is a two-stage learning-based framework that enhances parametric ML-based ECO 
optimization on timing graph representations of chip designs.

1. **Chunk & Store** — Parse timing reports, build timing graphs, and export
   k-hop neighborhoods and endpoint logic cones as serialized DGL sub-graphs.
2. **Distill** — Train a teacher (SGFormer on logic cones) and distill its
   knowledge into a lightweight student (GNN on 3-hop chunks).  Build a
   retrieval index from the student embeddings.
3. **RAL** — Train a cross-attention decoder that refines teacher predictions
   using retrieved similar sub-graphs from the index.

## Project Structure

```
ral_opt/
├── main.py                  # Unified entry point for all stages
├── config/
│   ├── global.json          # Shared settings (designs, features, device)
│   ├── chunk.json           # Chunk & store parameters
│   ├── distill.json         # Distillation & index building parameters
│   └── ral.json             # RAL decoder training parameters
├── models/
│   ├── base_models.py       # MLP, NodeFeatureBuilder, LocalGNN, TransConv
│   ├── ral_opt.py           # Teacher (SGFormer) and Student (3-hop GNN)
│   └── ral_decoder.py       # Retrieval index and RAL decoder
├── ral_opt/
│   ├── pretrain_distill.py  # Distillation training loop & index builder
│   ├── train_ral.py         # RAL decoder training loop & evaluation
│   └── ral_var.py           # Checkpoint path helpers
├── data/
│   ├── TimingGraph.py       # Timing graph construction from EDA reports
│   ├── Chunk_Store.py       # Sub-graph extraction and serialization
│   └── Data_var.py          # Data directory paths
├── parsers/                 # EDA report parsers (PrimeTime, ICC2)
├── utils/
│   ├── env_setup.py         # Reproducibility and warning setup
│   ├── eval_report.py       # Regression metrics and reporting
│   ├── selected_cell.py     # Cell type/size parsing from library names
│   ├── chunk_graph_data.py  # Chunk dataset and dataloader utilities
│   ├── distill_graph_data.py# Cone-to-outpin dataset for distillation
│   └── ral_graph_data.py    # Cone query dataset for RAL
└── work/
    ├── work_var.py          # Design directory mappings
    ├── pt_data/             # Sanitized PrimeTime reports (per design)
    └── icc2_data/           # Sanitized ICC2 reports (per design)
```

## Quick Start

### 1. Install Dependencies

```bash
conda create -n ral_opt python=3.12
conda activate ral_opt
pip install -r requirements.txt
```

> **Note:** DGL and PyTorch with CUDA support require matching CUDA versions.
> See [DGL installation](https://www.dgl.ai/pages/start.html) and
> [PyTorch installation](https://pytorch.org/get-started/locally/) for details.

### 2. Configure

Edit `config/global.json` to set your training/test design split and device:

```json
{
    "y_keys": ["slack_eco"],
    "train_designs": ["design_i", "design_k"],
    "test_designs":  ["design_j"],
    "device": "cuda"
}
```

### 3. Run the Pipeline

Execute the three stages sequentially:

```bash
# Stage 1: Build timing graphs and export sub-graph chunks
python main.py config/chunk.json

# Stage 2: Teacher-student distillation + build retrieval index
python main.py config/distill.json

# Stage 3: RAL decoder training with retrieval augmentation
python main.py config/ral.json
```

Each stage reads `config/global.json` for shared settings and merges with
its own stage-specific config file.

## Data

The `work/` directory contains sanitized EDA outputs organized by design:

- `work/pt_data/<design>/` — PrimeTime timing reports (pin slack, arrival,
  transition, cell reports, ECO change lists)
- `work/icc2_data/<design>/` — ICC2 physical reports (pin coordinates, block info)

All cell names, net names, and numeric values have been anonymized.
The data is sufficient to demonstrate the full RAL-Opt pipeline.

## Output

- **Checkpoints** are saved under `ral_opt/checkpoints/`:
  - `ral_distill/<y_key>/student_best.pt` — Best student encoder
  - `ral_distill/<y_key>/teacher_best.pt` — Best teacher model
  - `ral_train/<y_key>/decoder.pt` — Best RAL decoder
- **Chunk data** is stored under `data/chunk/` and `data/ep_cones/`
- **Retrieval index** is stored under `data/chunk/retrieval_index/`

## License

This project is for demonstration purposes only. All data has been sanitized
and does not contain proprietary design information.
