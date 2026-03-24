# DiffAlign-FM

A molecular alignment framework that replaces the diffusion process in DiffAlign with flow matching.

## Overview

This project builds on [DiffAlign](https://github.com/kim-iljung/DiffAlign), a conditional E(3)-equivariant diffusion model for flexible molecular alignment. The goal is to improve DiffAlign by replacing the diffusion generative framework with flow matching, following the approach used in [ET-Flow](https://github.com/shenoynikhil/ETFlow).

## Roadmap

- [x] Phase 0: Understand and reproduce DiffAlign (author's experimental code)
- [x] Phase 1: Replace diffusion → flow matching (keep EGNN backbone)
- [ ] Phase 2: Replace EGNN → Equivariant Transformer (keep flow matching)

## Key Changes from DiffAlign

### Phase 1: Flow Matching
- Replace DDPM noising schedule with linear interpolation
- Replace v-parameterization loss with velocity matching loss
- Replace DDPM sampler with Euler ODE integration
- Keep EGNN + CrossGraphAligner backbone unchanged
- Keep inference-time UFF/Vina steering unchanged

### Phase 2: Equivariant Transformer (planned)
- Replace EGNN with Equivariant Transformer from ET-Flow
- Keep flow matching framework from Phase 1

## Results

Evaluated on the [DISCO benchmark](https://github.com/...) (success rate %).

| Method | RMSD < 1 Å | RMSD < 2 Å | RMSD < 3 Å |
|---|---|---|---|
| DiffAlign + UFF with pocket (paper) | 6.0 | 18.4 | 27.9 |
| DiffAlign baseline | 5.6 | 19.2 | 30.0 | 
| FlowAlign baseline | 2.6 | 13.1 | 26.2 |
| **FlowAlign (ckpt 188)** | **6.6** | **19.4** | **31.7** |

FlowAlign (ckpt 188) outperforms the original diffusion-based DiffAlign across all thresholds, validating the flow matching replacement.

## Repository Structure

```
models/
  encoder/
    egnn.py          # E(3)-equivariant GNN (Phase 1 backbone)
  epsnet/
    diffusion.py     # Original DiffAlign (baseline)
    flow.py          # FlowAlign — flow matching replacement (Phase 1)
utils/
  datasets.py        # Data loading
  chem.py            # Chemistry utilities
train.py             # Original DiffAlign training script
train_flow.py        # FlowAlign training script (Phase 1)
```

## Data

Training uses a dataset of 74K molecular pairs (query + reference ligands), stored as PyTorch Geometric `Data` objects.

## Branches

- `main` — baseline DiffAlign (author's experimental code)
- `flow-matching` — Phase 1: flow matching replacement
