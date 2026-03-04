# DiffAlign-FM

A molecular alignment framework that replaces the diffusion process in DiffAlign with flow matching.

## Overview

This project builds on [DiffAlign](https://github.com/kim-iljung/DiffAlign), a conditional E(3)-equivariant diffusion model for flexible molecular alignment. The goal is to improve DiffAlign by replacing the diffusion generative framework with flow matching, following the approach used in [ET-Flow](https://github.com/sygil-dev/et-flow).

## Roadmap

- [x] Phase 0: Understand and reproduce DiffAlign (author's experimental code)
- [ ] Phase 1: Replace diffusion → flow matching (keep EGNN backbone)
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

## Repository Structure
```
models/
  encoder/
    egnn.py          # E(3)-equivariant GNN (Phase 1 backbone)
  epsnet/
    diffusion.py     # Original DiffAlign (baseline)
    flow.py          # FlowAlign - flow matching replacement (Phase 1)
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
