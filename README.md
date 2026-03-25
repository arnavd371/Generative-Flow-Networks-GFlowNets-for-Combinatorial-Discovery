# Generative Flow Networks for Combinatorial Discovery

Minimal research-style website for GFlowNets on discrete combinatorial search.

## Milestone 1 (Sorting Networks Environment)

The repository now includes a Gym-like DAG environment implementation at:

- `sorting_network_env.py`

It provides:

- `reset()`
- `step(action)` (forward transition)
- `backward_step(action)` (reverse transition)
- `get_mask()` and `get_backward_mask()` for legal-action masking

## Milestone 2 (Maximum Independent Set Environment)

The repository now includes a Gym-like DAG environment implementation at:

- `mis_env.py`

It provides:

- `reset()`
- `step(action)` (forward transition)
- `backward_step(action)` (reverse transition)
- `get_mask()` and `get_backward_mask()` for legal-action masking

## View locally

Open `index.html` directly, or run:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## GitHub Pages

This repository deploys via GitHub Actions to:

`https://arnavd371.github.io/Generative-Flow-Networks-GFlowNets-for-Combinatorial-Discovery/`

If the URL is not live yet, enable GitHub Pages in repository settings and set
source to **GitHub Actions**.
