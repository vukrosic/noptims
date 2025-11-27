# noptims

YouTube course [Code, Write & Publish AI Research Paper - Full Course - Muon vs Adam Optimizer] - https://youtu.be/O2yAMJu8LpI



## Highlights
- End-to-end MoE training stack built on PyTorch, AMP, and Hugging Face datasets.
- Custom Muon optimizer (`optimizers/muon.py`) plus hybrid Muon+AdamW scheduling for stability.
- Pluggable dataset/tokenizer configs and streaming-friendly data pipeline.
- Experiment harness for optimizer sweeps, logging, plotting, and checkpointing.
- Reproducible write-up sources in `paper.tex` / `paper.pdf`.

## Repository Layout
- `train_moe.py` – CLI entry point that wires configs, dataset prep, training, and checkpoint export.
- `configs/` – dataclass-based configs (`moe_config.py`, `dataset_config.py`) shared across scripts.
- `data/` – tokenizer setup, streaming dataset utilities, and DataLoader helpers.
- `models/` – minimal MoE LLM building blocks (`components.py`, `layers.py`, `moe_llm.py`).
- `optimizers/` – Muon optimizer implementation and related math kernels.
- `training/` – trainer loop, evaluation helpers, metric plotting, and logging utilities.
- `experiments/exp1_muon_vs_adam/` – scripts for Muon vs AdamW sweeps and reporting.
- `utils/` – miscellaneous helpers (seed control, logging setup).

## Quickstart
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Launch the default MoE training run:
   ```bash
   python train_moe.py
   ```
   The script:
   - Prints hardware/torch info, seeds RNGs, and instantiates `MoEModelConfig`.
   - Streams documents from `HuggingFaceTB/smollm-corpus`, performs a document-level train/val split, then tokenizes each split separately to avoid leakage.
   - Builds PyTorch `DataLoader`s, initializes the MoE LLM, and trains with mixed precision plus gradient clipping.
   - Logs metrics to both stdout and `./logs`, evaluates every `eval_every` steps, and saves the final checkpoint to `./checkpoints/final_model.pt`.

## Customization Tips
- **Model shape**: Edit `configs/moe_config.py` to adjust depth, width, expert count, or optimizer hyperparameters. The config dataclass computes derived dimensions and asserts valid settings.
- **Dataset/tokenizer**: Update `DataConfig` inside `train_moe.py` or create a new config module referencing different Hugging Face datasets, custom preprocessing callbacks, or cached on-disk datasets.
- **Optimizers**: `training/trainer.py` exposes `setup_muon_optimizer` for hybrid Muon+AdamW training; replace or extend this function to test alternative optimizers.
- **Experiments**: Use `experiments/exp1_muon_vs_adam/run_experiments.py` and related sweep scripts to reproduce optimizer comparisons. These utilities share the same trainer core but attach experiment-specific logging, plots, and JSON summaries.

## Logging & Outputs
- Checkpoints: `./checkpoints/` (default file `final_model.pt` with model state, config, and metrics).
- Metrics & plots (for experiment runs): saved under each experiment’s `output_dir` as `metrics.json`, `metrics_plot.png`, and `model.pt`.
- Console + file logging via `utils/logger.py`, which timestamps events and mirrors critical metrics.
