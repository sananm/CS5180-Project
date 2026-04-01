# GCP Training Guide

Everything needed to produce trained checkpoints for the paper.

## Prerequisites

1. GCP T4 instance with Python 3.11, PyTorch 2.6, CUDA
2. Clone repo and install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Confirm GPU is visible:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

---

## Step 1: Train AlphaZero

```bash
python3 experiments/train_alphazero.py \
    --device cuda \
    --num-iterations 200 \
    --games-per-iter 100 \
    --num-sims 100 \
    --eval-every 10 \
    --eval-games 50 \
    --checkpoint-dir checkpoints \
    --output-dir results/alphazero_training \
    --seed 42
```

**Outputs:**
- `checkpoints/az_iter0010.pt`, `az_iter0020.pt`, ... (every 10 iters)
- `checkpoints/az_best_vs_random.pt` (best checkpoint vs Random)
- `checkpoints/az_best_vs_minimax4.pt` (best checkpoint vs Minimax-4)
- `checkpoints/az_final.pt` (last iteration)
- `results/alphazero_training/training_curves.csv`
- `results/alphazero_training/loss_curves.csv`

**Targets:** ≥90% vs Random, ≥60% vs Minimax-4. Script stops early when both are met.

**Resume if interrupted:**
```bash
python3 experiments/train_alphazero.py ... --resume checkpoints/az_iter0050.pt
```

---

## Step 2: Train PPO (3 runs for ABL-02)

### Main run (pool_size=20, used for head-to-head and ABL-02 pool=20)
```bash
python3 experiments/train_ppo.py \
    --device cuda \
    --n-updates 500 \
    --pool-size 20 \
    --episodes-per-update 10 \
    --eval-every 25 \
    --eval-games 50 \
    --checkpoint-dir checkpoints \
    --output-dir results/ppo_training \
    --run-name ppo_pool20 \
    --seed 42
```

### ABL-02: pool_size=5
```bash
python3 experiments/train_ppo.py \
    --device cuda \
    --n-updates 500 \
    --pool-size 5 \
    --episodes-per-update 10 \
    --eval-every 25 \
    --eval-games 50 \
    --checkpoint-dir checkpoints \
    --output-dir results/ppo_training \
    --run-name ppo_pool5 \
    --seed 42
```

### ABL-02: pool_size=1 (no opponent pool — always plays self)
```bash
python3 experiments/train_ppo.py \
    --device cuda \
    --n-updates 500 \
    --pool-size 1 \
    --episodes-per-update 10 \
    --eval-every 25 \
    --eval-games 50 \
    --checkpoint-dir checkpoints \
    --output-dir results/ppo_training \
    --run-name ppo_pool1 \
    --seed 42
```

**Outputs per run:**
- `checkpoints/ppo_pool{N}_upd00025.pt`, ... (every 25 updates)
- `checkpoints/ppo_pool{N}_best_vs_random.pt`
- `checkpoints/ppo_pool{N}_best_vs_minimax4.pt`
- `checkpoints/ppo_pool{N}_final.pt`
- `results/ppo_training/ppo_pool{N}/training_curves.csv`
- `results/ppo_training/ppo_pool{N}/loss_curves.csv`

**Targets:** ≥90% vs Random, ≥50% vs Minimax-4.

---

## Step 3: Run Experiment Scripts (need trained checkpoints)

### Head-to-head + color analysis (EVAL-05, EVAL-07)
```bash
python3 experiments/run_head_to_head.py \
    --az-checkpoint checkpoints/az_final.pt \
    --ppo-checkpoint checkpoints/ppo_pool20_final.pt \
    --num-games 200 \
    --az-sims 200 \
    --output-dir results/head_to_head \
    --device cuda
```

### MCTS sim ablation (ABL-01)
```bash
python3 experiments/run_ablation_mcts.py \
    --az-checkpoint checkpoints/az_final.pt \
    --num-games 100 \
    --output-dir results/ablation_mcts \
    --device cuda
```

### Compute efficiency timing (EVAL-06)
```bash
python3 experiments/run_compute_efficiency.py \
    --algorithm az --n-iterations 20 --output-dir results/compute_efficiency --device cuda
python3 experiments/run_compute_efficiency.py \
    --algorithm ppo --n-iterations 20 --output-dir results/compute_efficiency --device cuda
```

### Loss/entropy curves (EVAL-08)
```bash
# AlphaZero loss curves are already in results/alphazero_training/loss_curves.csv
# PPO loss curves are already in results/ppo_training/ppo_pool20/loss_curves.csv
# Generate the multi-panel figure:
python3 experiments/run_loss_curves.py \
    --az-loss-csv results/alphazero_training/loss_curves.csv \
    --ppo-loss-csv results/ppo_training/ppo_pool20/loss_curves.csv \
    --output-dir results/figures
```

---

## Step 4: Generate Figures

```bash
# Training curves
python3 -c "
from othello_rl.evaluation.plotting import plot_training_curves
plot_training_curves('results/alphazero_training/training_curves.csv', 'results/figures/az_training.png', agent_name='AlphaZero')
plot_training_curves('results/ppo_training/ppo_pool20/training_curves.csv', 'results/figures/ppo_training.png', agent_name='PPO')
"

# Color asymmetry (from head-to-head run)
python3 -c "
from othello_rl.evaluation.plotting import plot_color_asymmetry
import csv
records = list(csv.DictReader(open('results/head_to_head/color_analysis.csv')))
plot_color_asymmetry(records, 'results/figures/color_asymmetry.png')
"

# Ablation results
python3 -c "
from othello_rl.evaluation.plotting import plot_ablation_results
plot_ablation_results('results/ablation_mcts/ablation_results.csv', 'results/figures/ablation_mcts.png', x_key='num_sims', x_label='MCTS Simulations')
"
```

---

## Checkpoint Summary for Paper

| File | Used for |
|------|----------|
| `checkpoints/az_final.pt` | Head-to-head, ABL-01 |
| `checkpoints/ppo_pool20_final.pt` | Head-to-head, ABL-02 pool=20 |
| `checkpoints/ppo_pool5_final.pt` | ABL-02 pool=5 |
| `checkpoints/ppo_pool1_final.pt` | ABL-02 pool=1 |
