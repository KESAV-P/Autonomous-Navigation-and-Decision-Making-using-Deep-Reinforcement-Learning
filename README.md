# Autonomous Navigation using Deep Reinforcement Learning (CNN-LSTM-Actor-Critic + PPO)

**Author:** Kesav  
**Framework:** `stable-baselines3` + `sb3-contrib` (RecurrentPPO)  
**Environment:** `Gymnasium` + `MiniGrid`

---

## 📌 Project Overview

This repository implements an autonomous navigation agent trained with Deep Reinforcement Learning. The architecture combines:
1. **Custom 3-Layer Convolutional Neural Network (CNN)** for spatial feature extraction from partial grid observations (`NavCNNExtractor`).
2. **Long Short-Term Memory (LSTM)** recurrent network for temporal memory under partial observability.
3. **Proximal Policy Optimization (PPO)** Actor-Critic algorithm (`RecurrentPPO`).
4. **Curriculum Learning & Reward Shaping** for progressive navigation in complex multi-room and key-door environments.

---

## 📁 Repository Structure

```
.
├── configs/                  # Hyperparameter YAML configuration files
│   ├── baseline_ppo.yaml
│   ├── cnn_ppo.yaml
│   ├── recurrent_ppo.yaml
│   └── curriculum.yaml
├── src/                      # Source modules
│   ├── envs/                 # Environment factory and wrappers
│   │   ├── make_env.py
│   │   ├── wrappers.py       # CollisionCounterWrapper & RewardShapingWrapper
│   │   ├── minigrid_probe.py
│   │   └── test_env.py
│   ├── models/               # Custom neural network feature extractors
│   │   └── cnn_extractor.py  # NavCNNExtractor
│   ├── training/             # Training scripts
│   │   ├── train_baseline.py
│   │   ├── train_cnn_ppo.py
│   │   ├── train_recurrent_ppo.py
│   │   └── train_curriculum.py
│   ├── evaluation/           # Evaluation and visualization scripts
│   │   ├── quick_eval.py
│   │   ├── compare_policies.py
│   │   ├── plot_results.py
│   │   └── render_trajectory_gif.py
│   └── utils/                # Utilities and environment diagnostics
│       └── env_check.py
├── reports/                  # Results, metrics, and markdown tables
│   ├── env_spec.md
│   ├── reward_shaping_notes.md
│   ├── curriculum_results.csv
│   ├── comparison_table.csv
│   └── comparison_table.md
├── media/                    # Rendered GIFs and comparison charts
│   ├── gifs/
│   └── reward_curves/
├── checkpoints/              # Model checkpoints (.zip)
├── PROGRESS.md               # Step-by-step phase execution log
└── requirements.txt          # Pinned dependencies
```

---

## 🚀 Environment Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment setup
python src/utils/env_check.py
python src/envs/minigrid_probe.py
```

---

## 🔬 How to Run Pipeline Phases

### Phase 1: Environment & Wrapper Verification
```bash
python src/envs/test_env.py
```

### Phase 2: Non-Recurrent MLP PPO Baseline
```bash
python src/training/train_baseline.py --config configs/baseline_ppo.yaml
python src/evaluation/quick_eval.py --checkpoint checkpoints/baseline_ppo.zip --env_id MiniGrid-Empty-8x8-v0
```

### Phase 3: Custom CNN Feature Extractor
```bash
python src/training/train_cnn_ppo.py --config configs/cnn_ppo.yaml
python src/evaluation/quick_eval.py --checkpoint checkpoints/cnn_ppo.zip --env_id MiniGrid-Empty-8x8-v0
```

### Phase 4: Recurrent PPO (CNN + LSTM)
```bash
python src/training/train_recurrent_ppo.py --config configs/recurrent_ppo.yaml
python src/evaluation/quick_eval.py --checkpoint checkpoints/recurrent_ppo_empty8x8.zip --env_id MiniGrid-Empty-8x8-v0 --is_recurrent
```

### Phase 5: Curriculum Training
```bash
python src/training/train_curriculum.py --config configs/curriculum.yaml
```

### Phase 7 & 8: Comparative Evaluation & Visualizations
```bash
python src/evaluation/compare_policies.py
python src/evaluation/plot_results.py
python src/evaluation/render_trajectory_gif.py
```

---

## 📊 Comparative Performance Results

Full comparison table generated in `reports/comparison_table.md`:

| Environment | Policy | Evaluation Split | Success Rate (%) | Mean Reward | Mean Steps | Mean Collisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniGrid-Empty-8x8-v0** | Random | Unseen | 25.0% | 0.1073 | 232.6 | 10.8 |
| **MiniGrid-Empty-8x8-v0** | MLP-PPO Baseline | Unseen | 100.0% | 0.9481 | 14.8 | 1.5 |
| **MiniGrid-Empty-8x8-v0** | CNN-PPO Extractor | Unseen | **100.0%** | **0.9610** | **11.1** | **0.0** |
| **MiniGrid-DoorKey-8x8-v0** | MLP-PPO Baseline | Unseen | 10.0% | 0.0473 | 613.5 | 412.7 |

---

## 📝 License
MIT License
