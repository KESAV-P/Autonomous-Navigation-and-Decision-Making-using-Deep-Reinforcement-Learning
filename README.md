# Autonomous Navigation using Deep Reinforcement Learning (CNN-LSTM-Actor-Critic + PPO)

**Author:** Kesav  
**Framework:** `stable-baselines3` + `sb3-contrib` (`RecurrentPPO`)  
**Environment:** `Gymnasium` + `MiniGrid`  

---

## 📌 Project Overview

This repository implements an autonomous navigation agent trained with Deep Reinforcement Learning. The architecture combines:
1. **Custom 3-Layer Convolutional Neural Network (CNN)** for spatial feature extraction from partial grid observations (`NavCNNExtractor`).
2. **Long Short-Term Memory (LSTM)** recurrent network for temporal memory under partial observability.
3. **Proximal Policy Optimization (PPO)** Actor-Critic algorithm (`RecurrentPPO`).
4. **Curriculum Learning & Potential-Based Reward Shaping** for progressive navigation in complex multi-room and key-door environments.

---

## 📁 Repository Structure

```
.
├── configs/                  # Hyperparameter YAML configuration files
│   ├── baseline_ppo.yaml
│   ├── cnn_ppo.yaml
│   ├── recurrent_ppo.yaml
│   ├── doorkey_16x16_training.yaml
│   └── curriculum.yaml
├── src/                      # Source modules
│   ├── envs/                 # Environment factory and wrappers
│   │   ├── make_env.py
│   │   ├── wrappers.py       # CollisionCounterWrapper & RewardShapingWrapper
│   │   └── minigrid_probe.py
│   ├── models/               # Custom neural network feature extractors
│   │   └── cnn_extractor.py  # NavCNNExtractor
│   ├── training/             # Training scripts
│   │   ├── train_baseline.py
│   │   ├── train_cnn_ppo.py
│   │   ├── train_recurrent_ppo.py
│   │   ├── train_complex_agent.py
│   │   └── train_curriculum.py
│   ├── evaluation/           # Evaluation and visualization scripts
│   │   ├── quick_eval.py
│   │   ├── compare_policies.py
│   │   ├── visual_demo.py     # Live Pygame window demo
│   │   ├── plot_results.py
│   │   └── render_trajectory_gif.py
│   └── utils/                # Utilities and environment diagnostics
│       └── env_check.py
├── reports/                  # Results, metrics, evaluation tables
│   ├── env_spec.md
│   ├── reward_shaping_notes.md
│   ├── larger_grid_eval.md
│   ├── comparison_table.csv
│   └── comparison_table.md
├── media/                    # Rendered GIFs and comparison charts
│   ├── gifs/
│   └── reward_curves/
├── checkpoints/              # Saved model checkpoints (.zip)
├── PROGRESS.md               # Step-by-step phase execution log
└── requirements.txt          # Pinned dependencies
```

---

## 🚀 Quick Start & Environment Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run live Pygame window visual navigation demo
python src/evaluation/visual_demo.py --checkpoint checkpoints/complex_doorkey_agent.zip --env_id MiniGrid-DoorKey-8x8-v0 --is_recurrent --use_shaping

# 3. Evaluate checkpoint on 20 held-out unseen map seeds (1000..1019)
python src/evaluation/quick_eval.py --checkpoint checkpoints/complex_doorkey_agent.zip --env_id MiniGrid-DoorKey-8x8-v0 --is_recurrent --n_episodes 20 --start_seed 1000
```

---

## 📊 Review 2 Experimental Evaluation Results

Evaluated across **20 held-out unseen map seeds (1000..1019)** never encountered during training:

| Environment | Policy Architecture | Evaluation Split | Success Rate (%) | Mean Reward | Mean Steps | Mean Collisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniGrid-Empty-8x8-v0** | Random Baseline | Unseen (1000..1019) | 25.0% | 0.0846 | 239.0 | 9.5 |
| **MiniGrid-Empty-8x8-v0** | MLP-PPO Baseline | Unseen (1000..1019) | 100.0% | 0.9508 | 14.0 | 1.2 |
| **MiniGrid-Empty-8x8-v0** | CNN-PPO Extractor | Unseen (1000..1019) | 100.0% | 0.9612 | 11.0 | 0.0 |
| **MiniGrid-Empty-8x8-v0** | **CNN-LSTM-PPO (Ours)** | Unseen (1000..1019) | **95.0%** | **0.7635** | **65.8** | **38.3** |
| **MiniGrid-DoorKey-8x8-v0** | Random Baseline | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 44.8 |
| **MiniGrid-DoorKey-8x8-v0** | MLP-PPO Baseline | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 416.8 |
| **MiniGrid-DoorKey-8x8-v0** | CNN-PPO Extractor | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 626.8 |
| **MiniGrid-DoorKey-8x8-v0** | **CNN-LSTM-PPO (Ours)** | **Unseen (1000..1019)** | **100.0%** | **0.9691** | **21.9** | **0.0** |

---

## 📝 License
MIT License
