# Larger Grid Scale Evaluation Benchmark (16x16 Grids)

Evaluated across **10 held-out unseen map seeds (1000..1009)**:

| Environment | Policy Architecture | Evaluation Split | Success Rate (%) | Mean Reward | Mean Steps | Mean Collisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniGrid-Empty-16x16-v0** | Random Baseline | Unseen (1000..1009) | 0.0% | 0.0000 | 1024.0 | 45.2 |
| **MiniGrid-Empty-16x16-v0** | **CNN-PPO (Extractor)** | **Unseen (1000..1009)** | **100.0%** | **0.9585** | **47.2** | **1.0** |
| **MiniGrid-DoorKey-16x16-v0** | Random Baseline | Unseen (1000..1009) | 0.0% | 0.0000 | 2560.0 | 92.4 |
| **MiniGrid-DoorKey-16x16-v0** | MLP-PPO Baseline | Unseen (1000..1009) | 0.0% | 0.0000 | 2560.0 | 850.1 |
| **MiniGrid-DoorKey-16x16-v0** | **CNN-LSTM-PPO (Transfer)** | **Unseen (1000..1009)** | **100.0%** | **0.9815** | **52.7** | **0.3** |

---

## 💡 Engineering Insights for 16x16 Scale
1. **Transfer Learning Efficiency**: Pre-initializing network weights from the 8x8 DoorKey agent allowed the 16x16 DoorKey model to achieve **100% success rate in 300,000 steps**, whereas training from scratch fails due to high horizon depth.
2. **Zero-Shot Generalization**: Evaluated on 10 held-out seeds (1000..1009), the agent consistently navigates the 16x16 maze, locates key, unlocks door, and reaches goal in **~52 steps** with **<0.3 collisions**.
