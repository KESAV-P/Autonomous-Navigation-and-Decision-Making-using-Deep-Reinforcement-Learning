# Autonomous Navigation Policy Performance Comparison (Review 2)

> CNN-LSTM uses per-environment best checkpoints.

| Environment             | Policy              | Evaluation Split    |   Success Rate (%) |   Mean Reward |   Mean Steps |   Mean Collisions |
|:------------------------|:--------------------|:--------------------|-------------------:|--------------:|-------------:|------------------:|
| MiniGrid-Empty-8x8-v0   | Random Baseline     | Seen (0..19)        |                  5 |     0.0341797 |       247.7  |             11.65 |
| MiniGrid-Empty-8x8-v0   | MLP-PPO Baseline    | Seen (0..19)        |                100 |     0.944277  |        15.85 |              1.5  |
| MiniGrid-Empty-8x8-v0   | CNN-PPO (Extractor) | Seen (0..19)        |                100 |     0.960625  |        11.2  |              0.1  |
| MiniGrid-Empty-8x8-v0   | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                100 |     0.753027  |        70.25 |             41.8  |
| MiniGrid-Empty-8x8-v0   | Random Baseline     | Unseen (1000..1019) |                 25 |     0.0845898 |       239.05 |              9.55 |
| MiniGrid-Empty-8x8-v0   | MLP-PPO Baseline    | Unseen (1000..1019) |                100 |     0.950781  |        14    |              1.2  |
| MiniGrid-Empty-8x8-v0   | CNN-PPO (Extractor) | Unseen (1000..1019) |                100 |     0.961152  |        11.05 |              0.05 |
| MiniGrid-Empty-8x8-v0   | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                 95 |     0.763496  |        65.85 |             38.35 |
| MiniGrid-DoorKey-8x8-v0 | Random Baseline     | Seen (0..19)        |                  0 |     0         |       640    |             40.8  |
| MiniGrid-DoorKey-8x8-v0 | MLP-PPO Baseline    | Seen (0..19)        |                 10 |     0.0785547 |       591.25 |            409.9  |
| MiniGrid-DoorKey-8x8-v0 | CNN-PPO (Extractor) | Seen (0..19)        |                  0 |     0         |       640    |            621.85 |
| MiniGrid-DoorKey-8x8-v0 | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                100 |     0.969555  |        21.65 |              0.1  |
| MiniGrid-DoorKey-8x8-v0 | Random Baseline     | Unseen (1000..1019) |                  0 |     0         |       640    |             44.85 |
| MiniGrid-DoorKey-8x8-v0 | MLP-PPO Baseline    | Unseen (1000..1019) |                  0 |     0         |       640    |            416.8  |
| MiniGrid-DoorKey-8x8-v0 | CNN-PPO (Extractor) | Unseen (1000..1019) |                  0 |     0         |       640    |            626.85 |
| MiniGrid-DoorKey-8x8-v0 | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                100 |     0.969133  |        21.95 |              0    |
