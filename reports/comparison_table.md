# Autonomous Navigation Policy Performance Comparison

| Environment                 | Policy              | Evaluation Split    |   Success Rate (%) |   Mean Reward |   Mean Steps |   Mean Collisions |
|:----------------------------|:--------------------|:--------------------|-------------------:|--------------:|-------------:|------------------:|
| MiniGrid-Empty-8x8-v0       | Random              | Seen (0..19)        |                 25 |        0.111  |        231.6 |               9.7 |
| MiniGrid-Empty-8x8-v0       | MLP-PPO Baseline    | Seen (0..19)        |                100 |        0.9443 |         15.8 |               1.5 |
| MiniGrid-Empty-8x8-v0       | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                  0 |        0      |        256   |               3.3 |
| MiniGrid-Empty-8x8-v0       | Random              | Unseen (1000..1019) |                 25 |        0.1073 |        232.6 |              10.8 |
| MiniGrid-Empty-8x8-v0       | MLP-PPO Baseline    | Unseen (1000..1019) |                100 |        0.9481 |         14.8 |               1.5 |
| MiniGrid-Empty-8x8-v0       | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                  0 |        0      |        256   |               3   |
| MiniGrid-DoorKey-8x8-v0     | Random              | Seen (0..19)        |                  0 |        0      |        640   |              41.8 |
| MiniGrid-DoorKey-8x8-v0     | MLP-PPO Baseline    | Seen (0..19)        |                 15 |        0.062  |        606.5 |             389.6 |
| MiniGrid-DoorKey-8x8-v0     | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                  0 |        0      |        640   |               8.6 |
| MiniGrid-DoorKey-8x8-v0     | Random              | Unseen (1000..1019) |                  0 |        0      |        640   |              44.7 |
| MiniGrid-DoorKey-8x8-v0     | MLP-PPO Baseline    | Unseen (1000..1019) |                 10 |        0.0473 |        613.5 |             412.7 |
| MiniGrid-DoorKey-8x8-v0     | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                  0 |        0      |        640   |              10.1 |
| MiniGrid-MultiRoom-N2-S4-v0 | Random              | Seen (0..19)        |                  5 |        0.0129 |         39.6 |               3.8 |
| MiniGrid-MultiRoom-N2-S4-v0 | MLP-PPO Baseline    | Seen (0..19)        |                  0 |        0      |         40   |              32.3 |
| MiniGrid-MultiRoom-N2-S4-v0 | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                  0 |        0      |         40   |               0.8 |
| MiniGrid-MultiRoom-N2-S4-v0 | Random              | Unseen (1000..1019) |                  0 |        0      |         40   |               3.5 |
| MiniGrid-MultiRoom-N2-S4-v0 | MLP-PPO Baseline    | Unseen (1000..1019) |                  0 |        0      |         40   |              35.2 |
| MiniGrid-MultiRoom-N2-S4-v0 | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                  0 |        0      |         40   |               1   |
| MiniGrid-MultiRoom-N4-S5-v0 | Random              | Seen (0..19)        |                  0 |        0      |        120   |               9.7 |
| MiniGrid-MultiRoom-N4-S5-v0 | MLP-PPO Baseline    | Seen (0..19)        |                  0 |        0      |        120   |              90.7 |
| MiniGrid-MultiRoom-N4-S5-v0 | CNN-LSTM-PPO (Ours) | Seen (0..19)        |                  0 |        0      |        120   |               2   |
| MiniGrid-MultiRoom-N4-S5-v0 | Random              | Unseen (1000..1019) |                  0 |        0      |        120   |              10   |
| MiniGrid-MultiRoom-N4-S5-v0 | MLP-PPO Baseline    | Unseen (1000..1019) |                  0 |        0      |        120   |              92.8 |
| MiniGrid-MultiRoom-N4-S5-v0 | CNN-LSTM-PPO (Ours) | Unseen (1000..1019) |                  0 |        0      |        120   |               2.4 |
