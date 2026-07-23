# Review 2 — Project Presentation Guide & Slide Script

**Project Title:** Autonomous Navigation and Decision Making using Deep Reinforcement Learning (CNN-LSTM-Actor-Critic + PPO)  
**Author:** Kesav  

---

## 📋 Overview of Presentation Flow (10-15 Minutes)

| Slide | Topic | Visual Asset to Display | Key Message / Talking Point |
|-------|-------|-------------------------|-----------------------------|
| 1 | Title & Problem Statement | Main Title Slide | Solving partial observability & sequential navigation in grid worlds using Deep RL |
| 2 | System Architecture | Architecture Diagram | 3-Layer CNN (Perception) ➔ LSTM Memory (State) ➔ PPO Actor-Critic (Control) |
| 3 | Environment & Sub-Goal Mechanics | `MiniGrid-DoorKey-8x8` screenshot | Partial 7x7 vision field; 3-step sequence: Pick Key ➔ Unlock Door ➔ Reach Goal |
| 4 | Training Methodology | WandB / Curriculum curves | Vectorized parallel environments (`n_envs=8`) + Potential-based Reward Shaping |
| 5 | Experimental Results | Performance Comparison Table | **100% Success Rate** on unseen held-out map seeds for CNN-LSTM on DoorKey |
| 6 | Baseline vs CNN vs CNN-LSTM Ablation | Bar Chart (`media/success_rate_comparison.png`) | Non-recurrent models fail completely (0% success) on DoorKey due to partial observability |
| 7 | Visual Trajectory Demonstration | Live Pygame Demo / GIF | Show agent picked key, opened door, and navigated to goal |
| 8 | Generalization to Unseen Seeds | CSV Results | Evaluated on 20 held-out random seeds (1000..1019) zero-shot |
| 9 | Technical Challenges & Solutions | Code snippet (`wrappers.py`) | Handled door toggle exploitation bug & reward scale imbalance |
| 10 | Conclusion & Future Work | Summary Slide | Extensible to continuous control, 3D robotics, and multi-agent systems |

---

## 🗣️ Slide-by-Slide Presentation Script

### Slide 1: Title & Problem Definition
- **Talking Point:** "Good morning/afternoon evaluators. My project is *Autonomous Navigation and Decision Making using Deep Reinforcement Learning*. The core challenge addressed is how an autonomous agent can navigate partially observable environments and solve multi-step sub-goals without pre-built global maps."
- **Key Metric to Highlight:** Partial Observability — the agent only sees a 7x7 local grid in front of it, not the whole map.

### Slide 2: Neural Network Architecture
- **Talking Point:** "To handle partial vision and sequential memory, I built a hybrid CNN-LSTM Actor-Critic architecture using Proximal Policy Optimization (RecurrentPPO)."
- **Breakdown:**
  1. **Spatial Feature Extractor (`NavCNNExtractor`):** 3 convolutional layers processing local grid tensor (7x7x3).
  2. **Temporal Memory (LSTM):** Retains hidden state across timesteps so the agent remembers where it picked up the key and where the door was located.
  3. **Policy & Value Heads (Actor-Critic):** Predicts discrete movement actions (`Turn Left`, `Turn Right`, `Move Forward`, `Pick Up`, `Toggle`).

### Slide 3: Environment Challenges (`MiniGrid-DoorKey-8x8`)
- **Talking Point:** "Standard point navigation is trivial. DoorKey requires sequential decision-making under sparse rewards:
  - Step 1: Search and pick up yellow key.
  - Step 2: Navigate to locked door and execute `Toggle` action.
  - Step 3: Traverse into the inner room to reach the green goal tile."

### Slide 4: Reward Shaping & Training Efficiency
- **Talking Point:** "Sparse rewards mean the agent gets zero feedback until reaching the final goal. To accelerate learning, I implemented a potential-based `RewardShapingWrapper`:
  - **+0.2** one-time bonus for key acquisition.
  - **+0.3** one-time bonus for opening locked door.
  - Potential-based distance progress bonus ($\Delta d$).
  - **-0.0005** step penalty to prevent loop spinning."

### Slide 5 & 6: Experimental Performance & Ablation Results

Show this exact comparison table:

| Environment | Policy | Evaluation Split | Success Rate (%) | Mean Reward | Mean Steps | Mean Collisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniGrid-DoorKey-8x8-v0** | Random Baseline | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 44.8 |
| **MiniGrid-DoorKey-8x8-v0** | MLP-PPO Baseline | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 416.8 |
| **MiniGrid-DoorKey-8x8-v0** | CNN-PPO (Extractor) | Unseen (1000..1019) | 0.0% | 0.0000 | 640.0 | 626.8 |
| **MiniGrid-DoorKey-8x8-v0** | **CNN-LSTM-PPO (Ours)** | **Unseen (1000..1019)** | **100.0%** | **0.9691** | **21.9** | **0.0** |

- **Key Takeaway to Tell Evaluators:** "Notice that both MLP-PPO and CNN-PPO without LSTM achieve **0% success** on DoorKey. Without recurrent temporal memory, non-recurrent agents forget whether they are holding a key or looking for the door. Our CNN-LSTM-PPO model achieves **100% success rate** with **zero collisions** on 20 held-out unseen seed layouts!"

---

## 💻 How to Launch Live Visual Demonstration During Presentation

To run a live interactive visual demo in Pygame during your presentation:

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Run live demonstration on DoorKey (shows key pickup -> unlock -> navigation)
python src/evaluation/visual_demo.py --checkpoint checkpoints/complex_doorkey_agent.zip --env_id MiniGrid-DoorKey-8x8-v0 --is_recurrent --use_shaping

# 2. Run fast optimal demo on Empty-8x8
python src/evaluation/visual_demo.py --checkpoint checkpoints/recurrent_ppo_empty8x8.zip --env_id MiniGrid-Empty-8x8-v0 --is_recurrent
```

---

## ❓ Evaluator Q&A Preparation (Anticipated Questions)

### Q1: "Why did non-recurrent PPO get 0% on DoorKey?"
**Answer:** DoorKey requires temporal memory. When the agent is holding a key, its local 7x7 visual observation field does not explicitly show "holding key" unless it looks at its own state vector. More importantly, when it turns away from the key or door, a feedforward network loses state context. The LSTM hidden state maintains memory across turns.

### Q2: "How did you prevent the agent from cheating/exploiting shaped rewards?"
**Answer:** The shaping bonuses (+0.2 for key, +0.3 for door) were kept small compared to the goal completion reward (~1.0). In early iterations, we observed an action-exploitation bug where the agent continuously toggled empty cells for rewards. We fixed this in `wrappers.py` by requiring state-change validation (checking `door_open_after` vs `door_open_before`).

### Q3: "Did the agent memorize the grid maps?"
**Answer:** No. Evaluation was strictly performed on 20 held-out random seeds (`1000..1019`) that were never seen during training. The 100% success rate on unseen seeds proves zero-shot generalization.

---

## 🏁 Summary Checklist for Review 2
- [x] Codebase fully operational with pinned dependencies in `.venv`
- [x] 100% evaluation success rate verified on held-out test splits
- [x] Comparison table updated in `reports/comparison_table.md` & `reports/comparison_table.csv`
- [x] Smooth live Pygame demo ready (`src/evaluation/visual_demo.py`)
