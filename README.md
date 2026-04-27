# 🐾 Quadruped RL: Decoupled Nominal Walking & Recovery

This repository provides a specialized framework for training and deploying Reinforcement Learning (RL) agents on quadruped robots (Unitree A1/Laikago). It features a **decoupled control architecture** that separates standard locomotion from high-intensity recovery maneuvers.

---

## 🌟 The Core Concept

We decouple the "Nominal Walking" policy from the "Recovery" policy to achieve the best of both worlds:

- **Nominal Agent:** Focused on high-fidelity gait imitation and energy efficiency. Trained for **96M steps** (65M pure imitation + 31M with slight $60\text{N}$ - $100\text{N}$ disturbances).
- **Recovery Agent:** A specialized policy (SAC/PPO) designed for extreme conditions ($100\text{N}$ - $250\text{N}$ forces) where standard walking would fail.

---

## 🛠 Setup & Installation

### 1. Requirements

- **OS:** Developed on **Ubuntu 24.04**; compatible with **macOS** and **Windows**.
- **Hardware:** CPU-only training is sufficient for this repository.
- **Python:** 3.10 or higher recommended.

### 2. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv rlproj
source rlproj/bin/activate  # Windows: rlproj\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Testing & Integration

### 1. Test the Nominal Agent

Verify the baseline gait (96M step model).

```bash
python3 -m motion_imitation.run_test \
  --mode test \
  --motion_file motion_imitation/data/motions/dog_pace.txt \
  --model_file models/final_model.zip \
  --output_dir models \
  --visualize
```

### 2. Test the Recovery Agent

Evaluate the specialized recovery policy in isolation.

```bash
python3 -m motion_imitation.run_sacRecovery \
  --mode test \
  --visualize \
  --num_test_episodes 10 \
  --model_file output_test1/final_model.zip
```

### 3. Full Integrated System

To run the robot with both agents integrated (switching to Recovery mode when the Center of Mass leaves the support polygon):

```bash
python main.py
```

---

## 🏋️ Training Instructions

You can refer to the `main()` functions in each `run_*.py` script for a full list of configurable arguments. For **headless training** (faster), simply remove the `--visualize` flag.

### A. Nominal Walking Policy

**Fine-tune existing model:**

```bash
python3 -m motion_imitation.run_test \
  --mode train \
  --motion_file motion_imitation/data/motions/dog_pace.txt \
  --total_timesteps 5000000 \
  --model_file models/final_model.zip \
  --int_save_freq 125000 \
  --output_dir output_nominal
```

_To train from scratch, do not pass the `--model_file` argument._

**Fine-tune with slight push recovery:**

```bash
python3 -m motion_imitation.run_recovery \
  --mode train \
  --total_timesteps 1000000 \
  --int_save_freq 6250 \
  --output_dir output_pushRecovery \
  --model_file models/final_model.zip
```

### B. SAC Recovery Policy

**Train the specialized high-force recovery agent:**

```bash
python3 -m motion_imitation.run_sacRecovery \
  --mode train \
  --total_timesteps 1000000 \
  --int_save_freq 12500 \
  --output_dir output \
  --logdir output \
  --model_file output_test1/final_model.zip
```
