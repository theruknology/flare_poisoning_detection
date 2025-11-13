# FLARE Project - Quick Start Guide

## 5-Minute Setup

### Prerequisites
- Python 3.6+
- CUDA 10.0+ (for GPU, optional but recommended)
- 4GB+ RAM (8GB+ with CUDA)

### Installation

```bash
# Navigate to project
cd /home/theruknology/Desktop/reswork/flare_poisoning_detection

# Install dependencies
pip install tensorflow keras numpy scipy scikit-learn matplotlib

# For GPU support (optional)
# Follow: https://www.tensorflow.org/install/gpu
```

### First Run (CPU - Quick Test)

```bash
# Simple benign training (MNIST, 10 agents, 5 rounds, ~2 min)
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 10 \
  --E 2 \
  --T 5 \
  --B 64 \
  --train
```

**Expected Output:**
```
Time step 0
Set of agents chosen: [0 1 2 3...]
training 10 agents
Agent 0, Step 0, Loss 2.3...
...
test complete
Time step 1
...
```

---

## Complete Experiment Workflows

### Workflow 1: Benign Federated Learning (Baseline)

```bash
# Train benign global model
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  --eta 1e-3 \
  --optimizer adam \
  --train \
  --lr_reduce \
  --gar avg \
  --detect_method none
```

**What happens:**
1. ✅ Loads MNIST (50 benign agents)
2. ✅ Each round: selects 10 agents (C=0.2), trains locally
3. ✅ Aggregates with simple averaging
4. ✅ Evaluates global model on test set
5. ✅ Saves results to `result/MNIST/acc_*.csv`

**Expected metrics:**
- Final accuracy: ~99%
- Convergence: ~15 rounds
- Output: `weights/MNIST/model_0/adam/k50_E5_B64_C2.0e-01_lr1.0e-03/`

---

### Workflow 2: Model Poisoning Attack (No Defense)

```bash
# Train with malicious agents attacking via backdoor
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  --train \
  --mal \
  --mal_obj target_backdoor \
  --mal_strat converge \
  --attacker_num 5 \
  --attack_type backdoor \
  --mal_boost 1.0 \
  --gar avg
```

**What happens:**
1. ✅ Same as benign, but with 5 malicious agents
2. ✅ Malicious agents inject backdoor triggers into training
3. ✅ Global model trained to misclassify backdoored inputs
4. ✅ Logs attack success rate (ASR) per round

**Expected metrics:**
- Clean accuracy: ~98% (slight drop)
- Attack success rate: ~100% (after convergence)
- Result: Backdoor successfully injected into global model

---

### Workflow 3: Defense with Robust Aggregation

```bash
# Train with Byzantine-robust aggregation (Krum)
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  --train \
  --mal \
  --mal_obj target_backdoor \
  --attack_type backdoor \
  --gar krum  # Byzantine-robust
```

**What happens:**
1. ✅ Same attack setup as Workflow 2
2. ✅ But aggregation uses Krum (selects trusted agent)
3. ✅ Krum ignores most distant agents (likely poisoned)
4. ✅ Attack effectiveness significantly reduced

**Expected metrics:**
- Clean accuracy: ~99%
- Attack success rate: ~10-30% (vs 100% with avg)
- Defense effectiveness: 70-90% ASR reduction

**Other robust aggregation options:**
```bash
--gar coomed        # Coordinate-wise median
--gar trimmedmean   # Trimmed mean with 10% removal
--gar bulyan        # Multi-layer robust aggregation
```

---

### Workflow 4: Defense with Anomaly Detection

```bash
# Train with FLARE detection mechanism
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  --train \
  --mal \
  --mal_obj target_backdoor \
  --attack_type backdoor \
  --detect \
  --detect_method detect_penul \
  --aux_data_num 200 \
  --gar avg  # Uses detection instead of robust GAR
```

**What happens:**
1. ✅ Same attack setup
2. ✅ Every round: detection analyzes penultimate layer reps
3. ✅ Calculates trust score for each agent (high=benign, low=malicious)
4. ✅ Weights aggregation by trust scores
5. ✅ Malicious agents get low weight, benign get high weight

**Expected metrics:**
- Clean accuracy: ~99%
- Attack success rate: ~5-20%
- Detection TPR: ~90%+ (identifies malicious agents)

---

### Workflow 5: Comparison (Run Multiple Variants)

```bash
# Baseline
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --mal --mal_obj target_backdoor --attack_type backdoor \
  --gar avg --detect_method none

# Robust Krum
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --mal --mal_obj target_backdoor --attack_type backdoor \
  --gar krum

# Detection
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --mal --mal_obj target_backdoor --attack_type backdoor \
  --detect --detect_method detect_penul --gar avg

# Combination: Detection + Robust Aggregation
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --mal --mal_obj target_backdoor --attack_type backdoor \
  --detect --detect_method detect_penul --gar krum
```

**Analysis:**
- Compare results from `result/MNIST/acc_*.csv` files
- Check attack success rates and detection metrics
- Determine which defense is most effective

---

## Dataset Selection

### MNIST (Recommended for Quick Testing)
```bash
--dataset MNIST
# Training time: ~1-2 hours (50 agents, 30 rounds)
# Memory: ~2 GB
# Accuracy: 99%+
```

### Fashion-MNIST (Similar to MNIST)
```bash
--dataset fMNIST
# Training time: ~1.5-2.5 hours
# Accuracy: ~92%
# More challenging than MNIST
```

### CIFAR-10 (More Complex)
```bash
--dataset CIFAR-10
# Training time: ~3-5 hours
# Memory: ~6 GB
# Accuracy: ~94%
# Larger images (32x32)
```

### Census (Tabular Data)
```bash
--dataset census
# Training time: ~30 min
# Memory: ~1 GB
# Binary classification, feature data (no images)
```

### Kather (Histopathology - Most Complex)
```bash
--dataset kather
# Training time: ~2-3 hours (10 agents)
# Memory: ~8 GB
# 8-class tissue classification
# Requires more computational resources
```

---

## GPU Usage

### Single GPU
```bash
export CUDA_VISIBLE_DEVICES=0
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --gpu_ids 0
```

### Multiple GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python dist_train_w_attack.py --dataset MNIST --k 50 --train \
  --gpu_ids 0 1 2 3
# Distributes agents across 4 GPUs
```

### Check GPU Usage During Training
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

---

## Monitoring Training

### Check Current Progress
```bash
# List output files being written
ls -lh output_files/MNIST/model_0/adam/k50_*/

# View real-time metrics
tail -f output_files/MNIST/model_0/adam/k50_E5_B64_C2.0e-01_lr1.0e-03/output_global_eval_loss.txt

# Check saved weights
ls -lh weights/MNIST/model_0/adam/k50_E5_B64_C2.0e-01_lr1.0e-03/ | head -10
```

### View Results After Training
```bash
# Accuracy metrics (CSV)
cat result/MNIST/acc_*.csv

# Anomaly scores (if detection enabled)
cat result/MNIST/mmd_*.csv

# Plot learning curves
python << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

# Read accuracy file
df = pd.read_csv('result/MNIST/acc_backdoor_detect_penul_avg_200_50_5.csv')
plt.plot(df['eval_success'])
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.savefig('accuracy_plot.png')
EOF
```

---

## Common Errors & Solutions

### Error: GPU Out of Memory
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:**
```bash
# Reduce batch size
--B 32  # instead of 64

# Reduce max agents per GPU
# Edit global_vars.py max_agents_per_gpu

# Or use CPU
export CUDA_VISIBLE_DEVICES=-1
```

### Error: Dataset Not Found
```
FileNotFoundError: data/fashion-mnist/...
```
**Solution:**
```bash
# Auto-download by running any training first
# Or manually download from:
# https://github.com/zalandoresearch/fashion-mnist
```

### Error: Weights File Not Found
```
FileNotFoundError: weights/.../global_weights_t0.npy
```
**Solution:**
```bash
# Run training with --train flag first
# Or check that weights/ directory exists and is writable
mkdir -p weights/
chmod 755 weights/
```

### Error: Too Many Agents
```
cuda:OutOfMemory or system freeze
```
**Solution:**
```bash
# Reduce number of agents
--k 20  # instead of 50

# Reduce agents per GPU
--gpu_ids 0 1  # use multiple GPUs
```

---

## Pre-configured Scripts

### Run Pre-built MNIST Experiments
```bash
# No attack (baseline)
bash script/bash_mnist/mnist_non_attack.sh

# Attack with robust Krum aggregation
bash script/bash_mnist/mnist_bakrum.sh

# Attack with coordinate-wise median
bash script/bash_mnist/mnist_bacoomed.sh

# Attack with multiple detection methods
bash script/bash_mnist/mnist_dba.sh
```

### Run Fashion-MNIST Experiments
```bash
bash script/bash_fmnist/fmnist_non_attack.sh
bash script/bash_fmnist/fmnist_duplicate.sh
```

### Run CIFAR-10 Experiments
```bash
bash script/bash_cifar/cifar_non_attack.sh
bash script/bash_cifar/cifar_duplicate.sh
```

---

## Parameter Tuning Guide

### For Faster Convergence
```bash
--eta 1e-2      # Higher learning rate
--E 10          # More local epochs
--B 32          # Smaller batch (more updates)
```

### For Stronger Attacks
```bash
--mal_boost 10.0     # Amplify attack (5x default)
--mal_E 5            # More training for attackers
--attacker_num 10    # More malicious agents
```

### For Better Defense
```bash
--aux_data_num 500   # More samples for detection
--gar bulyan         # Strongest aggregation rule
--detect_method detect_penul  # Most advanced detection
```

### For Non-IID Data
```bash
--noniid                  # Enable non-IID distribution
--degree_noniid 0.4      # Heterogeneity degree
```

---

## Advanced: Custom Configuration File

Create `my_experiment.json`:
```json
{
  "dataset": "MNIST",
  "image_rows": 28,
  "image_cols": 28,
  "num_channels": 1,
  "num_classes": 10,
  "model_arch": "cnn",
  "optimizer": "adam",
  "eta": 1e-3,
  "k": 50,
  "C": 0.2,
  "E": 5,
  "T": 30,
  "B": 64,
  "target_class": 5,
  "attacker_num": 5,
  "mal_obj": "target_backdoor",
  "mal_boost": 1.0,
  "mal_E": 2,
  "ls": 1,
  "aux_data_num": 200,
  "degree_noniid": 0.4
}
```

Then run (when main.py supports JSON):
```bash
python main.py --json_file my_experiment.json --train
```

---

## Performance Benchmarks

| Scenario | Dataset | Agents | Rounds | Time | Accuracy |
|----------|---------|--------|--------|------|----------|
| Baseline | MNIST | 50 | 30 | 1.5h | 99.0% |
| + Attack | MNIST | 50 | 30 | 1.5h | 98.5% |
| + Krum | MNIST | 50 | 30 | 2.0h | 99.0% |
| + Detection | MNIST | 50 | 30 | 2.5h | 99.0% |
| Baseline | CIFAR-10 | 50 | 30 | 4.0h | 94.0% |
| + Attack | CIFAR-10 | 50 | 30 | 4.0h | 92.0% |
| + Krum | CIFAR-10 | 50 | 30 | 5.0h | 94.0% |

---

## Next Steps

1. **Read the documentation:**
   - `CODEBASE_ANALYSIS.md` - Detailed file-by-file analysis
   - `ARCHITECTURE.md` - System architecture and flows

2. **Run experiments:**
   - Start with baseline (Workflow 1)
   - Progress to attack (Workflow 2)
   - Compare defenses (Workflows 3-4)

3. **Analyze results:**
   - Check `result/MNIST/` for CSV files
   - Plot accuracy/attack success curves
   - Calculate defense effectiveness metrics

4. **Modify for research:**
   - Implement new attack strategies in `malicious_agent.py`
   - Add detection mechanisms in `detect.py`
   - Develop new aggregation rules in `agg_alg.py`

---

## Support & Resources

- **Paper:** "FLARE: Defending Federated Learning against Model Poisoning Attacks via Latent Space Representations" (AsiaCCS 2022)
- **Code:** `/home/theruknology/Desktop/reswork/flare_poisoning_detection/`
- **Main Entry:** `dist_train_w_attack.py` (or look for `main.py`)

---

Last Updated: November 2025
