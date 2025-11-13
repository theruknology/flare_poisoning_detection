# FLARE Project - Cheat Sheet & Reference Card

## File Dependency Map

```
dist_train_w_attack.py (MAIN)
  ├─ global_vars.py
  │   └─ argparse, tensorflow
  ├─ io_utils.py
  │   ├─ mnist.py / cifar_utils.py / census_utils.py / kather_utils.py
  │   └─ numpy, keras
  ├─ agents.py
  │   ├─ [model files]
  │   └─ eval_utils.py
  ├─ malicious_agent.py
  │   ├─ [model files]
  │   └─ eval_utils.py
  ├─ agg_alg.py
  │   ├─ dist_utils.py
  │   └─ numpy
  ├─ detect.py
  │   ├─ mmd.py
  │   ├─ sklearn
  │   └─ eval_utils.py
  ├─ attack.py
  │   ├─ agg_alg.py
  │   └─ eval_utils.py
  └─ eval_utils.py
      ├─ [model files]
      └─ io_utils.py
```

---

## Command Cheat Sheet

### One-Liners by Scenario

```bash
# 1. QUICK TEST (CPU, ~2 min)
python dist_train_w_attack.py --dataset MNIST --k 10 --E 2 --T 5 --train

# 2. BENIGN BASELINE (1.5 hours)
python dist_train_w_attack.py --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 --train --lr_reduce --gar avg

# 3. ATTACK (NO DEFENSE)
python dist_train_w_attack.py --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 --train --mal --mal_obj target_backdoor --attack_type backdoor --gar avg

# 4. ROBUST KRUM
python dist_train_w_attack.py --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 --train --mal --attack_type backdoor --gar krum

# 5. WITH DETECTION
python dist_train_w_attack.py --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 --train --mal --attack_type backdoor --detect --detect_method detect_penul

# 6. STRONG DEFENSE (BULYAN + DETECTION)
python dist_train_w_attack.py --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 --train --mal --attack_type backdoor --gar bulyan --detect --detect_method detect_penul

# 7. CIFAR-10 WITH ATTACK
python dist_train_w_attack.py --dataset CIFAR-10 --k 50 --E 5 --T 30 --train --mal --attack_type backdoor --gpu_ids 0 1

# 8. MULTIPLE GPU SCALING
export CUDA_VISIBLE_DEVICES=0,1,2,3
python dist_train_w_attack.py --dataset MNIST --k 100 --train --gpu_ids 0 1 2 3
```

---

## Parameter Quick Reference

### Performance Tuning
```
FASTER CONVERGENCE:
  --eta 1e-2                    # ↑ learning rate
  --E 10                        # ↑ local epochs
  --B 32                        # ↓ batch size (more updates)
  --C 1.0                       # Use all agents each round

SLOWER BUT MORE STABLE:
  --eta 1e-4                    # ↓ learning rate
  --E 2                         # ↓ local epochs
  --B 128                       # ↑ batch size
  --C 0.1                       # Select 10% of agents
```

### Attack Tuning
```
STRONGER ATTACK:
  --mal_boost 10.0              # 10x amplification
  --attacker_num 10             # More attackers
  --mal_E 10                    # More training for attackers
  --mal_strat dist              # Distance-constrained (harder to detect)
  --rho 1e-3                    # Larger distance bound

WEAKER ATTACK (FOR TESTING):
  --mal_boost 1.0               # No amplification
  --attacker_num 1              # Single attacker
  --mal_E 2                     # Same as benign
  --mal_strat converge          # Simple strategy
```

### Defense Tuning
```
STRONGEST DEFENSE:
  --gar bulyan                  # Strongest aggregation
  --detect                      # Enable detection
  --detect_method detect_penul  # Best detection
  --aux_data_num 500            # More detection data

WEAKEST DEFENSE (BASELINE):
  --gar avg                     # Vulnerable aggregation
  --detect_method none          # No detection
  --aux_data_num 200            # Minimum detection data
```

---

## Output Interpretation Guide

### File Locations
```
weights/MNIST/model_0/adam/k50_E5_B64_C2.0e-01_lr1.0e-03/
  ├─ global_weights_t0.npy          # Initial model
  ├─ global_weights_t1.npy          # After round 1
  └─ ben_delta_0_t0.npy             # Agent 0's update, round 0

result/MNIST/
  ├─ acc_backdoor_detect_penul_*.csv
  └─ mmd_backdoor_detect_penul_*.csv

output_files/MNIST/model_0/adam/.../
  ├─ output_global_eval_loss.txt
  └─ output_global_eval_accuracy.txt
```

### Result CSV Format
```
# acc_*.csv
eval_success,eval_loss,mal_suc_count
0.9500,0.2340,0
0.9520,0.2100,0
0.9580,0.1890,10        # Attack succeeds at round 3
0.9600,0.1750,80
...

# mmd_*.csv (if detection enabled)
agent_id,trust_score,neighbor_count,is_detected
0,0.198,15,0
1,0.201,14,0
2,0.195,16,0
3,0.015,1,1              # Malicious agent detected
4,0.012,0,1
...
```

---

## Expected Accuracy Ranges

### MNIST Baseline
```
Epoch 0-5:   ~80-90% (random exploration)
Epoch 5-15:  ~95-99% (convergence)
Epoch 15-30: ~99.2-99.5% (plateau)
```

### MNIST With Backdoor (Averaging)
```
Epoch 0-10:  ~98-99% (benign phase)
Epoch 10-20: ~97-98% (attack weakens)
Epoch 20-30: ~98-99% (plateau with backdoor)
Attack Success Rate: 100% (backdoor works)
```

### MNIST With Backdoor + Krum
```
Epoch 0-30:  ~99-99.5% (attack blocked)
Attack Success Rate: 10-30% (mostly blocked)
```

### MNIST With Backdoor + Detection
```
Epoch 0-30:  ~99-99.5% (attack blocked)
Attack Success Rate: 5-15% (mostly blocked)
Detection TPR: 90%+ (detects malicious)
```

---

## Dataset Characteristics

### MNIST
```
Size: 60k train, 10k test
Dimensions: 28×28×1 (grayscale)
Classes: 10
Model: Simple CNN
Baseline Accuracy: 99%+
Training Time (50 agents): 1.5 hours
Recommended for: Quick testing, understanding mechanisms
```

### Fashion-MNIST
```
Size: 60k train, 10k test
Dimensions: 28×28×1 (grayscale)
Classes: 10
Model: Simple CNN
Baseline Accuracy: 92-93%
Training Time: 1.5-2.5 hours
Recommended for: Comparative studies
```

### CIFAR-10
```
Size: 50k train, 10k test
Dimensions: 32×32×3 (RGB)
Classes: 10
Model: ResNet-like CNN
Baseline Accuracy: 94%+
Training Time (50 agents): 4-5 hours
Recommended for: Complex scenarios, requires GPU
Memory: ~6GB
```

### Census
```
Size: ~32k samples
Dimensions: 105 features (tabular)
Classes: 2 (binary)
Model: MLP/Dense network
Baseline Accuracy: 85%
Training Time: 30 minutes
Recommended for: Non-image learning, heterogeneous data
Memory: ~1GB
```

### Kather (Histopathology)
```
Size: 5k+ images
Dimensions: 128×128×3 (RGB)
Classes: 8
Model: CNN
Baseline Accuracy: 85%+
Training Time (10 agents): 2-3 hours
Recommended for: Medical imaging, complex scenarios
Memory: ~8GB
Requires: Multiple GPU recommended
```

---

## Aggregation Rules Comparison Matrix

| Rule | Robustness | Speed | Complexity | Use Case |
|------|-----------|-------|-----------|----------|
| **avg** | ✗ Low | ✓✓✓ O(k) | Simple | Baseline only |
| **krum** | ✓✓ Medium | ✓✓ O(k²) | Moderate | General-purpose |
| **coomed** | ✓✓ Medium | ✓✓ O(k log k) | Moderate | Image learning |
| **trimmedmean** | ✓✓ Medium | ✓✓ O(k log k) | Moderate | Any |
| **bulyan** | ✓✓✓ High | ✓ O(θ·k²) | Complex | Maximum safety |
| **soft_agg** | ✓✓✓ High* | ✓ O(k) | Simple | With detection |

*With good detection

---

## Attack Success Metrics Interpretation

### Attack Success Rate (ASR)
```
ASR = 100% ──► Attack fully successful, global model compromised
ASR = 50% ───► Attack partially successful, backdoor inconsistent
ASR = 0% ────► Attack completely blocked, defense effective
```

### Clean Accuracy
```
Clean Acc = 99% ──► Normal performance maintained
Clean Acc = 95% ──► Degraded due to attack/defense tradeoff
Clean Acc = 80% ──► Severe degradation, defense too aggressive
```

### Combined Interpretation
```
Clean 99% + ASR 100% ──► Successful backdoor (bad)
Clean 99% + ASR 10% ───► Good defense, attack mostly blocked
Clean 95% + ASR 0% ────► Perfect defense, acceptable tradeoff
```

---

## GPU Memory Estimation

### Per Agent (Approximate)
```
MNIST:      50-200 MB
CIFAR-10:   200-400 MB
Kather:     400-800 MB
Census:     10-50 MB
```

### Per GPU (50 agents, k GPUs)
```
1 GPU:  Total = 50 × agent_mem
2 GPUs: Total = 25 × agent_mem per GPU
4 GPUs: Total = 12.5 × agent_mem per GPU
8 GPUs: Total = 6.25 × agent_mem per GPU
```

### Examples
```
MNIST 50 agents:
  1 GPU:  ~2.5-10 GB
  2 GPUs: ~1.25-5 GB each
  4 GPUs: ~0.6-2.5 GB each ← Recommended

Kather 10 agents:
  1 GPU:  ~4-8 GB
  2 GPUs: ~2-4 GB each ← Recommended
```

---

## Troubleshooting Decision Tree

```
Error?
  │
  ├─ OOM (Out of Memory)
  │   ├─ Reduce --B (batch size)
  │   ├─ Reduce --k (agents)
  │   └─ Use more --gpu_ids
  │
  ├─ FileNotFoundError: data/
  │   ├─ Run once to auto-download
  │   └─ Check data/ directory exists
  │
  ├─ FileNotFoundError: weights/
  │   ├─ Create: mkdir -p weights/
  │   └─ Run with --train flag first
  │
  ├─ Process hanging
  │   ├─ Kill: pkill -f dist_train
  │   └─ Check GPU with nvidia-smi
  │
  ├─ Slow training
  │   ├─ Use GPU: export CUDA_VISIBLE_DEVICES=0
  │   ├─ Reduce --E (epochs)
  │   └─ Reduce --k (agents)
  │
  └─ Results look wrong
      ├─ Check accuracy increasing over time
      ├─ Verify --train flag set
      └─ Check result/ CSV files
```

---

## Key Hyperparameters to Vary

### For Empirical Evaluation
```
VARY THIS          EFFECT
--k               # agents → parallelism/robustness
--C               # fraction → convergence speed
--E               # epochs → per-client computation
--B               # batch → gradient noise
--eta             # LR → convergence behavior
--attacker_num    # attackers → attack difficulty
--mal_boost       # amplification → attack strength
--gar             # aggregation → robustness level
--aux_data_num    # detection data → detection TPR
```

### For Paper Results
```
Vary --k (10, 20, 50, 100)
  + Different --gar rules
  + With/without --mal
  + With/without --detect
  = Multiple grid points for comparison tables
```

---

## Visualization Ideas

### Plot 1: Accuracy Over Time
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result/MNIST/acc_backdoor_detect_penul_avg_*.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['eval_success'], label='Test Accuracy', linewidth=2)
plt.axhline(y=0.99, color='r', linestyle='--', label='Target Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy.png')
```

### Plot 2: Attack Success Rate
```python
plt.figure(figsize=(10, 6))
for method in ['avg', 'krum', 'median', 'detection']:
    # Load and plot ASR for method
plt.xlabel('Round')
plt.ylabel('Attack Success Rate (%)')
plt.legend()
plt.grid(True)
plt.savefig('attack_success.png')
```

### Plot 3: Defense Comparison
```python
methods = ['Baseline\n(Avg)', 'Krum', 'Median', 'Detection', 'Bulyan']
asr = [100, 25, 20, 10, 5]
acc = [99, 99, 99, 99, 98]
# Create bar chart comparing defense effectiveness
```

---

## Quick Formulas

### Average Aggregation
```
w_global ← w_global + (1/k) * Σ delta_i
```

### Krum
```
score_i ← Σ ||delta_i - delta_j|| (k-2 smallest)
select_i ← argmin(score_i)
w_global ← w_global + delta_select
```

### Trimmed Mean
```
For each coordinate j:
  values_j ← [delta_0[j], delta_1[j], ..., delta_k-1[j]]
  sorted_j ← sort(values_j)
  trimmed_j ← remove top/bottom β% of sorted_j
  mean_j ← average(trimmed_j)
```

### MMD Distance
```
MMD(X, Y) = ||μ(X) - μ(Y)||² in feature space
Used to detect distribution differences
```

### Trust Score
```
count_i ← how many times agent i in others' top 50%
alpha_i ← exp(count_i / τ) / Σ exp(count_j)
w_global ← Σ alpha_i * delta_i (soft aggregation)
```

---

## Citation

```bibtex
@inproceedings{flare2022,
  title={FLARE: Defending Federated Learning against Model Poisoning 
         Attacks via Latent Space Representations},
  booktitle={17th ACM ASIA Conference on Computer and Communications Security},
  year={2022},
  note={Available at: github.com/theruknology/flare_poisoning_detection}
}
```

---

## Useful Links & Resources

- **Paper:** AsiaCCS 2022 proceedings
- **Federated Learning:** https://github.com/google/federated
- **Byzantine Aggregation:** Krum, Bulyan papers
- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/

---

**Created:** November 2025  
**For:** FLARE Poisoning Detection Project  
**Last Updated:** November 11, 2025
