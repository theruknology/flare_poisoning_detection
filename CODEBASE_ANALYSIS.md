# FLARE Poisoning Detection - Comprehensive Codebase Analysis

## Project Overview

**Project Name:** detect_model_poisoning_fl  
**Framework:** Federated Learning (FL)  
**Paper:** FLARE: Defending Federated Learning against Model Poisoning Attacks via Latent Space Representations  
**Conference:** 17th ACM ASIA Conference on Computer and Communications Security (AsiaCCS 2022)

This repository implements a federated learning framework with defenses against model poisoning attacks. It supports multiple attack types and defense mechanisms using various aggregation rules and detection methods.

---

## File-by-File Analysis

### Root-Level Python Files

#### 1. `dist_train_w_attack.py` (Main Entry Point)
**Purpose:** Core orchestrator for federated training with model poisoning attacks and defenses

**Key Functions:**
- `train_fn()` - Main training loop that orchestrates:
  - Agent training across multiple rounds
  - Gradient aggregation with multiple rules (avg, krum, coordinated median, trimmed mean, bulyan)
  - Attack execution (untargeted or targeted backdoor)
  - Detection mechanisms
  - Global weight aggregation and updates

- `main()` - Initialization and control flow:
  - Data loading and sharding
  - Master model initialization
  - Training/testing mode selection
  - Malicious data setup

**Dependencies:** 
- `global_vars.py`, `agents.py`, `malicious_agent.py`
- `agg_alg.py` (aggregation algorithms)
- `detect.py` (detection mechanisms)
- `attack.py` (attack algorithms)
- Utilities: `io_utils.py`, `eval_utils.py`
- TensorFlow 1.x compatibility

**Entry Point:** Yes - `if __name__ == "__main__"`

---

#### 2. `global_vars.py`
**Purpose:** Global configuration and variable initialization

**Key Functions:**
- `init()` - Parses command-line arguments and initializes global variables
- `dir_name_fn()` - Constructs dynamic directory paths based on configuration

**Arguments Supported:**
- `--dataset` - Dataset selection: 'kather', 'MNIST', 'fMNIST', 'census', 'CIFAR-10'
- `--model_num` - Model architecture variant
- `--optimizer` - 'adam' or 'sgd'
- `--eta` - Learning rate (default: 1e-3)
- `--k` - Number of agents/clients (default: 10)
- `--C` - Fraction of agents per round (default: 1.0)
- `--E` - Local epochs per agent (default: 5)
- `--T` - Maximum training rounds (default: 40)
- `--B` - Batch size (default: 50)
- `--train` - Enable training mode
- `--mal` - Enable malicious agents
- `--detect` - Enable detection mechanisms
- `--attack_type` - Attack type: 'backdoor_krum', 'backdoor_coomed', 'untargeted_krum', etc.
- `--detect_method` - Detection method: 'detect_acc', 'detect_penul', 'detect_fltrust'
- `--mal_strat` - Malicious strategy: 'converge', 'dist', 'data_poison'
- `--gar` - Gradient aggregation rule: 'avg', 'krum', 'coomed', 'trimmedmean', 'bulyan'
- `--aux_data_num` - Number of auxiliary data points (default: 200)
- `--attacker_num` - Number of malicious agents (default: 1)

**Global Variables Set:**
- Image dimensions, number of classes, batch sizes (dataset-specific)
- GPU configuration, memory fractions
- Output directory paths for weights, figures, results

---

#### 3. `agents.py`
**Purpose:** Implements benign agent training logic

**Key Functions:**
- `agent()` - Trains a benign agent locally:
  - Loads global weights
  - Performs SGD/Adam optimization for specified epochs
  - Computes local weight deltas
  - Evaluates on test set
  - Saves model updates to disk

- `agent_copy()` - Alternative training variant with random sampling

- `train_clean_model()` - Trains a clean reference model (used for comparison)

- `master()` - Initializes global model and saves initial weights

**Dependencies:** 
- Model definitions from `utils/`
- `eval_utils.py` for evaluation

**Architecture:**
- Uses TensorFlow 1.x placeholders
- Builds models based on dataset type (MNIST, Census, Kather, CIFAR)
- Supports batch-based training with configurable epochs

---

#### 4. `malicious_agent.py` (Long file: 861 lines)
**Purpose:** Implements malicious agent attacks on FL systems

**Key Functions:**
- `benign_train()` - Initial benign training phase of malicious agent
- Backdoor attacks: Various methods to craft poisoned updates
- Data poisoning: Modifies training data to inject triggers/patterns
- Attack optimization: Computes optimal attack parameters

**Attack Strategies:**
- **Targeted Backdoor:** Tries to misclassify specific inputs to a target label
- **Data Poisoning:** Corrupts training data with triggers/patterns
- **Distance-constrained attacks:** Attacks respecting certain distance constraints

**Key Parameters:**
- `mal_strat` - Strategy: 'converge', 'dist' (distance-constrained), 'data_poison'
- `mal_boost` - Amplification factor for attack strength
- `mal_E` - Number of epochs for malicious training
- `rho` - Distance constraint weight

---

#### 5. `agg_alg.py`
**Purpose:** Implements gradient aggregation algorithms (Byzantine-robust and standard)

**Key Aggregation Methods:**

- **`avg_agg()`** - Simple averaging (vulnerable to attacks)
- **`krum_agg()`** - Krum algorithm: selects agent with minimum sum of distances to nearest neighbors
- **`coomed_agg()`** - Coordinate-wise median: robust aggregation
- **`trimmed_mean()`** - Removes extreme values before averaging
- **`bulyan_agg()`** - Multi-layer robust aggregation combining krum and trimmed mean
- **`soft_agg()`** - Weighted aggregation using trust scores

**Helper Functions:**
- `collate_weights()` - Flattens model weights and biases
- `model_shape_size()` - Tracks layer shapes for reconstruction
- `krum_one_layer()` - Krum algorithm for single layer

---

#### 6. `detect.py` (426 lines)
**Purpose:** Implements detection mechanisms for poisoned updates

**Key Class:**
- `Detect` - Detection engine with methods:
  - `penul_check()` - Analyzes penultimate layer representations
  - `eval_plr_per_class()` - Extracts penultimate layer representations
  - `cal_nearest_neighbor()` - Calculates anomaly scores using MMD
  - `count_to_trustscore()` - Converts anomaly counts to trust scores

**Detection Methods:**
- Uses Maximum Mean Discrepancy (MMD) to identify outliers
- Compares latent space representations across agents
- Generates trust scores for each agent (higher = more trustworthy)

**Integration with Aggregation:**
- Trust scores used with `soft_agg()` for weighted model updates
- Anomalous agents get lower weights in final aggregation

---

#### 7. `attack.py`
**Purpose:** Implements attack algorithms that target specific aggregation rules

**Key Attack Functions:**

- **`attack_trimmedmean()`** - Crafts attacks optimized for trimmed mean aggregation
  - Estimates benign update statistics
  - Creates perturbations that survive trimming
  - Uses boosting factor to amplify attack strength

- **`attack_krum()`** - Crafts attacks optimized for Krum aggregation
  - Binary search to find optimal attack parameters (lambda)
  - Makes malicious updates appear as trusted benign updates
  - Layer-by-layer optimization

- **`attack_krum_idx()`** - Parameter-level Krum attack

- **`eval_mal()`** - Evaluates attack success metrics

**Attack Metrics:**
- Target confidence: Probability of misclassification to target
- Actual confidence: Probability of current incorrect prediction
- Success rate: Percentage of malicious objectives achieved

---

#### 8. `yolo_demo.py`
**Purpose:** Standalone demo for trust score calculation using YOLO framework

**Key Functions:**
- `cal_nearest_neighbor()` - MMD-based anomaly scoring
- `penul_check()` - Detection using penultimate layer representations

**Note:** This appears to be a separate integration point for YOLO object detection models.

---

### Utility Modules (`utils/` directory)

#### `io_utils.py`
**Purpose:** Data I/O and dataset loading

**Key Functions:**
- `data_setup()` - Loads and preprocesses datasets (MNIST, Fashion-MNIST, CIFAR-10, Census, Kather)
- `mal_data_setup()` - Creates malicious backdoor/trojan data
- `file_write()` - CSV logging of metrics
- `poison()` - Applies visual triggers (patterns, semantic features)

**Supported Datasets:**
- MNIST/Fashion-MNIST: 28x28 grayscale images, 10 classes
- CIFAR-10: 32x32 RGB images, 10 classes
- Census: Tabular data, 105 features, 2 classes
- Kather (Histopathology): 128x128 RGB images, 8 tissue types

---

#### `eval_utils.py` (280 lines)
**Purpose:** Model evaluation utilities

**Key Functions:**
- `eval_setup()` - Initializes TensorFlow session and loads model
- `eval_minimal()` - Tests model on data, returns accuracy and loss
- `mal_eval_single()` - Evaluates single backdoor target
- `mal_eval_multiple()` - Evaluates multiple simultaneous targets
- `eval_func()` - Multiprocess-safe evaluation wrapper
- `eval_plr_per_class()` - Extracts penultimate layer representations

**Penultimate Layer Indices (for each dataset):**
- MNIST/CIFAR-10: Layer 7
- Census: Layer 3
- Kather: Layer 11

---

#### `mnist.py` (256 lines)
**Purpose:** MNIST/Fashion-MNIST specific model definitions and data loading

**Key Functions:**
- `data_mnist()` - Loads and normalizes MNIST/Fashion-MNIST data
- `model_mnist()` - Defines CNN architecture (variants available)

**Model Architecture:**
```
Conv2D(64, 5x5) -> ReLU -> Conv2D(64, 5x5) -> ReLU -> 
Dropout(0.25) -> Flatten -> Dense(128) -> ReLU -> 
Dropout(0.5) -> Dense(10)
```

---

#### `cifar_utils.py`
**Purpose:** CIFAR-10 specific utilities

**Key Functions:**
- `data_cifar()` - CIFAR-10 data loading and preprocessing
- `model_cifar()` - ResNet-style CNN architecture

---

#### `census_utils.py`
**Purpose:** Census dataset utilities for tabular learning

**Key Functions:**
- `data_census()` - Census adult dataset loading
- `census_model_1()` - Fully connected neural network for tabular data

**Architecture:**
- Multi-layer dense network with dropout
- Input dimension: 105, Output: 2 classes

---

#### `kather_utils.py`
**Purpose:** Histopathology (Kather) dataset utilities

**Key Functions:**
- `data_kather()` - Loads tissue image classification data
- `model_kather()` - CNN for 8-class tissue classification

---

#### `dist_utils.py` (207 lines)
**Purpose:** Distance and weight manipulation utilities

**Key Functions:**
- `collate_weights()` - Flattens model weights/biases into vectors
- `model_shape_size()` - Tracks model layer shapes
- `est_accuracy()` - Estimates convergence from weight deltas
- `weight_constrain()` - L2 distance constraint enforcement
- `plr_constrain()` - Penultimate layer representation constraint

---

#### `mmd.py`
**Purpose:** Maximum Mean Discrepancy statistical test

**Key Functions:**
- `kernel_mmd()` - Computes MMD between two distributions
- `MMD2u()` - Unbiased MMD statistic
- `compute_null_distribution()` - Bootstrap null distribution

**Use Case:** Detects distribution differences in latent representations

---

#### `l2dist_calc.py`
**Purpose:** L2 distance calculations for constraint-based attacks

---

#### `mmd_test.py`
**Purpose:** Testing utilities for MMD kernel functions

---

#### `fmnist.py`
**Purpose:** Fashion-MNIST specific data loader

---

#### `image_utils.py`
**Purpose:** Image processing and augmentation utilities

---

## Project Architecture Summary

### Hierarchical Structure

```
flare_poisoning_detection/
│
├── CORE TRAINING ORCHESTRATION
│   └── dist_train_w_attack.py (main entry point, FL training loop)
│
├── CONFIGURATION & GLOBALS
│   └── global_vars.py (argument parsing, path setup)
│
├── AGENTS
│   ├── agents.py (benign agent training)
│   └── malicious_agent.py (attack implementations)
│
├── AGGREGATION ALGORITHMS
│   └── agg_alg.py (avg, krum, median, trimmed mean, bulyan)
│
├── DEFENSE MECHANISMS
│   └── detect.py (anomaly detection, trust scoring)
│
├── ATTACKS
│   └── attack.py (GAR-specific attack optimization)
│
├── UTILITIES (utils/)
│   ├── Data I/O
│   │   └── io_utils.py (dataset loading, malicious data creation)
│   │
│   ├── Model Definitions
│   │   ├── mnist.py (CNN for MNIST/Fashion-MNIST)
│   │   ├── cifar_utils.py (CNN for CIFAR-10)
│   │   ├── census_utils.py (MLP for tabular data)
│   │   └── kather_utils.py (CNN for histopathology)
│   │
│   ├── Evaluation
│   │   └── eval_utils.py (model testing, backdoor evaluation)
│   │
│   ├── Analysis
│   │   ├── dist_utils.py (weight distance calculations)
│   │   ├── mmd.py (statistical distribution testing)
│   │   └── l2dist_calc.py (constraint calculations)
│   │
│   ├── Other
│   │   ├── fmnist.py (Fashion-MNIST loader)
│   │   ├── image_utils.py (image processing)
│   │   ├── mmd_test.py (MMD testing)
│   │   └── __init__.py
│
├── STANDALONE DEMO
│   └── yolo_demo.py (YOLO integration example)
│
├── CONFIGURATION EXAMPLES
│   └── script/
│       ├── bash_mnist/ (MNIST experiment scripts)
│       ├── bash_fmnist/ (Fashion-MNIST experiment scripts)
│       └── bash_cifar/ (CIFAR-10 experiment scripts)
│
└── DATA & OUTPUTS
    ├── data/ (datasets, preprocessed)
    ├── weights/ (saved model weights per iteration)
    ├── output_files/ (training logs, metrics)
    ├── figures/ (visualization outputs)
    └── result/ (experimental results, CSVs)
```

### Control Flow Diagram

```
MAIN (dist_train_w_attack.py)
  ↓
1. Data Setup (io_utils.py)
   - Load and split datasets
   - Create malicious data if needed
  ↓
2. Initialize Master Model (agents.py)
   - Create global model
   - Save initial weights
  ↓
3. TRAINING LOOP (for t = 0 to T-1):
   ├─ Select subset of agents
   ├─ AGENT TRAINING (parallel):
   │  ├─ Benign agents: agent() [agents.py]
   │  └─ Malicious agents: mal_agent() [malicious_agent.py]
   ├─ ATTACK (if malicious):
   │  └─ attack_krum() or attack_trimmedmean() [attack.py]
   ├─ DETECTION (if enabled):
   │  └─ Detect.penul_check() [detect.py]
   │     Uses trust scores with soft_agg()
   ├─ AGGREGATION:
   │  ├─ avg_agg() [agg_alg.py]
   │  ├─ krum_agg() [agg_alg.py]
   │  ├─ coomed_agg() [agg_alg.py]
   │  ├─ trimmed_mean() [agg_alg.py]
   │  └─ bulyan_agg() [agg_alg.py]
   ├─ Update Global Model
   ├─ EVALUATION:
   │  └─ eval_func() [eval_utils.py]
   │     Computes accuracy, loss, attack success
   └─ Log Results to CSV

4. Final Results (result/)
   - Accuracy curves
   - Attack success rates
   - Detection metrics
```

### Data Flow

```
INPUT DATASETS (data/)
  ↓
io_utils.py::data_setup()
  ├─ data_mnist()
  ├─ data_cifar()
  ├─ data_census()
  └─ data_kather()
  ↓
PREPROCESSING
  - Normalization
  - One-hot encoding
  - Resizing (if needed)
  ↓
SHARDING & DISTRIBUTION
  - Split to k agents
  - Create train/test splits
  ↓
TRAINING PROCESS
  - Each agent: local updates on shard
  - Save: ben_delta_t{t}.npy, ben_delta_{i}_t{t}.npy
  - Load: global_weights_t{t}.npy
  ↓
OUTPUT ARTIFACTS (weights/)
  - global_weights_t{t}.npy (after aggregation round t)
  - ben_delta_{i}_t{t}.npy (agent i's update at round t)
  - global_weights_t{t}.npy (initial: t=0)
  ↓
EVALUATION & LOGGING (output_files/, result/)
  - Accuracy per round
  - Loss per round
  - Attack success metrics (target_conf, success rate)
  - Detection metrics (trust scores)
```

---

## Environment and Dependencies

### Required Python Version
- Python 3.6+ (uses TensorFlow 1.x compatibility mode)

### External Dependencies

```
tensorflow==1.x (using tf.compat.v1)
keras
numpy
scipy
scikit-learn
matplotlib
```

### Optional Dependencies
- CUDA toolkit (for GPU acceleration)
- cuDNN (for GPU acceleration)

### Installation

**Note:** No `requirements.txt` found in repository. Install manually:

```bash
# Core dependencies
pip install tensorflow==1.15  # or 2.x with compat.v1
pip install keras numpy scipy scikit-learn matplotlib

# Optional GPU support (CUDA 10.0)
# Follow TensorFlow GPU installation guide

# Development (optional)
pip install jupyter ipython
```

### GPU Configuration

The project uses TensorFlow GPU memory management:
- Memory fraction set per dataset:
  - MNIST/Fashion-MNIST: 0.08
  - Census: 0.05
  - Kather: 0.24
  - CIFAR-10: 0.24

- Multiple GPUs supported via `--gpu_ids` parameter
- Agents distributed across GPUs to maximize parallelization

---

## Execution Guide

### Basic Setup

```bash
# Clone and navigate
cd /home/theruknology/Desktop/reswork/flare_poisoning_detection

# Set CUDA devices if using GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Running Training

#### 1. **No Attack (Benign Training)**
```bash
python dist_train_w_attack.py \
  --dataset MNIST \
  --model_num 0 \
  --k 10 \
  --C 1.0 \
  --E 5 \
  --T 30 \
  --B 64 \
  --eta 1e-3 \
  --optimizer adam \
  --train \
  --detect_method none \
  --gar avg
```

#### 2. **Targeted Backdoor Attack**
```bash
python dist_train_w_attack.py \
  --dataset MNIST \
  --model_num 0 \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  --train \
  --mal \
  --mal_obj target_backdoor \
  --mal_strat converge \
  --mal_boost 1.0 \
  --attacker_num 5 \
  --attack_type backdoor_krum \
  --gar avg
```

#### 3. **With Defense (Detection)**
```bash
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --train \
  --mal \
  --attack_type backdoor \
  --detect \
  --detect_method detect_penul \
  --gar avg
```

#### 4. **Robust Aggregation (Byzantine-Robust)**
```bash
python dist_train_w_attack.py \
  --dataset MNIST \
  --k 50 \
  --train \
  --gar krum  # or coomed, trimmedmean, bulyan
```

### Pre-configured Scripts

Located in `script/bash_{dataset}/`:

```bash
# Fashion-MNIST experiments
bash script/bash_fmnist/fmnist_non_attack.sh

# MNIST experiments
bash script/bash_mnist/mnist_non_attack.sh
bash script/bash_mnist/mnist_bacoomed.sh
bash script/bash_mnist/mnist_bakrum.sh
bash script/bash_mnist/mnist_dba.sh

# CIFAR-10 experiments
bash script/bash_cifar/cifar_non_attack.sh
bash script/bash_cifar/cifar_duplicate.sh
```

### Testing Trained Models

```bash
python dist_train_w_attack.py \
  --dataset MNIST \
  --model_num 0 \
  --k 50 \
  --C 0.2 \
  --E 5 \
  --T 30 \
  --B 64 \
  # (no --train flag, load existing weights from weights/ directory)
```

---

## Key Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | 'kather' | Dataset: MNIST, fMNIST, CIFAR-10, census, kather |
| `--k` | int | 10 | Number of agents/clients |
| `--C` | float | 1.0 | Fraction of agents per round |
| `--E` | int | 5 | Local training epochs per agent |
| `--T` | int | 40 | Maximum training rounds |
| `--B` | int | 50 | Batch size |
| `--eta` | float | 1e-3 | Learning rate |
| `--optimizer` | str | 'adam' | Optimizer: adam or sgd |
| `--mal` | flag | False | Enable malicious agents |
| `--mal_strat` | str | 'converge' | Attack strategy |
| `--mal_boost` | float | 10.0 | Attack amplification factor |
| `--attacker_num` | int | 1 | Number of malicious agents |
| `--gar` | str | 'krum' | Aggregation rule: avg, krum, coomed, trimmedmean, bulyan |
| `--detect` | flag | False | Enable detection mechanism |
| `--detect_method` | str | 'none' | Detection: detect_acc, detect_penul, detect_fltrust |
| `--train` | flag | False | Enable training mode |

---

## Output Artifacts

### Directory Structure After Training

```
weights/{dataset}/model_{num}/{optimizer}/k{k}_E{E}_B{B}_C{C}_lr{eta}[_mal_{attack}_{strategy}]/
  ├── global_weights_t0.npy  (initial weights)
  ├── global_weights_t1.npy
  ├── global_weights_t2.npy
  │ ... (one per round)
  ├── ben_delta_0_t0.npy     (agent 0's update at round 0)
  ├── ben_delta_1_t0.npy
  │ ... (all agents' updates)
  
output_files/{dataset}/model_{num}/{optimizer}/k{k}_..._C{C}_lr{eta}/
  ├── output_global_eval_loss.txt
  ├── output_global_eval_accuracy.txt
  
result/{dataset}/
  ├── acc_*_*_*_*_*_*.csv    (accuracy metrics)
  ├── mmd_*_*_*_*_*_*.csv    (anomaly scores)

figures/{dataset}/model_{num}/{optimizer}/k{k}_..._C{C}_lr{eta}/
  ├── plots and visualizations
```

### Output File Formats

**Metric CSVs (result/):**
- Columns: eval_success, eval_loss per round (or attack-specific metrics)
- Rows: One per training round

**Weight Files (weights/):**
- NumPy `.npy` format
- Serialized Keras model weights as list/tuple

---

## Common Experimental Scenarios

### Scenario 1: Baseline (Benign Federated Learning)
```bash
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --detect_method none --gar avg
```
**Expected:** Model converges to ~99% accuracy

---

### Scenario 2: Backdoor Attack on Average Aggregation
```bash
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --mal --attacker_num 5 --attack_type backdoor \
  --mal_obj target_backdoor --gar avg
```
**Expected:** Attack succeeds after convergence phase, backdoor accuracy → 100%

---

### Scenario 3: Defense with Robust Aggregation
```bash
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --mal --attack_type backdoor \
  --gar krum  # or coomed, trimmedmean, bulyan
```
**Expected:** Attack effectiveness reduced, normal accuracy maintained

---

### Scenario 4: Defense with Detection
```bash
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --mal --attack_type backdoor \
  --detect --detect_method detect_penul \
  --gar avg
```
**Expected:** Detection identifies malicious agents, applies low trust scores, mitigates attack

---

## Troubleshooting

### Issue: GPU Memory Error
**Solution:**
```bash
# Reduce batch size or max agents per GPU
python dist_train_w_attack.py ... --B 32 --gpu_ids 0 1
```

### Issue: Data Loading Error
**Solution:** Ensure data files are present in `data/` directory. First run may download automatically from Keras.

### Issue: Processes Not Terminating
**Solution:** Check `ps aux | grep python` and manually kill hanging processes:
```bash
pkill -f "dist_train_w_attack.py"
```

### Issue: Weights Directory Errors
**Solution:** Weights directories are auto-created. If permission denied:
```bash
chmod -R 755 weights/
```

---

## Performance Characteristics

### Timing Estimates (on GPU)
- **MNIST (50 agents, 30 rounds):** ~1-2 hours
- **Fashion-MNIST (50 agents, 30 rounds):** ~1.5-2.5 hours
- **CIFAR-10 (50 agents, 30 rounds):** ~3-5 hours
- **Kather (10 agents, 30 rounds):** ~2-3 hours

### Scalability Notes
- Linear scaling with number of agents (up to GPU memory limit)
- Each agent consumes ~0.2-0.6 GB VRAM depending on dataset
- Detection adds ~5-10% overhead

---

## Research Notes

### Key Concepts

1. **Model Poisoning:** Malicious agents craft updates to degrade or hijack the global model
2. **Backdoor Attacks:** Inject trojans to misclassify specific trigger patterns
3. **Byzantine Aggregation:** Aggregation rules robust to a fraction of malicious updates
4. **Latent Space Defense:** FLARE uses penultimate layer representations for anomaly detection
5. **Trust Scoring:** Assigns lower weights to suspected malicious agents

### Supported Attack Types
- `backdoor_krum` - Optimized attack for Krum aggregation
- `backdoor_coomed` - Optimized for coordinate-wise median
- `untargeted_krum` - Prevents model convergence
- `untargeted_trimmedmean` - Prevents convergence for trimmed mean
- `backdoor` - Generic backdoor without specific aggregation optimization

### Defense Mechanisms
- **Robust Aggregation:** Byzantine-robust aggregation rules reduce attack impact
- **Anomaly Detection:** Detects outlier updates using statistical tests (MMD)
- **Trust Scoring:** Weights agent contributions by trustworthiness
- **Latent Representation Analysis:** Analyzes penultimate layer features for consistency

---

## Future Work & Extensions

Potential improvements based on code structure:
1. Support for differential privacy
2. Gradient compression techniques
3. Personalized federated learning
4. More detection mechanisms (e.g., clustering in latent space)
5. Adaptive learning rates per agent
6. Asynchronous aggregation

---

## Citation

If using this code, cite the AsiaCCS 2022 paper:
```
FLARE: Defending Federated Learning against Model Poisoning Attacks 
via Latent Space Representations
(Authors and details in README.md)
```

---

## License & Ownership

See repository LICENSE file and README.md for details.

Last Updated: November 2025  
Repository: flare_poisoning_detection
