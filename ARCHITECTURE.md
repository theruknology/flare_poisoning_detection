# FLARE Architecture & Execution Flow

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FEDERATED LEARNING SYSTEM                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ ENTRY POINT: dist_train_w_attack.py::main()                 │   │
│  │ Orchestrates entire federated learning training             │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │                                             │
│         ┌───────────────┼───────────────┐                            │
│         ▼               ▼               ▼                            │
│    ┌─────────────┐ ┌─────────────┐ ┌──────────────┐               │
│    │ global_vars │ │ io_utils    │ │ agents.py    │               │
│    │   .init()   │ │ data_setup()│ │ master()     │               │
│    │             │ │             │ │              │               │
│    │ Argument    │ │ Load dataset│ │ Init global  │               │
│    │ parsing     │ │ Create      │ │ model &      │               │
│    │ Setup paths │ │ mal. data   │ │ weights      │               │
│    └─────────────┘ └─────────────┘ └──────────────┘               │
│         │               │                  │                        │
│         └───────────────┼──────────────────┘                        │
│                         │                                             │
│                         ▼                                             │
│    ┌────────────────────────────────────────────────────┐           │
│    │      MAIN TRAINING LOOP (for t=0 to T-1)          │           │
│    │ train_fn() in dist_train_w_attack.py               │           │
│    └────────────────────────────────────────────────────┘           │
│         │                                                             │
│         │  Each Round t:                                            │
│         │  1. Select agents subset                                  │
│         │  2. Train agents (parallel)                               │
│         │  3. Execute attack (optional)                             │
│         │  4. Detect anomalies (optional)                           │
│         │  5. Aggregate updates                                     │
│         │  6. Evaluate global model                                 │
│         │                                                             │
│         ├─► ┌─────────────────────────────────────────┐             │
│         │   │  AGENT TRAINING (Multiprocessing)       │             │
│         │   ├─────────────────────────────────────────┤             │
│         │   │                                         │             │
│         │   │  For each selected agent i:             │             │
│         │   │  ┌─────────────────────────────────┐   │             │
│         │   │  │ BENIGN AGENTS                   │   │             │
│         │   │  │ agents.py::agent()              │   │             │
│         │   │  │ ├─ Load global_weights_t{t}.npy│   │             │
│         │   │  │ ├─ Local SGD/Adam training      │   │             │
│         │   │  │ ├─ Compute delta               │   │             │
│         │   │  │ ├─ Evaluate on test set        │   │             │
│         │   │  │ └─ Save ben_delta_{i}_t{t}.npy │   │             │
│         │   │  └─────────────────────────────────┘   │             │
│         │   │                                         │             │
│         │   │  ┌─────────────────────────────────┐   │             │
│         │   │  │ MALICIOUS AGENTS                │   │             │
│         │   │  │ malicious_agent.py::mal_agent() │   │             │
│         │   │  │ ├─ benign_train() phase        │   │             │
│         │   │  │ ├─ craft backdoor/trojan      │   │             │
│         │   │  │ ├─ compute attack update      │   │             │
│         │   │  │ └─ save poisoned delta        │   │             │
│         │   │  └─────────────────────────────────┘   │             │
│         │   └─────────────────────────────────────────┘             │
│         │                                                             │
│         ├─► ┌─────────────────────────────────────────┐             │
│         │   │ ATTACK OPTIMIZATION (optional)          │             │
│         │   ├─────────────────────────────────────────┤             │
│         │   │  attack.py                              │             │
│         │   │  ├─ attack_krum()                       │             │
│         │   │  │  └─ Craft updates for Krum GAR      │             │
│         │   │  ├─ attack_trimmedmean()                │             │
│         │   │  │  └─ Craft updates for trimmed mean  │             │
│         │   │  └─ attack.py::attack_*()              │             │
│         │   │     └─ Boost malicious updates         │             │
│         │   └─────────────────────────────────────────┘             │
│         │                                                             │
│         ├─► ┌─────────────────────────────────────────┐             │
│         │   │ DETECTION (optional)                    │             │
│         │   ├─────────────────────────────────────────┤             │
│         │   │  detect.py::Detect.penul_check()        │             │
│         │   │  ├─ Extract penultimate layer reps     │             │
│         │   │  ├─ Compute MMD distances              │             │
│         │   │  │  (uses mmd.py::kernel_mmd)          │             │
│         │   │  ├─ Calculate nearest neighbor counts  │             │
│         │   │  ├─ Convert to trust scores            │             │
│         │   │  └─ Alpha[0..k-1] ∈ [0, 1]            │             │
│         │   └─────────────────────────────────────────┘             │
│         │                                                             │
│         ├─► ┌─────────────────────────────────────────┐             │
│         │   │ AGGREGATION                             │             │
│         │   ├─────────────────────────────────────────┤             │
│         │   │  agg_alg.py                             │             │
│         │   │  ├─ avg_agg() - Simple averaging        │             │
│         │   │  ├─ krum_agg() - Robust (Krum)         │             │
│         │   │  ├─ coomed_agg() - Coordinate median   │             │
│         │   │  ├─ trimmed_mean() - Robust trimming   │             │
│         │   │  ├─ bulyan_agg() - Multi-layer robust  │             │
│         │   │  └─ soft_agg() - Weighted by trust     │             │
│         │   │                                         │             │
│         │   │  Input:  [ben_delta_0, ..., ben_delta_k]             │
│         │   │  Output: global_weights += aggregated  │             │
│         │   │                                         │             │
│         │   │  Save: global_weights_t{t+1}.npy       │             │
│         │   └─────────────────────────────────────────┘             │
│         │                                                             │
│         └─► ┌─────────────────────────────────────────┐             │
│             │ EVALUATION & LOGGING                    │             │
│             ├─────────────────────────────────────────┤             │
│             │  eval_utils.py::eval_func()             │             │
│             │  ├─ Test global model accuracy          │             │
│             │  ├─ Test backdoor success (if attack)   │             │
│             │  ├─ Log metrics to CSV                  │             │
│             │  └─ Save to output_files/ & result/     │             │
│             └─────────────────────────────────────────┘             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────────┐
│  Raw Datasets    │
│ ├─ MNIST         │
│ ├─ Fashion-MNIST │
│ ├─ CIFAR-10      │
│ ├─ Census        │
│ └─ Kather        │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  io_utils.py                         │
│  ├─ data_mnist()                     │
│  ├─ data_cifar()                     │
│  ├─ data_census()                    │
│  └─ data_kather()                    │
│                                      │
│  Output:                             │
│  ├─ X_train, Y_train (normalized)    │
│  └─ X_test, Y_test (normalized)      │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Data Preprocessing                  │
│  ├─ Normalization [0,1] or [-1,1]   │
│  ├─ One-hot encoding                 │
│  ├─ Image resizing (if needed)       │
│  └─ Validation split                 │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Sharding for k Agents               │
│  ├─ Split X_train into k shards     │
│  ├─ Split Y_train into k shards     │
│  └─ Each agent gets shard i         │
└────────┬─────────────────────────────┘
         │
         ├─ Benign Data Shards ─────────────┐
         │                                  │
         │ Each agent i trains on:          │
         │  X_train_shards[i] (1/k of data) │
         │  Y_train_shards[i]               │
         │                                  │
         └──────────────────────────────────┘
         │
         ├─ Malicious Data (if attack)──────┐
         │                                  │
         │ io_utils.py::mal_data_setup()    │
         │  ├─ Select trigger samples      │
         │  ├─ Apply visual triggers       │
         │  │  ├─ Pattern: add pattern     │
         │  │  └─ Semantic: use specific   │
         │  ├─ Set target label            │
         │  └─ Save: mal_data_X, mal_Y     │
         │                                  │
         └──────────────────────────────────┘
         │
         ▼
    [Training Loop]
         │
         ├─ Save Intermediate Files:
         │  ├─ global_weights_t{t}.npy
         │  ├─ ben_delta_{i}_t{t}.npy
         │  └─ (per round)
         │
         ▼
    [Testing/Evaluation]
         │
         ▼
┌──────────────────────────────────────┐
│  Output Artifacts                    │
│  ├─ weights/                         │
│  │  └─ Model snapshots per round     │
│  ├─ output_files/                    │
│  │  └─ Metric CSVs                   │
│  └─ result/                          │
│     └─ Accuracy & anomaly scores     │
└──────────────────────────────────────┘
```

## Aggregation Algorithm Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│                    AGGREGATION ALGORITHMS (GAR)                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INPUT: local_updates = [delta_0, delta_1, ..., delta_{k-1}]         │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ AVERAGE (avg)                                        │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  global_update = (1/k) * Σ delta_i                  │             │
│  │  Byzantine Robustness: NO (vulnerable)              │             │
│  │  Attack Impact: HIGH                                │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ KRUM - Robust Selection (krum)                       │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  1. For each agent i:                               │             │
│  │     compute sum of k-2 nearest distances            │             │
│  │  2. Select agent with minimum sum                   │             │
│  │  global_update = selected_delta                     │             │
│  │  Byzantine Robustness: YES (up to ~10% malicious)  │             │
│  │  Complexity: O(k²)                                  │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ COORDINATE-WISE MEDIAN (coomed)                     │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  For each parameter coordinate:                     │             │
│  │    select median value across agents                │             │
│  │  global_update = [median_0, median_1, ...]         │             │
│  │  Byzantine Robustness: YES (up to 50% malicious)   │             │
│  │  Complexity: O(k log k)                             │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ TRIMMED MEAN (trimmedmean)                           │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  1. For each coordinate:                            │             │
│  │     sort values                                      │             │
│  │     remove top/bottom β% (e.g., 10%)               │             │
│  │     average remaining                               │             │
│  │  Byzantine Robustness: YES (proportional to β)      │             │
│  │  Complexity: O(k log k)                             │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ BULYAN - Multi-Layer Robust (bulyan)                │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  1. Apply sub-algorithm (Krum/Trimmed Mean) θ times │             │
│  │     θ = k - 2f  (f = #malicious agents)            │             │
│  │  2. Aggregate selected updates using another rule   │             │
│  │  Byzantine Robustness: HIGHEST (theoretical f)     │             │
│  │  Complexity: O(θ * k²) or O(θ * k log k)           │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ SOFT AGGREGATION (soft_agg) - With Trust Scores     │             │
│  ├──────────────────────────────────────────────────────┤             │
│  │  Input: alpha = [α_0, α_1, ..., α_{k-1}]           │             │
│  │  (trust scores from detection)                      │             │
│  │  global_update = Σ α_i * delta_i  (weighted sum)   │             │
│  │  Byzantine Robustness: ADAPTIVE (depends on        │             │
│  │                       detection accuracy)          │             │
│  │  Complexity: O(k)                                   │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Detection Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  DETECTION: detect.py::Detect.penul_check()                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Extract Penultimate Layer Representations                 │
│  ─────────────────────────────────────────────                     │
│  For each agent i in current_agents:                               │
│    weights_i = global_weights + delta_i                            │
│    model_i = create_model(weights_i)                               │
│    plr_i = model_i.layer[penul_idx](X_test_subset)                │
│    │                                                                │
│    └─ Dimensions typically 128-512 (latent features)               │
│                                                                     │
│  Output: penul_ls = [plr_0, plr_1, ..., plr_{k-1}]               │
│           (k agents × N samples × D features)                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 2: Compute Pairwise MMD (Maximum Mean Discrepancy)          │
│  ─────────────────────────────────────────────────────             │
│  For each pair (i, j):                                             │
│    mmd_ij = kernel_mmd(plr_i, plr_j)                              │
│    └─ Distance between distributions in latent space               │
│                                                                     │
│  Creates k×k distance matrix:                                      │
│       [  0   mmd01 mmd02 ...]                                      │
│       [mmd10  0   mmd12 ...]                                       │
│       [mmd20 mmd21  0   ...]                                       │
│       [...  ...   ...  ...]                                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 3: Find Nearest Neighbors (50% threshold)                    │
│  ────────────────────────────────────────────────                 │
│  For each agent i:                                                 │
│    neighbors = k nearest agents to i (by MMD distance)             │
│    count_i = how many times agent i appears in others' top 50%    │
│                                                                     │
│  Benign agents ─ highly similar to others ─ high count             │
│  Malicious agents ─ different distribution ─ low count             │
│                                                                     │
│  Output: count = [c_0, c_1, ..., c_{k-1}]                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 4: Convert Count to Trust Scores                             │
│  ───────────────────────────────────────                           │
│  Method: Exponential weighting                                     │
│                                                                     │
│  count_avg = average(count)                                        │
│  exp_i = exp(count_i / (tau * count_avg))  [tau=1]                │
│  alpha_i = exp_i / Σ exp_j   (softmax normalization)               │
│                                                                     │
│  Benign agents  ─► high count  ─► high exp_i  ─► α_i ≈ 0.2        │
│  Malicious agents ─► low count  ─► low exp_i  ─► α_i ≈ 0.01      │
│                                                                     │
│  Output: alpha = [α_0, α_1, ..., α_{k-1}]                        │
│          Σ α_i = 1.0 (probability distribution)                    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 5: Weighted Aggregation with Trust Scores                    │
│  ───────────────────────────────────────────────────              │
│  Use soft_agg() with alpha:                                        │
│    global_update = Σ α_i * delta_i                                │
│                                                                     │
│  Effect:                                                           │
│    - Benign agents contribute ~20% each                            │
│    - Malicious agents contribute ~1% each                          │
│    - Attack impact reduced significantly                           │
│                                                                     │
│  Output: updated global_weights                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Attack Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ATTACK SCENARIOS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Attack Type: TARGETED BACKDOOR                                    │
│  ─────────────────────────────────────                             │
│                                                                     │
│  Goal: Misclassify specific inputs to a target label               │
│                                                                     │
│  Example:                                                          │
│    Input: Picture of traffic sign (original: "stop")               │
│    Trigger: Add small pattern in corner                            │
│    Output: Model misclassifies as "yield" (target)                 │
│                                                                     │
│  Malicious Agent Strategy:                                         │
│    1. benign_train() - Train honestly first                        │
│    2. mal_train() - Craft trigger+target pair                      │
│    3. Train on backdoored data                                     │
│    4. Send poisoned update to server                               │
│                                                                     │
│  Attack Success Metric:                                            │
│    - Target confidence: P(target_label | trigger)                  │
│    - Goal: target_confidence → 1.0                                 │
│                                                                     │
│  Parameters:                                                       │
│    --mal_obj target_backdoor                                       │
│    --attack_type backdoor                                          │
│    --mal_boost 1.0-10.0 (amplification)                            │
│                                                                     │
│  ───────────────────────────────────────────────────────────────   │
│                                                                     │
│  Attack Type: UNTARGETED (Model Corruption)                        │
│  ─────────────────────────────────────────────                     │
│                                                                     │
│  Goal: Prevent global model from converging / reduce accuracy      │
│                                                                     │
│  Example:                                                          │
│    Send random/adversarial updates that degrade model              │
│    Result: Global accuracy never exceeds baseline                  │
│                                                                     │
│  Parameters:                                                       │
│    --attack_type untargeted_krum or untargeted_trimmedmean        │
│    --mal_strat converge or dist                                    │
│                                                                     │
│  ───────────────────────────────────────────────────────────────   │
│                                                                     │
│  Attack Type: DATA POISONING                                       │
│  ─────────────────────────────                                     │
│                                                                     │
│  Goal: Inject malicious training data into agent shards            │
│                                                                     │
│  Techniques:                                                       │
│    - Label flipping: Change correct labels to wrong ones           │
│    - Feature injection: Add specific patterns to features          │
│    - Trigger patterns: Subtle visual modifications                 │
│                                                                     │
│  Parameters:                                                       │
│    --mal_strat data_poison                                         │
│    --data_rep 10 (repetitions of poisoned samples)                 │
│    --trojan semantic or trojan (trigger type)                      │
│                                                                     │
│  ───────────────────────────────────────────────────────────────   │
│                                                                     │
│  Attack Type: DISTANCE-CONSTRAINED (dist)                          │
│  ────────────────────────────────────────                          │
│                                                                     │
│  Goal: Craft attacks that stay within a distance constraint        │
│        (makes attack less detectable)                              │
│                                                                     │
│  Technique:                                                        │
│    - Compute benign update norm                                    │
│    - Craft malicious update within rho * ||benign||                │
│    - More subtle but still effective                               │
│                                                                     │
│  Parameters:                                                       │
│    --mal_strat dist                                                │
│    --rho 1e-4 (distance weight, smaller = more subtle)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Metrics & Evaluation

```
┌─────────────────────────────────────────────────────────────────────┐
│                  EVALUATION METRICS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BENIGN MODEL METRICS                                              │
│  ────────────────────────                                          │
│  - Accuracy: % correct predictions on clean test set               │
│  - Loss: Cross-entropy loss on test set                            │
│  - Convergence: How fast accuracy reaches target (e.g., 99%)       │
│                                                                     │
│  ATTACK SUCCESS METRICS                                            │
│  ────────────────────────                                          │
│  For backdoor attacks:                                             │
│    - Attack Success Rate (ASR): % of backdoored samples            │
│      correctly classified to target                                │
│    - Target Confidence: Average confidence on target label         │
│    - Clean Accuracy Maintenance: Keep normal accuracy high         │
│                                                                     │
│  For untargeted attacks:                                           │
│    - Accuracy Degradation: Drop in final model accuracy            │
│    - Convergence Time: How long until reaching plateau             │
│    - Stability: Variance in accuracy across rounds                 │
│                                                                     │
│  DETECTION METRICS                                                 │
│  ─────────────────                                                 │
│  - True Positive Rate (TPR): % malicious correctly identified      │
│  - False Positive Rate (FPR): % benign incorrectly flagged         │
│  - Detection Accuracy: (TP + TN) / (TP + TN + FP + FN)            │
│  - Precision: TP / (TP + FP)                                       │
│  - Recall: TP / (TP + FN)                                          │
│                                                                     │
│  DEFENSE EFFECTIVENESS                                             │
│  ──────────────────────                                            │
│  - Attack Success with Defense vs. without                         │
│  - Robust Aggregation Impact: ASR reduction (%)                    │
│  - Detection + Aggregation Combo Benefit                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## File I/O Structure

```
PROJECT_ROOT/
│
├─ Input (data/)
│  ├─ mnist/ (auto-downloaded if missing)
│  ├─ fashion-mnist/ (auto-downloaded)
│  ├─ cifar-10/ (auto-downloaded)
│  ├─ census/ (user-provided)
│  ├─ kather/ (user-provided)
│  ├─ mal_X_*.npy (malicious samples)
│  ├─ mal_Y_*.npy (malicious labels)
│  └─ true_labels_*.npy
│
├─ Configuration (script/)
│  ├─ bash_mnist/
│  │  ├─ *.sh (bash scripts)
│  │  └─ mnist.json (parameter config)
│  ├─ bash_fmnist/
│  ├─ bash_cifar/
│  └─ [similar structure]
│
├─ Intermediate (weights/)
│  └─ {dataset}/model_{num}/{optimizer}/k{k}_E{E}_B{B}_C{C}_lr{eta}/
│     ├─ global_weights_t0.npy (initial)
│     ├─ global_weights_t1.npy
│     │ ... (all T rounds)
│     ├─ ben_delta_0_t0.npy (agent 0, round 0)
│     ├─ ben_delta_1_t0.npy
│     │ ... (all agents, all rounds)
│     └─ [repeated for each attack variant]
│
├─ Logs (output_files/)
│  └─ {dataset}/model_{num}/{optimizer}/k{k}_..._lr{eta}/
│     ├─ output_*.txt (per-round metrics)
│     └─ [one per experiment variant]
│
├─ Results (result/)
│  └─ {dataset}/
│     ├─ acc_*_*_*_*.csv (accuracy per round)
│     ├─ mmd_*_*_*_*.csv (anomaly scores)
│     └─ [multiple files for different experiments]
│
└─ Figures (figures/)
   └─ {dataset}/model_{num}/{optimizer}/k{k}_..._lr{eta}/
      ├─ accuracy_plot.png
      ├─ loss_plot.png
      ├─ backdoor_success.png
      └─ [visualizations]
```

## Parameter Dependencies & Valid Combinations

```
┌─────────────────────────────────────────────────────────────────────┐
│  PARAMETER COMBINATION RULES                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CONDITION 1: --mal flag                                           │
│  ─────────────────────────                                         │
│  If --mal is True:                                                 │
│    • --attack_type must be set (not "none")                        │
│    • --mal_obj must match attack (e.g., "target_backdoor" for     │
│      backdoor attack, or "single"/"multiple" for targeted)        │
│    • --mal_strat defines strategy (converge, dist, data_poison)   │
│    • --attacker_num specifies # malicious agents (1 to k)         │
│    • --mal_data_num or implicit from mal_data_setup()             │
│                                                                     │
│  CONDITION 2: --detect flag                                        │
│  ─────────────────────────                                         │
│  If --detect is True:                                              │
│    • --detect_method must be specified                            │
│    • Detection works best with --gar avg (to contrast)            │
│    • --aux_data_num: size of test subset for detection            │
│    • Disables/overrides --gar when detection used                 │
│                                                                     │
│  CONDITION 3: Attack optimization for specific GAR                │
│  ──────────────────────────────────────────────────               │
│  IF --attack_type = "backdoor_krum":                              │
│    • Use attack.attack_krum() to optimize for Krum GAR            │
│    • --gar krum recommended                                        │
│                                                                     │
│  IF --attack_type = "untargeted_trimmedmean":                     │
│    • Use attack.attack_trimmedmean() to optimize                  │
│    • --gar trimmedmean recommended                                │
│                                                                     │
│  CONDITION 4: Malicious agent parameters                          │
│  ──────────────────────────────────────                           │
│  --mal_boost: multiplicative factor (1.0-10.0)                    │
│    • Only used if --mal_strat != "data_poison"                    │
│    • Higher values = stronger attack (easier to detect)           │
│                                                                     │
│  --rho: distance constraint (used with --mal_strat dist)          │
│    • Only applies if "dist" in --mal_strat                        │
│    • Range: 1e-5 to 1e-2                                          │
│    • Smaller = more subtle attack                                 │
│                                                                     │
│  --mal_E: malicious agent epochs (≥ --E)                          │
│    • If > --E: malicious agents train longer                      │
│    • Gives them more computation budget                           │
│                                                                     │
│  CONDITION 5: Non-IID data distribution                           │
│  ─────────────────────────────────────────                        │
│  --noniid flag:                                                    │
│    • Activates non-IID data distribution across agents            │
│    • Can affect attack/defense effectiveness                      │
│    • Requires --degree_noniid parameter                           │
│                                                                     │
│  CONDITION 6: Learning rate reduction                             │
│  ──────────────────────────────────────                           │
│  --lr_reduce flag:                                                │
│    • Enables learning rate decay over rounds                      │
│    • Typically applied in production settings                     │
│    • dir_name changes to indicate this variant                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Commands

### Minimal Setup (CPU, quick test)
```bash
python dist_train_w_attack.py \
  --dataset MNIST --k 10 --E 2 --T 5 --train
# Runs in ~1-2 minutes on CPU
```

### Standard Experiment (GPU, 50 agents, 30 rounds)
```bash
export CUDA_VISIBLE_DEVICES=0
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --lr_reduce --gpu_ids 0
# Runs in ~1-2 hours on single GPU
```

### Full Attack + Defense Experiment
```bash
export CUDA_VISIBLE_DEVICES=0,1
python dist_train_w_attack.py \
  --dataset MNIST --k 50 --C 0.2 --E 5 --T 30 \
  --train --mal --attacker_num 5 --attack_type backdoor \
  --mal_obj target_backdoor \
  --detect --detect_method detect_penul \
  --gar avg --gpu_ids 0 1
```

---

Last Updated: November 2025
