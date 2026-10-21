# Label-Efficient Facial Expression Recognition

*How Much Labeled Data Do You Really Need? A Comparison of Pretraining Paradigms (Scratch, Supervised, Self-Supervised, and ArcFace) for Facial Emotion Classification on RAF-DB*

---

## 🎯 Research Overview

Facial Expression Recognition (FER) in real-world scenarios suffers from high annotation costs, subjective label ambiguity, and class imbalance. This research benchmark systematically investigates **sample efficiency** across four distinct pretraining paradigms as labeled training data is scaled down from **100% to 1%** on the Real-world Affective Faces Database (**RAF-DB**).

### Core Questions
1. **At what data regime does domain-specific facial pretraining (ArcFace) provide the highest marginal benefit over generic pretraining (ImageNet / DINOv2)?**
2. **Can generic self-supervised foundation models (DINOv2) compete with face-specialized representations when labeled emotion data is scarce ($\le 10\%$)?**
3. **How does pretraining affect calibration error (ECE) and minority class recall (*Fear*, *Disgust*) under severe data scarcity?**

---

## 📊 Dataset: RAF-DB (Real-world Affective Faces Database)

* **Images**: ~15,339 in-the-wild facial images featuring diverse head poses, lighting, age, occlusions, and gender.
* **Task**: 7 Basic Facial Emotion Categories:
  1. `0: Surprise`
  2. `1: Fear` (Minority class, ~2.3%)
  3. `2: Disgust` (Minority class, ~5.8%)
  4. `3: Happiness` (Majority class, ~38.9%)
  5. `4: Sadness` (~16.1%)
  6. `5: Anger` (~5.7%)
  7. `6: Neutral` (~20.9%)
* **Partitioning**:
  * **Master Split**: Official RAF-DB partition (12,271 Train, 3,068 Test) with a stratified 10% validation holdout for model selection.
  * **Data Fractions**: **100%, 50%, 25%, 10%, 5%, 1%** sampled with stratification across **3 seeds (42, 43, 44)**.
  * **Total Experiment Grid**: $4 \text{ models} \times 6 \text{ fractions} \times 3 \text{ seeds} = \mathbf{72 \text{ runs}}$.

---

## 🔬 Evaluated Model Architectures

| Model | Architecture | Pretraining Strategy | Pretraining Source | Parameters | Feature Dimension |
|---|---|---|---|---|---|
| **Model A** | ResNet-50 | Random Initialization (Scratch) | None | ~23.5M | 2048 |
| **Model B** | ResNet-50 | Supervised Pretraining | ImageNet-1k | ~23.5M | 2048 |
| **Model C** | ViT-S/14 | Self-Supervised Vision Transformer | Meta DINOv2 (LVD-142M) | ~22.1M | 384 |
| **Model D** | iResNet-50 | Domain-Specific Face Pretraining | ArcFace (MS1MV2 / Glint360k) | ~31.0M | 512 |

> **Architectural Control**: Models A, B, and D all utilize 50-layer residual backbones, enabling a direct and fair comparison between random initialization, generic ImageNet pretraining, and face-specific ArcFace pretraining.

---

## 📈 Evaluation & Calibration Metrics

Because RAF-DB exhibits notable class imbalance (*Happiness* has ~17× more samples than *Fear*), accuracy alone is misleading. Every experiment is evaluated using:
1. **Balanced Accuracy** (unweighted mean of per-class recall)
2. **Macro F1-Score** (primary checkpoint selection criterion to prevent majority-class bias)
3. **Multiclass Macro AUROC** (One-vs-Rest)
4. **$7 \times 7$ Confusion Matrix**
5. **Per-Class Sensitivity (Recall) & Specificity**
6. **Expected Calibration Error (ECE)** (10-bin reliability diagram binning)

---

## 📁 Repository Structure

```
├── data/
│   ├── splits/
│   │   ├── rafdb_master_split.csv             # Fixed Train / Val / Test partition
│   │   └── fractions/
│   │       ├── seed_42/                       # train_{100,50,25,10,5,1}.csv
│   │       ├── seed_43/
│   │       ├── seed_44/
│   │       └── fraction_sampling_summary.csv  # Sample counts per fraction
├── src/
│   ├── config.py                              # Centralized ExperimentConfig system
│   ├── data/
│   │   ├── dataset.py                         # RAFDBDataset PyTorch class (7 basic emotions)
│   │   └── transforms.py                      # Face-specific augmentations & ArcFace/ImageNet normalization
│   ├── evaluation/
│   │   └── metrics.py                         # Balanced Acc, Macro F1, AUROC, ECE, Conf Matrix
│   ├── models/
│   │   ├── model_a.py                         # Scratch ResNet-50
│   │   ├── model_b.py                         # ImageNet ResNet-50
│   │   ├── model_c.py                         # DINOv2 ViT-S/14
│   │   └── model_d.py                         # ArcFace iResNet-50
│   └── training/
│       └── trainer.py                         # Resumable training loop, checkpointing, skip logic
├── scripts/
│   ├── run_batch_experiments.py               # Sequential batch runner for Kaggle/Colab
│   ├── run_kaggle_pilot.py                    # GPU VRAM & throughput pilot benchmark
│   └── generate_rafdb_fractions.py            # Stratified 6-fraction subset generator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Nilkamal21/label-efficient-skin-lesion-classification.git
cd label-efficient-skin-lesion-classification
pip install -r requirements.txt
```

---

## ⚡ Running on Kaggle / Google Colab

The batch execution script (`scripts/run_batch_experiments.py`) is engineered for long-running GPU sessions:
* **Automatic Resume**: If a Kaggle session disconnects, re-running the command automatically resumes from the last completed epoch via `latest_checkpoint.pt`.
* **Zero Compute Waste**: Completed experiments with valid `metrics.json` markers are automatically skipped.
* **Failure Isolation**: If one run fails, subsequent runs in the batch continue uninterrupted.

### Execution Options:

#### Option A: Run a Predefined Batch (Recommended for Kaggle)
The 72-run grid is divided into 24 sequential batches of 3 runs (one experimental condition across seeds 42, 43, 44):
```bash
# Example: Run Batch 19 (Model D / ArcFace at 100% data across seeds 42, 43, 44)
python scripts/run_batch_experiments.py --batch 19

# Example: Run Batch 20 (Model D / ArcFace at 50% data across seeds 42, 43, 44)
python scripts/run_batch_experiments.py --batch 20
```

#### Option B: Target Specific Experiments by Name
```bash
# Run Model D at 25% data on seed 42
python scripts/run_batch_experiments.py --experiments model_d:25%:42

# Run a sequence of experiments
python scripts/run_batch_experiments.py --experiments model_a:10%:42 model_b:10%:42 model_c:10%:42 model_d:10%:42
```

#### Option C: Target by Model, Fractions, and Seeds
```bash
python scripts/run_batch_experiments.py --model model_d --fractions 50% 25% --seeds 42 43 44
```

---

## 📝 Generated Output Artifacts

Each experiment automatically outputs the following under `experiments/{model}/fraction_{frac}/seed_{seed}/`:

* **`metrics.json`**: Test Balanced Acc, Macro F1, AUROC, ECE, confusion matrix, per-class sensitivity/specificity. *(Feeds Table 1 & Figures 1, 3, 4 in the research paper)*.
* **`history.json`**: Epoch-by-epoch loss and validation metrics. *(Feeds Figure 2: Convergence Curves)*.
* **`predictions.csv`**: Sample-level true label, predicted emotion, and full 7-class probability distributions. *(Feeds statistical significance tests & error analysis)*.
* **`config.json`**: Full hyperparameters and reproducibility metadata.
* **`train.log`**: Detailed timestamped execution log.
* **`best_checkpoint.pt`**: Model weights saved at peak validation Macro F1.
* **`latest_checkpoint.pt`**: Full optimizer, model, and RNG state for interruption recovery.

### Downloading Paper Results
To download all results from Kaggle without downloading heavy model weights:
```bash
zip -r rafdb_paper_results.zip experiments/ -x "*.pt"
```
*(Total size for all 72 completed experiments: under 5 MB).*
