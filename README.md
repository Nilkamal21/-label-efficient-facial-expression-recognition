# Label-Efficient Skin Lesion Classification

*How Much Labeled Data Do You Really Need? A Comparison of Pretraining and Fine-Tuning Strategies for Skin Lesion Classification*

This research benchmark systematically compares supervised, self-supervised, and biomedical vision-language pretraining strategies across varying data regimes (1% to 100% labeled training data) on the HAM10000 skin lesion dataset.

---

## 🎯 Research Overview

- **Core Question**: How do different pretraining paradigms (scratch, generic supervised, generic self-supervised, domain-specific vision-language) perform as labeled training data is systematically scaled down from 100% to 1%?
- **Dataset**: HAM10000 (10,015 dermatoscopic images across 7 diagnostic categories).
- **Zero-Leakage Split**: Group-aware split on `lesion_id` (70% Train, 15% Val, 15% Test) ensuring no patient lesions overlap between splits.
- **Data Regimes**: 8 training fractions (100%, 75%, 50%, 30%, 20%, 10%, 5%, 1%) sampled deterministically across 3 random seeds (42, 43, 44), yielding a 96-experiment grid.

---

## 🔬 Model Architectures

| Model | Architecture | Pretraining Strategy | Pretraining Source | Parameters | Feature Dim |
|---|---|---|---|---|---|
| **Model A** | EfficientNet-B0 | Random Initialization (Scratch) | None | ~4.016M | 1280 |
| **Model B** | EfficientNet-B0 | Supervised | ImageNet-1k | ~4.016M | 1280 |
| **Model C** | ViT-S/14 | Self-Supervised (DINOv2) | LVD-142M | ~22.06M | 384 |
| **Model D** | ViT-B/16 | Contrastive Vision-Language | BiomedCLIP (PMC-15M) | ~86.20M | 512 |

---

## 📊 Evaluation & Metrics

To account for class imbalance (dominant class `nv` comprises ~67% of samples), models are evaluated using:
1. **Balanced Accuracy** (unweighted mean of per-class recall)
2. **Macro F1-Score** (primary checkpoint selection criterion)
3. **Macro AUROC** (One-vs-Rest multiclass area under ROC curve)
4. **Expected Calibration Error (ECE)** (10-bin reliability diagram analysis)
5. **Per-Class Sensitivity & Specificity**

---

## 📁 Repository Structure

```
├── data/
│   ├── HAM10000_metadata.csv        # Full dataset metadata (10,015 rows)
│   ├── sample_images/               # 10 local verification images
│   ├── splits/                      # Master split & group-aware fraction CSVs
│   │   ├── ham10000_master_split.csv
│   │   └── fractions/
│   │       ├── seed_42/             # train_{100,75,50,30,20,10,5,1}.csv
│   │       ├── seed_43/
│   │       └── seed_44/
│   └── verification_reports/        # Complete audit reports for Stages 1-8
├── src/
│   ├── config.py                    # Centralized ExperimentConfig system
│   ├── data/
│   │   ├── dataset.py               # HAM10000Dataset PyTorch dataset class
│   │   └── transforms.py            # Preprocessing & normalization pipelines
│   ├── evaluation/
│   │   └── metrics.py               # Comprehensive evaluation metrics & ECE
│   ├── models/
│   │   ├── model_a.py               # Scratch EfficientNet-B0
│   │   ├── model_b.py               # ImageNet EfficientNet-B0
│   │   ├── model_c.py               # DINOv2 ViT-S/14
│   │   └── model_d.py               # BiomedCLIP ViT-B/16
│   └── training/
│       └── trainer.py               # Modular reproducible training pipeline
├── scripts/
│   └── run_kaggle_pilot.py          # Stage 9D GPU pilot run & VRAM benchmark
├── .gitignore
├── requirements.txt
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

### 2. Running the Pilot GPU Benchmark
```bash
python scripts/run_kaggle_pilot.py
```
This runs a 3-epoch pilot benchmark of Model D (BiomedCLIP) to measure peak GPU VRAM allocation, throughput, and verify artifact generation before launching the full experiment grid.
