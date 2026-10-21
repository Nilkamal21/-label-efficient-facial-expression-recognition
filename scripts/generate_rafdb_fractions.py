"""
Script to generate reproducible master split and stratified training data fractions for RAF-DB.

Target Fractions: 100%, 75%, 50%, 25%, 10%, 5%, 1%
Target Seeds: 42, 43, 44

Handles:
- Loading list_patition_label.txt or existing metadata CSV
- Converting RAF-DB 1-based labels (1=Surprise ... 7=Neutral) to 0-based indices and emotion names
- Fixed Master Validation split from official training set
- Stratified sampling across all 7 emotion classes
- Guaranteed minimum samples per class (ensures minority classes like Fear never drop to 0 at 1% or 5%)
- Zero data leakage into validation or test partitions
- Outputs CSV files under data/splits/ and data/splits/fractions/seed_{42,43,44}/
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import CLASSES, RAFDB_NUMERIC_MAP

TARGET_FRACTIONS = [1.00, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01]
FRACTION_NAMES = ["100", "75", "50", "25", "10", "5", "1"]
SEEDS = [42, 43, 44]


def parse_rafdb_label_file(file_path):
    """
    Parses standard RAF-DB list_patition_label.txt:
    e.g. 'train_00001_aligned.jpg 5'
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                raw_label = int(parts[1])
                emotion_idx = RAFDB_NUMERIC_MAP.get(raw_label, raw_label - 1)
                emotion_name = CLASSES[emotion_idx]
                split = 'train' if img_name.startswith('train') else 'test'
                records.append({
                    'image_id': img_name,
                    'label': emotion_idx,
                    'emotion': emotion_name,
                    'raw_split': split
                })
    return pd.DataFrame(records)


def stratified_sample_with_min(df, frac, seed, min_per_class=1):
    """
    Stratified sampling that preserves class proportions while strictly
    guaranteeing that every class has at least `min_per_class` samples.
    This prevents minority classes (e.g., Fear with ~2% support in RAF-DB)
    from having 0 samples at low fractions (1%, 5%).
    """
    if frac >= 1.0:
        return df.copy()

    sampled_subsets = []
    for label, group in df.groupby('label'):
        target_n = int(round(len(group) * frac))
        actual_n = max(min_per_class, target_n)
        actual_n = min(actual_n, len(group))
        sample = group.sample(n=actual_n, random_state=seed)
        sampled_subsets.append(sample)

    combined = pd.concat(sampled_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def create_rafdb_splits(metadata_or_label_path, val_ratio=0.10):
    """
    Creates master train, val, test split and the 7 training fraction subsets.
    """
    print("=" * 70)
    print("=== GENERATING RAF-DB DATA SPLITS AND 7-FRACTION DATASETS ===")
    print("=" * 70)

    if metadata_or_label_path.endswith('.txt'):
        df = parse_rafdb_label_file(metadata_or_label_path)
    else:
        df = pd.read_csv(metadata_or_label_path)

    # Separate train and test according to standard RAF-DB partition
    if 'raw_split' in df.columns:
        train_pool = df[df['raw_split'] == 'train'].copy()
        test_df = df[df['raw_split'] == 'test'].copy()
        test_df['split'] = 'test'
    else:
        # Fallback if unpartitioned
        train_pool, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
        test_df['split'] = 'test'

    # Partition official train pool into Master Train and Master Val (stratified, seed 42)
    master_train_df, master_val_df = train_test_split(
        train_pool,
        test_size=val_ratio,
        stratify=train_pool['label'],
        random_state=42
    )
    master_train_df = master_train_df.copy()
    master_val_df = master_val_df.copy()

    master_train_df['split'] = 'train'
    master_val_df['split'] = 'val'

    # Master Split DF
    master_split_df = pd.concat([master_train_df, master_val_df, test_df], ignore_index=True)
    splits_dir = os.path.join(PROJECT_ROOT, 'data', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    master_csv_path = os.path.join(splits_dir, 'rafdb_master_split.csv')
    master_split_df.to_csv(master_csv_path, index=False)
    print(f"[OK] Master Split saved to {master_csv_path}")
    print(f"     Train: {len(master_train_df)} | Val: {len(master_val_df)} | Test: {len(test_df)}")

    # Generate 7 fractions across 3 seeds
    summary_records = []
    fractions_base_dir = os.path.join(splits_dir, 'fractions')

    for seed in SEEDS:
        seed_dir = os.path.join(fractions_base_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for frac_float, frac_str in zip(TARGET_FRACTIONS, FRACTION_NAMES):
            subset_df = stratified_sample_with_min(
                master_train_df,
                frac=frac_float,
                seed=seed,
                min_per_class=1
            )

            subset_csv_path = os.path.join(seed_dir, f"train_{frac_str}.csv")
            subset_df.to_csv(subset_csv_path, index=False)

            # Get per-class breakdown for logging transparency
            class_counts = subset_df['label'].value_counts().to_dict()
            min_c = min(class_counts.values())
            max_c = max(class_counts.values())

            summary_records.append({
                'seed': seed,
                'fraction_pct': f"{frac_str}%",
                'image_count': len(subset_df),
                'min_class_count': min_c,
                'max_class_count': max_c,
                'file_path': subset_csv_path
            })
            print(f"  [Seed {seed}] Fraction {frac_str:>3}%: {len(subset_df):>5} images (min/max class: {min_c}/{max_c}) saved.")

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(fractions_base_dir, 'fraction_sampling_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Summary table saved to {summary_csv_path}")
    print("=" * 70 + "\n")
    return master_split_df, summary_df


KAGGLE_CANDIDATE_LABEL_PATHS = [
    "/kaggle/input/raf-db-dataset/basic/EmoLabel/list_patition_label.txt",
    "/kaggle/input/real-world-affective-faces-database/basic/EmoLabel/list_patition_label.txt",
    "/kaggle/input/raf-db/basic/EmoLabel/list_patition_label.txt",
    "/kaggle/input/raf-db-dataset/list_patition_label.txt",
    "/kaggle/input/raf-db/list_patition_label.txt",
    "/kaggle/input/real-world-affective-faces-database/list_patition_label.txt",
]


if __name__ == '__main__':
    label_file = None
    if len(sys.argv) > 1:
        label_file = sys.argv[1]
    else:
        # Search candidate Kaggle paths
        for p in KAGGLE_CANDIDATE_LABEL_PATHS:
            if os.path.isfile(p):
                label_file = p
                print(f"[INFO] Auto-detected RAF-DB label file at: {label_file}")
                break

    if label_file and os.path.isfile(label_file):
        create_rafdb_splits(label_file)
    else:
        print("Usage: python scripts/generate_rafdb_fractions.py <path_to_list_patition_label.txt_or_csv>")
        print("Candidate paths searched:")
        for p in KAGGLE_CANDIDATE_LABEL_PATHS:
            print(f"  - {p}")
