"""
Batch Experiment Execution Script for RAF-DB Facial Expression Recognition Research Study.

Executes a specified sequential batch of experiments (e.g., 2–4 runs).
Features:
- Sequential execution of experiment specifications (model, fraction, seed).
- Pre-flight validation of Kaggle/local image paths and required split CSVs.
- Completed experiment detection: skips already completed runs.
- Resumable training: automatically resumes partially completed runs from latest_checkpoint.pt.
- Restart-safe: re-running the same command skips finished runs and continues seamlessly.
- Error isolation: failures are logged clearly and never marked as completed.
- Progress summary table printed at the conclusion of the batch.
"""

import os
import sys
import time
import argparse
import traceback

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ExperimentConfig
from src.training.trainer import run_experiment, is_experiment_completed

# Standard Kaggle RAF-DB mount paths
KAGGLE_RAFDB_CANDIDATE_PATHS = [
    # Aligned images directory (most standard for RAF-DB)
    "/kaggle/input/raf-db-dataset/basic/Image/aligned",
    "/kaggle/input/real-world-affective-faces-database/basic/Image/aligned",
    "/kaggle/input/raf-db/basic/Image/aligned",
    "/kaggle/input/raf-db-dataset/DATASET/train",
    "/kaggle/input/raf-db-dataset/DATASET/test",
    "/kaggle/input/raf-db/DATASET/train",
    "/kaggle/input/raf-db/DATASET/test",
    # Parent dataset folders
    "/kaggle/input/raf-db-dataset",
    "/kaggle/input/real-world-affective-faces-database",
    "/kaggle/input/raf-db"
]

# Canonical grid definitions (4 models x 7 fractions x 3 seeds = 84 runs)
ALL_MODELS = ['model_a', 'model_b', 'model_c', 'model_d']
ALL_FRACTIONS = ['100%', '75%', '50%', '25%', '10%', '5%', '1%']
ALL_SEEDS = [42, 43, 44]


def resolve_image_dirs(image_dirs=None):
    """
    Resolves RAF-DB image directories:
    1. Explicit directory argument (str or list)
    2. Auto-detection across known Kaggle RAF-DB mount paths
    """
    if image_dirs:
        if isinstance(image_dirs, str):
            return [image_dirs]
        return list(image_dirs)

    # Search for existing Kaggle directories that contain face images
    detected = []
    for p in KAGGLE_RAFDB_CANDIDATE_PATHS:
        if os.path.isdir(p):
            # Check if this dir or subdirs has jpg/png images
            has_imgs = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(p))
            if has_imgs:
                detected.append(p)

    if detected:
        return detected

    raise FileNotFoundError(
        "RAF-DB image directories were not detected on Kaggle.\n"
        "Please specify image directories with --image_dirs, e.g.:\n"
        "  --image_dirs /kaggle/input/raf-db-dataset/basic/Image/aligned\n"
        "or for local testing:\n"
        "  --image_dirs data/sample_images"
    )


def validate_environment(image_dirs, experiment_specs):
    """
    Pre-flight check:
    - Verifies image directories exist and contain JPG images.
    - Verifies required master split and fraction CSVs exist.
    """
    print("=" * 70)
    print("=== PRE-FLIGHT VALIDATION ===")
    print("=" * 70)

    # 1. Image directories
    for d in image_dirs:
        if not os.path.exists(d) or not os.path.isdir(d):
            raise FileNotFoundError(f"[ERROR] Image directory not found: {d}")
        img_count = sum(1 for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if img_count == 0:
            raise FileNotFoundError(f"[ERROR] No image files (JPG/PNG) found in {d}")
        print(f"  [OK] Image Dir: {d} ({img_count} images)")

    # 2. Master split CSV (RAF-DB master split)
    master_split_csv = os.path.join(PROJECT_ROOT, 'data', 'splits', 'rafdb_master_split.csv')
    if not os.path.isfile(master_split_csv):
        # Attempt to auto-generate RAF-DB splits if label file is present on Kaggle
        from scripts.generate_rafdb_fractions import KAGGLE_CANDIDATE_LABEL_PATHS, create_rafdb_splits
        found_label = None
        for p in KAGGLE_CANDIDATE_LABEL_PATHS:
            if os.path.isfile(p):
                found_label = p
                break
        if found_label:
            print(f"[INFO] Auto-generating RAF-DB splits from {found_label}...")
            create_rafdb_splits(found_label)
        else:
            raise FileNotFoundError(
                f"[ERROR] Master split CSV missing: {master_split_csv}\n"
                f"Please run 'python scripts/generate_rafdb_fractions.py <path_to_list_patition_label.txt>' first."
            )
    print(f"  [OK] Master Split CSV: {master_split_csv}")

    # 3. Fraction CSVs for all requested specs
    for m, frac, seed in experiment_specs:
        clean_frac = frac.replace('%', '')
        frac_csv = os.path.join(PROJECT_ROOT, 'data', 'splits', 'fractions', f"seed_{seed}", f"train_{clean_frac}.csv")
        if not os.path.isfile(frac_csv):
            raise FileNotFoundError(f"[ERROR] Required fraction CSV missing: {frac_csv}")

    print("  [OK] All required fraction CSVs confirmed.")
    print("=" * 70 + "\n")


def parse_experiment_spec(spec_str):
    """
    Parses a string like 'model_a:100%:42' or 'model_a:100:42' into (model, frac, seed).
    """
    parts = spec_str.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid experiment spec '{spec_str}'. Expected format: 'model:fraction:seed' (e.g. model_a:100%:42)")
    model, frac, seed_str = parts
    model = model.lower()
    if not frac.endswith('%'):
        frac = f"{frac}%"
    seed = int(seed_str)

    if model not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model}'. Valid: {ALL_MODELS}")
    if frac not in ALL_FRACTIONS:
        raise ValueError(f"Invalid fraction '{frac}'. Valid: {ALL_FRACTIONS}")
    if seed not in ALL_SEEDS:
        raise ValueError(f"Invalid seed '{seed}'. Valid: {ALL_SEEDS}")

    return model, frac, seed


def get_predefined_batches(batch_size=3):
    """
    Partitions the full 96-run grid into sequential small batches.
    Standard grouping: by model, by fraction across 3 seeds (e.g. 32 batches of 3 runs).
    """
    grid = []
    for m in ALL_MODELS:
        for frac in ALL_FRACTIONS:
            for s in ALL_SEEDS:
                grid.append((m, frac, s))

    batches = []
    for i in range(0, len(grid), batch_size):
        batches.append(grid[i:i + batch_size])
    return batches


def run_batch(experiment_specs, image_dirs=None, epochs=None, batch_size=None, learning_rate=None):
    """
    Executes a sequence of experiment specifications with resume and skip safety.
    """
    resolved_image_dirs = resolve_image_dirs(image_dirs)
    validate_environment(resolved_image_dirs, experiment_specs)

    print("=" * 70)
    print(f"=== BATCH EXECUTION START: {len(experiment_specs)} EXPERIMENTS QUEUED ===")
    print("=" * 70)
    for i, (m, f, s) in enumerate(experiment_specs, 1):
        print(f"  {i}. Model: {m:<8} | Fraction: {f:<5} | Seed: {s}")
    print("=" * 70 + "\n")

    results_summary = []

    for idx, (model_name, frac_pct, seed) in enumerate(experiment_specs, 1):
        spec_str = f"{model_name}:{frac_pct}:{seed}"
        print("\n" + "#" * 70)
        print(f"### [{idx}/{len(experiment_specs)}] EXPERIMENT: {spec_str}")
        print("#" * 70)

        # Build config
        config_kwargs = {
            'model_name': model_name,
            'fraction_pct': frac_pct,
            'seed': seed,
            'project_root': PROJECT_ROOT,
            'experiments_dir': os.path.join(PROJECT_ROOT, 'experiments')
        }
        if epochs is not None:
            config_kwargs['num_epochs'] = epochs
        if batch_size is not None:
            config_kwargs['batch_size'] = batch_size
        if learning_rate is not None:
            config_kwargs['learning_rate'] = learning_rate

        config = ExperimentConfig(**config_kwargs)

        # Check completed status
        if is_experiment_completed(config.run_dir):
            print(f"[SKIP] Experiment already completed: {spec_str}")
            results_summary.append({
                'spec': spec_str,
                'status': 'SKIPPED (Already Completed)',
                'best_val_f1': '-',
                'test_macro_f1': '-',
                'test_bal_acc': '-',
                'runtime_min': '-'
            })
            continue

        # Check whether this will be a fresh start or a resume
        latest_cp = os.path.join(config.run_dir, 'latest_checkpoint.pt')
        initial_mode = "RESUMING" if os.path.exists(latest_cp) else "FRESH START"
        print(f"[INFO] Execution Mode: {initial_mode}")

        exp_start = time.time()
        try:
            metrics = run_experiment(config, image_dirs=resolved_image_dirs)
            duration_min = round((time.time() - exp_start) / 60.0, 2)
            print(f"\n[SUCCESS] Experiment {spec_str} completed in {duration_min} minutes.")

            results_summary.append({
                'spec': spec_str,
                'status': f"COMPLETED ({'Resumed' if initial_mode == 'RESUMING' else 'Fresh'})",
                'best_val_f1': f"{metrics.get('best_val_macro_f1', 0.0):.4f}",
                'test_macro_f1': f"{metrics.get('macro_f1', 0.0):.4f}",
                'test_bal_acc': f"{metrics.get('balanced_accuracy', 0.0):.4f}",
                'runtime_min': f"{duration_min:.2f}"
            })

        except Exception as e:
            duration_min = round((time.time() - exp_start) / 60.0, 2)
            print(f"\n[ERROR] Experiment {spec_str} failed with error: {e}")
            traceback.print_exc()
            results_summary.append({
                'spec': spec_str,
                'status': f"FAILED: {type(e).__name__}",
                'best_val_f1': 'FAILED',
                'test_macro_f1': 'FAILED',
                'test_bal_acc': 'FAILED',
                'runtime_min': f"{duration_min:.2f}"
            })

    # Print final batch summary table
    print("\n" + "=" * 80)
    print("=== BATCH RUN EXECUTION SUMMARY ===")
    print("=" * 80)
    header = f"{'Experiment':<22} | {'Status':<25} | {'Val Macro F1':<12} | {'Test F1':<10} | {'Runtime'}"
    print(header)
    print("-" * 80)
    for r in results_summary:
        print(f"{r['spec']:<22} | {r['status']:<25} | {r['best_val_f1']:<12} | {r['test_macro_f1']:<10} | {r['runtime_min']} min")
    print("=" * 80 + "\n")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Batch Runner for RAF-DB Facial Expression Recognition Research Experiments.")
    parser.add_argument(
        '--experiments',
        nargs='+',
        help="List of experiments in format 'model:fraction:seed' (e.g. model_a:100%:42 model_a:100%:43 model_a:100%:44)"
    )
    parser.add_argument(
        '--batch',
        type=int,
        help="Predefined batch number (1 to 28, each batch runs 3 experiments across seeds 42, 43, 44)"
    )
    parser.add_argument(
        '--model',
        choices=ALL_MODELS,
        help="Run for a specific model combined with --fractions and --seeds"
    )
    parser.add_argument(
        '--fractions',
        nargs='+',
        help="Fractions to run with --model (e.g. 100% 75%)"
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        help="Seeds to run with --model (e.g. 42 43 44)"
    )
    parser.add_argument(
        '--image_dirs',
        nargs='+',
        help="Explicit image directories (optional; automatically detects Kaggle mounts if omitted)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help="Override total epochs for testing"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help="Override batch size"
    )

    args = parser.parse_args()

    specs = []
    if args.experiments:
        specs = [parse_experiment_spec(s) for s in args.experiments]
    elif args.batch is not None:
        batches = get_predefined_batches(batch_size=3)
        if args.batch < 1 or args.batch > len(batches):
            print(f"[ERROR] Batch number must be between 1 and {len(batches)}")
            sys.exit(1)
        specs = batches[args.batch - 1]
        print(f"[INFO] Selected Predefined Batch #{args.batch} of {len(batches)}")
    elif args.model:
        fractions = args.fractions or ['100%']
        seeds = args.seeds or ALL_SEEDS
        fractions = [f if f.endswith('%') else f"{f}%" for f in fractions]
        for f in fractions:
            for s in seeds:
                specs.append((args.model, f, s))
    else:
        print("[INFO] No experiment specification provided. Displaying predefined batch mapping:")
        batches = get_predefined_batches(batch_size=3)
        for i, b in enumerate(batches, 1):
            sample = b[0]
            print(f"  Batch {i:2d}: {sample[0]} / {sample[1]} / seeds {', '.join(str(x[2]) for x in b)}")
        print("\nUsage example:")
        print("  python scripts/run_batch_experiments.py --batch 1")
        print("  python scripts/run_batch_experiments.py --experiments model_a:100%:42 model_a:100%:43 model_a:100%:44")
        sys.exit(0)

    run_batch(
        experiment_specs=specs,
        image_dirs=args.image_dirs,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
