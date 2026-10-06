"""
Stage 9D: Single Kaggle GPU Pilot Run Script.

Executes a 3-epoch pilot run of Model D (BiomedCLIP ViT-B/16) on 100% data (Seed 42).
- Starting batch size: 32 (dynamic fallback to 16/8 if CUDA OOM occurs)
- Measures peak GPU VRAM allocation & reservation
- Measures average per-epoch training throughput
- Verifies output artifact generation
- Saves stage9d_pilot_vram_report.json
"""

import os
import sys
import time
import json
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ExperimentConfig
from src.training.trainer import run_experiment, set_seed


def run_pilot(image_dirs=None, initial_batch_size=32):
    print("=" * 70)
    print("=== STAGE 9D: KAGGLE GPU PILOT RUN (MODEL D BIOMEDCLIP 100% SEED 42) ===")
    print("=" * 70)
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Hardware Device: {device_name}")
    print(f"[INFO] Proposed Starting Batch Size: {initial_batch_size}")
    
    batch_size = initial_batch_size
    feasible = False
    config = None
    metrics = None
    vram_allocated_mb = 0.0
    vram_reserved_mb = 0.0
    
    # Checkpoint output directory for pilot
    pilot_exp_dir = os.path.join(PROJECT_ROOT, "experiments")
    
    while not feasible and batch_size >= 1:
        print(f"\n[PILOT ATTEMPT] Testing Batch Size = {batch_size}...")
        
        config = ExperimentConfig(
            model_name='model_d',
            fraction_pct='100%',
            seed=42,
            batch_size=batch_size,
            learning_rate=1e-4,
            weight_decay=1e-4,
            num_epochs=3,
            patience=5,
            project_root=PROJECT_ROOT,
            experiments_dir=pilot_exp_dir
        )
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            start_pilot_time = time.time()
            metrics = run_experiment(config, image_dirs=image_dirs)
            total_pilot_seconds = time.time() - start_pilot_time
            
            feasible = True
            
            if torch.cuda.is_available():
                vram_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                
            print(f"\n[SUCCESS] Batch Size = {batch_size} executed successfully!")
            
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[WARNING] CUDA Out-Of-Memory error with batch size {batch_size}!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = batch_size // 2
            print(f"[FALLBACK] Reducing batch size to {batch_size} and retrying...")
            
        except Exception as e:
            # If unexpected error occurs
            print(f"[ERROR] Pilot execution encountered exception: {e}")
            raise e

    if not feasible:
        raise RuntimeError("Pilot failed: Unable to fit Model D even with minimum batch size!")

    # Summary Report
    vram_allocated_gb = vram_allocated_mb / 1024.0
    vram_reserved_gb = vram_reserved_mb / 1024.0
    avg_epoch_seconds = total_pilot_seconds / 3.0
    
    print("\n" + "=" * 70)
    print("=== PILOT EXPERIMENT RESULTS SUMMARY ===")
    print("=" * 70)
    print(f"Model Name            : Model D (BiomedCLIP ViT-B/16)")
    print(f"Dataset Fraction      : 100% (Seed 42)")
    print(f"Proposed Batch Size   : {initial_batch_size}")
    print(f"Actual Feasible Batch : {batch_size}")
    print(f"Peak GPU VRAM Alloc   : {vram_allocated_mb:.2f} MB ({vram_allocated_gb:.2f} GB)")
    print(f"Peak GPU VRAM Reserv  : {vram_reserved_mb:.2f} MB ({vram_reserved_gb:.2f} GB)")
    print(f"Total 3-Epoch Runtime : {total_pilot_seconds:.2f} s ({total_pilot_seconds/60:.2f} min)")
    print(f"Avg Runtime per Epoch : {avg_epoch_seconds:.2f} s")
    print(f"Best Val Macro F1     : {metrics.get('best_val_macro_f1', 0.0):.4f}")
    print(f"Test Balanced Acc     : {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"Test Macro F1         : {metrics.get('macro_f1', 0.0):.4f}")
    print(f"Test ECE              : {metrics.get('ece', 0.0):.4f}")
    print("=" * 70)
    
    # Save Stage 9D Pilot VRAM Report
    report_data = {
        'model_name': 'model_d',
        'backbone': 'BiomedCLIP ViT-B/16',
        'fraction_pct': '100%',
        'seed': 42,
        'proposed_batch_size': initial_batch_size,
        'actual_feasible_batch_size': batch_size,
        'peak_vram_allocated_mb': round(vram_allocated_mb, 2),
        'peak_vram_allocated_gb': round(vram_allocated_gb, 2),
        'peak_vram_reserved_mb': round(vram_reserved_mb, 2),
        'peak_vram_reserved_gb': round(vram_reserved_gb, 2),
        'total_3_epoch_runtime_seconds': round(total_pilot_seconds, 2),
        'total_3_epoch_runtime_minutes': round(total_pilot_seconds / 60.0, 2),
        'avg_epoch_runtime_seconds': round(avg_epoch_seconds, 2),
        'best_val_macro_f1': metrics.get('best_val_macro_f1'),
        'test_metrics': {
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'macro_auroc': metrics.get('macro_auroc'),
            'ece': metrics.get('ece')
        },
        'output_artifacts_verified': {
            'config.json': os.path.exists(os.path.join(config.run_dir, 'config.json')),
            'history.json': os.path.exists(os.path.join(config.run_dir, 'history.json')),
            'metrics.json': os.path.exists(os.path.join(config.run_dir, 'metrics.json')),
            'predictions.csv': os.path.exists(os.path.join(config.run_dir, 'predictions.csv')),
            'best_checkpoint.pt': os.path.exists(os.path.join(config.run_dir, 'best_checkpoint.pt')),
            'train.log': os.path.exists(os.path.join(config.run_dir, 'train.log'))
        }
    }
    
    report_path = os.path.join(PROJECT_ROOT, 'data', 'stage9d_pilot_vram_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\n[INFO] Stage 9D Pilot Report saved to {report_path}")
    return report_data


if __name__ == '__main__':
    run_pilot()
