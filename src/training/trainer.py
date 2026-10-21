"""
Modular Trainer Pipeline for HAM10000 Research Experiments.

Handles:
- Deterministic seed setup & RNG state preservation
- Model factory loading (Models A, B, C, D)
- Fixed master validation & test evaluation
- Class-weighted CrossEntropyLoss training
- Epoch history tracking & logging
- Best checkpoint selection (Primary Criterion: Best Validation Macro F1-Score)
- Latest checkpoint saving after every epoch for interruption safety
- Resumable training: automatically resumes from latest_checkpoint.pt if incomplete
- Completed experiment skip detection: skips completed runs with completion marker
- Final test evaluation & Stage 7 metric logging
- Reproducible artifact saving (config.json, history.json, metrics.json, predictions.csv, best_checkpoint.pt, latest_checkpoint.pt)
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import RAFDBDataset, HAM10000Dataset, CLASS_TO_IDX
from src.data.transforms import get_transforms
from src.models.model_a import build_model_a
from src.models.model_b import build_model_b
from src.models.model_c import build_model_c
from src.models.model_d import build_model_d
from src.evaluation.metrics import calculate_all_metrics


def set_seed(seed):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states(device='cpu'):
    """Captures random generator states across Python, NumPy, and PyTorch."""
    states = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }
    if torch.cuda.is_available() and 'cuda' in str(device):
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states, device='cpu'):
    """Restores random generator states across Python, NumPy, and PyTorch."""
    if not states:
        return
    if 'torch' in states:
        torch.set_rng_state(states['torch'])
    if 'numpy' in states:
        np.random.set_state(states['numpy'])
    if 'random' in states:
        random.setstate(states['random'])
    if torch.cuda.is_available() and 'cuda' in str(device) and 'torch_cuda' in states:
        torch.cuda.set_rng_state_all(states['torch_cuda'])


def safe_torch_load(path, map_location='cpu'):
    """
    Safely loads PyTorch checkpoints across PyTorch versions,
    handling PyTorch 2.6+ weights_only unpickling behavior.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_experiment_completed(run_dir):
    """
    Checks whether an experiment is fully completed.
    An experiment is complete if and only if:
    1. metrics.json exists and contains "status": "completed"
    2. predictions.csv exists and is non-empty
    3. best_checkpoint.pt exists and is non-empty
    4. history.json exists and is non-empty
    5. config.json exists and is non-empty
    """
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        return False

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        if metrics_data.get('status') != 'completed':
            return False
    except Exception:
        return False

    required_artifacts = ['predictions.csv', 'best_checkpoint.pt', 'history.json', 'config.json']
    for req_file in required_artifacts:
        p = os.path.join(run_dir, req_file)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False

    return True


def get_model(model_name, num_classes=7):
    """Model factory function."""
    if model_name == 'model_a':
        return build_model_a(num_classes=num_classes)
    elif model_name == 'model_b':
        return build_model_b(num_classes=num_classes)
    elif model_name == 'model_c':
        return build_model_c(num_classes=num_classes)
    elif model_name == 'model_d':
        return build_model_d(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def calculate_class_weights(dataset=None):
    """
    Calculates inverse class frequency weights normalized to sum to 7.0 for RAF-DB.
    If dataset is provided with samples, computes dynamic weights for the active fraction;
    otherwise defaults to standard RAF-DB training partition frequencies.
    """
    num_classes = 7
    if dataset is not None and hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        counts = [0] * num_classes
        label_col = getattr(dataset, 'label_col', 'label')
        for val in dataset.samples[label_col]:
            try:
                c_idx = dataset._parse_label(val)
                if 0 <= c_idx < num_classes:
                    counts[c_idx] += 1
            except Exception:
                pass
        counts = [max(c, 1) for c in counts]
        total_samples = sum(counts)
        class_counts = counts
    else:
        # Canonical RAF-DB basic emotion train partition counts (12,271 images)
        # Surprise: 1290, Fear: 281, Disgust: 717, Happiness: 4772, Sadness: 1982, Anger: 705, Neutral: 2524
        total_samples = 12271
        class_counts = [1290, 281, 717, 4772, 1982, 705, 2524]

    weights = [total_samples / (num_classes * count) for count in class_counts]
    sum_w = sum(weights)
    normalized_weights = [w * (num_classes / sum_w) for w in weights]

    return torch.tensor(normalized_weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs 1 training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate(model, loader, criterion, device, num_classes=7):
    """Evaluates model on validation or test set."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_ids = batch['image_id']

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_image_ids.extend(image_ids)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if len(all_labels) == 0:
        # Edge case guard for empty loader (e.g., local micro testing)
        dummy_metrics = {
            'balanced_accuracy': 0.0,
            'macro_f1': 0.0,
            'macro_auroc': 0.0,
            'confusion_matrix': np.zeros((num_classes, num_classes)),
            'sensitivities': [0.0] * num_classes,
            'specificities': [0.0] * num_classes,
            'ece': 0.0,
            'bin_accuracies': [0.0] * 10,
            'bin_confidences': [0.0] * 10,
            'bin_counts': [0] * 10,
            'loss': 0.0
        }
        return dummy_metrics, np.array([]), np.zeros((0, num_classes)), []

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    metrics = calculate_all_metrics(all_labels, all_probs, num_classes=num_classes)
    metrics['loss'] = avg_loss

    return metrics, all_labels, all_probs, all_image_ids


def run_experiment(config, image_dirs):
    """
    Executes a complete training and evaluation run with robust resume and skip safety.
    """
    # 1. Skip if already completed
    if is_experiment_completed(config.run_dir):
        skip_msg = f"[SKIP] Experiment already completed: Model={config.model_name} | Frac={config.fraction_pct} | Seed={config.seed}"
        print(skip_msg)
        log_file = os.path.join(config.run_dir, 'train.log')
        if os.path.exists(log_file):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {skip_msg}\n")
        metrics_path = os.path.join(config.run_dir, 'metrics.json')
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Deterministic seed initialization
    set_seed(config.seed)
    os.makedirs(config.run_dir, exist_ok=True)

    # Save / update config.json
    config_json_path = os.path.join(config.run_dir, 'config.json')
    config.save_json(config_json_path)

    # Setup Logger (mode='a' appends across resumes)
    log_file = os.path.join(config.run_dir, 'train.log')
    logger = logging.getLogger(f"exp_{config.model_name}_{config.seed}_{config.fraction_pct}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(sh)

    logger.info("=" * 60)
    logger.info(f"Starting Experiment: Model={config.model_name} | Frac={config.fraction_pct} | Seed={config.seed}")
    logger.info(f"Device: {config.device} | Primary Criterion: Best Validation Macro F1")
    logger.info("=" * 60)

    # Build Data Transforms
    train_transform = get_transforms(is_training=True, image_size=config.image_size, mean=config.norm_mean, std=config.norm_std)
    eval_transform = get_transforms(is_training=False, image_size=config.image_size, mean=config.norm_mean, std=config.norm_std)

    # Build Datasets
    train_dataset = RAFDBDataset(config.fraction_csv, image_dirs, split='train', transform=train_transform)
    val_dataset = RAFDBDataset(config.master_split_csv, image_dirs, split='val', transform=eval_transform)
    test_dataset = RAFDBDataset(config.master_split_csv, image_dirs, split='test', transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    logger.info(f"Loaded Datasets: Train={len(train_dataset)} images | Val={len(val_dataset)} images | Test={len(test_dataset)} images")

    # Build Model
    device = torch.device(config.device)
    model = get_model(config.model_name, num_classes=config.num_classes).to(device)

    # Class-Weighted Loss & Optimizer (computed dynamically for active fraction or standard RAF-DB)
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint_path = os.path.join(config.run_dir, 'best_checkpoint.pt')
    latest_checkpoint_path = os.path.join(config.run_dir, 'latest_checkpoint.pt')

    start_epoch = 1
    best_val_macro_f1 = -1.0
    patience_counter = 0
    history = []

    # 2. Check for resumable latest checkpoint
    if os.path.exists(latest_checkpoint_path):
        try:
            logger.info(f"[RESUME] Found latest checkpoint at {latest_checkpoint_path}")
            latest_cp = safe_torch_load(latest_checkpoint_path, map_location=device)
            last_completed_epoch = latest_cp['epoch']
            start_epoch = last_completed_epoch + 1
            model.load_state_dict(latest_cp['model_state_dict'])
            optimizer.load_state_dict(latest_cp['optimizer_state_dict'])
            best_val_macro_f1 = latest_cp.get('best_val_macro_f1', -1.0)
            patience_counter = latest_cp.get('patience_counter', 0)
            history = latest_cp.get('history', [])
            if 'rng_states' in latest_cp:
                set_rng_states(latest_cp['rng_states'], device=config.device)
            logger.info(
                f"[RESUME] Resuming from epoch {start_epoch:02d}/{config.num_epochs:02d} "
                f"(Best Val Macro F1 so far: {best_val_macro_f1:.4f}, Patience counter: {patience_counter})"
            )
        except Exception as e:
            logger.warning(f"[RESUME WARNING] Failed to load latest checkpoint: {e}. Starting fresh from epoch 1.")
            start_epoch = 1
            best_val_macro_f1 = -1.0
            patience_counter = 0
            history = []

    start_time = time.time()

    # 3. Epoch Loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, num_classes=config.num_classes)

        val_loss = val_metrics['loss']
        val_acc = val_metrics['balanced_accuracy']
        val_f1 = val_metrics['macro_f1']

        epoch_time = time.time() - epoch_start

        epoch_log = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_balanced_acc': round(val_acc, 4),
            'val_macro_f1': round(val_f1, 4),
            'val_ece': round(val_metrics['ece'], 4),
            'epoch_seconds': round(epoch_time, 2)
        }
        history.append(epoch_log)

        logger.info(f"Epoch {epoch:02d}/{config.num_epochs:02d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BalAcc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} [{epoch_time:.1f}s]")

        # Primary Checkpoint Selection Criterion: Best Validation Macro F1-Score
        if val_f1 > best_val_macro_f1 or (epoch == 1 and not os.path.exists(checkpoint_path)):
            best_val_macro_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_f1,
                'config': config.to_dict()
            }, checkpoint_path)
            logger.info(f"   [SAVED BEST CHECKPOINT] Val Macro F1: {val_f1:.4f}")
        else:
            patience_counter += 1

        # Save latest_checkpoint.pt after EVERY completed epoch (interruption safety)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_macro_f1': best_val_macro_f1,
            'patience_counter': patience_counter,
            'history': history,
            'rng_states': get_rng_states(device=config.device),
            'config': config.to_dict()
        }, latest_checkpoint_path)
        logger.info(f"   [SAVED LATEST CHECKPOINT] Epoch {epoch:02d} complete.")

        # Save history.json incrementally
        with open(os.path.join(config.run_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)

        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (Patience={config.patience})")
            break

    total_training_time = time.time() - start_time
    logger.info(f"Training loop concluded in {total_training_time / 60:.2f} minutes.")

    # Save final history.json
    with open(os.path.join(config.run_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    # 4. Evaluate Best Checkpoint ONCE on master FIXED Test Set
    logger.info("\nLoading Best Checkpoint for Final Test Set Evaluation...")
    best_cp = safe_torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(best_cp['model_state_dict'])

    test_metrics, test_labels, test_probs, test_image_ids = evaluate(model, test_loader, criterion, device, num_classes=config.num_classes)

    logger.info("=== FINAL TEST SET METRICS ===")
    logger.info(f"  Balanced Accuracy : {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1-Score    : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Macro AUROC       : {test_metrics['macro_auroc']:.4f}")
    logger.info(f"  ECE (10 Bins)     : {test_metrics['ece']:.4f}")

    # Save predictions.csv
    pred_df = pd.DataFrame({
        'image_id': test_image_ids,
        'true_label': test_labels,
        'pred_label': np.argmax(test_probs, axis=1) if len(test_probs) > 0 else []
    })
    for c in range(config.num_classes):
        pred_df[f'prob_class_{c}'] = test_probs[:, c] if len(test_probs) > 0 else []
    pred_df.to_csv(os.path.join(config.run_dir, 'predictions.csv'), index=False)

    # Save metrics.json with completion marker: status = "completed"
    serializable_metrics = {
        'status': 'completed',
        'model_name': config.model_name,
        'fraction_pct': config.fraction_pct,
        'seed': config.seed,
        'balanced_accuracy': round(test_metrics['balanced_accuracy'], 4),
        'macro_f1': round(test_metrics['macro_f1'], 4),
        'macro_auroc': round(test_metrics['macro_auroc'], 4) if not np.isnan(test_metrics['macro_auroc']) else None,
        'ece': round(test_metrics['ece'], 4),
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'sensitivities': [round(x, 4) if not np.isnan(x) else None for x in test_metrics['sensitivities']],
        'specificities': [round(x, 4) if not np.isnan(x) else None for x in test_metrics['specificities']],
        'bin_accuracies': [round(x, 4) if not np.isnan(x) else None for x in test_metrics['bin_accuracies']],
        'bin_confidences': [round(x, 4) if not np.isnan(x) else None for x in test_metrics['bin_confidences']],
        'bin_counts': test_metrics['bin_counts'],
        'best_epoch': best_cp['epoch'],
        'best_val_macro_f1': round(best_cp['val_macro_f1'], 4),
        'total_training_minutes': round(total_training_time / 60, 2)
    }
    with open(os.path.join(config.run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4)

    logger.info(f"Experiment outputs successfully saved to {config.run_dir}")
    logger.info("[COMPLETED] Experiment status marked 'completed'.")

    # Close logger handlers to release file lock on train.log
    fh.close()
    sh.close()
    logger.removeHandler(fh)
    logger.removeHandler(sh)

    return serializable_metrics
