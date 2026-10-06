"""
Modular Trainer Pipeline for HAM10000 Research Experiments.

Handles:
- Deterministic seed setup
- Model factory loading (Models A, B, C, D)
- Fixed master validation & test evaluation
- Class-weighted CrossEntropyLoss training
- Epoch history tracking & logging
- Best checkpoint selection (Primary Criterion: Best Validation Macro F1-Score)
- Final test evaluation & Stage 7 metric logging
- Reproducible artifact saving (config.json, history.json, metrics.json, predictions.csv, best_checkpoint.pt)
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

from src.data.dataset import HAM10000Dataset, CLASS_TO_IDX
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


def calculate_class_weights():
    """Calculates inverse class frequency weights normalized to sum to 7.0."""
    total_samples = 10015
    num_classes = 7
    class_counts = [327, 514, 1099, 115, 1113, 6705, 142]
    
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
    Executes a complete training and evaluation run.
    """
    set_seed(config.seed)
    os.makedirs(config.run_dir, exist_ok=True)
    
    # Save config.json
    config_json_path = os.path.join(config.run_dir, 'config.json')
    config.save_json(config_json_path)
    
    # Setup Logger
    log_file = os.path.join(config.run_dir, 'train.log')
    logger = logging.getLogger(f"exp_{config.model_name}_{config.seed}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(sh)
    
    logger.info(f"Starting Experiment: Model={config.model_name} | Frac={config.fraction_pct} | Seed={config.seed}")
    logger.info(f"Device: {config.device} | Primary Criterion: Best Validation Macro F1")
    
    # Build Data Transforms
    train_transform = get_transforms(is_training=True, image_size=config.image_size, mean=config.norm_mean, std=config.norm_std)
    eval_transform = get_transforms(is_training=False, image_size=config.image_size, mean=config.norm_mean, std=config.norm_std)
    
    # Build Datasets
    train_dataset = HAM10000Dataset(config.fraction_csv, image_dirs, split='train', transform=train_transform)
    val_dataset = HAM10000Dataset(config.master_split_csv, image_dirs, split='val', transform=eval_transform)
    test_dataset = HAM10000Dataset(config.master_split_csv, image_dirs, split='test', transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    
    logger.info(f"Loaded Datasets: Train={len(train_dataset)} images | Val={len(val_dataset)} images | Test={len(test_dataset)} images")
    
    # Build Model
    device = torch.device(config.device)
    model = get_model(config.model_name, num_classes=config.num_classes).to(device)
    
    # Class-Weighted Loss & Optimizer
    class_weights = calculate_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    checkpoint_path = os.path.join(config.run_dir, 'best_checkpoint.pt')
    
    best_val_macro_f1 = -1.0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
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
        if val_f1 > best_val_macro_f1 or epoch == 1:
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
            if patience_counter >= config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Patience={config.patience})")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training Complete in {total_training_time / 60:.2f} minutes.")
    
    # Save history.json
    with open(os.path.join(config.run_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
        
    # Evaluate Best Checkpoint ONCE on master FIXED Test Set
    logger.info("\nLoading Best Checkpoint for Final Test Set Evaluation...")
    best_cp = torch.load(checkpoint_path, map_location=device)
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
    
    # Save metrics.json
    serializable_metrics = {
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
    
    # Close logger handlers to release file lock on train.log
    fh.close()
    sh.close()
    logger.removeHandler(fh)
    logger.removeHandler(sh)
    
    return serializable_metrics
