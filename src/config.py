"""
Centralized Configuration System for HAM10000 Research Experiments.

Manages all hyperparameters, model selection, normalization specs, paths,
loss configurations, and reproducibility settings.
"""

import os
import json
import torch

from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BIOMEDCLIP_MEAN,
    BIOMEDCLIP_STD
)


class ExperimentConfig:
    """
    Experiment Configuration container for HAM10000 training and evaluation.
    """
    def __init__(
        self,
        model_name='model_a',        # 'model_a', 'model_b', 'model_c', 'model_d'
        fraction_pct='100%',         # '100%', '75%', '50%', '30%', '20%', '10%', '5%', '1%'
        seed=42,                     # 42, 43, 44
        batch_size=32,               # Mini-batch size
        learning_rate=1e-4,          # Learning rate
        weight_decay=1e-4,           # Weight decay for AdamW
        num_epochs=15,               # Total training epochs per run
        patience=5,                  # Early stopping patience epochs
        image_size=224,              # Input image height & width
        num_classes=7,               # Number of diagnostic categories
        device=None,                 # 'cuda' or 'cpu'
        project_root=None,           # Base directory path
        experiments_dir=None         # Output directory for results
    ):
        self.model_name = model_name
        self.fraction_pct = fraction_pct
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Device auto-detection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Paths management
        if project_root is None:
            self.project_root = r"C:\Users\adhik\OneDrive\Desktop\Label-Efficient Skin Lesion Classification"
        else:
            self.project_root = project_root
            
        if experiments_dir is None:
            self.experiments_dir = os.path.join(self.project_root, 'experiments')
        else:
            self.experiments_dir = experiments_dir
            
        self.master_split_csv = os.path.join(self.project_root, 'data', 'splits', 'ham10000_master_split.csv')
        
        # Fraction CSV path
        frac_filename = f"train_{fraction_pct.replace('%', '')}.csv"
        self.fraction_csv = os.path.join(self.project_root, 'data', 'splits', 'fractions', f"seed_{seed}", frac_filename)
        
        # Normalization selection per model
        if self.model_name == 'model_d':
            self.norm_mean = BIOMEDCLIP_MEAN
            self.norm_std = BIOMEDCLIP_STD
            self.norm_type = 'BiomedCLIP'
        else:
            self.norm_mean = IMAGENET_MEAN
            self.norm_std = IMAGENET_STD
            self.norm_type = 'ImageNet'
            
        # Specific output directory for this run: experiments/{model_name}/{fraction_pct}/seed_{seed}/
        frac_dir_name = f"fraction_{fraction_pct.replace('%', '')}"
        self.run_dir = os.path.join(self.experiments_dir, self.model_name, frac_dir_name, f"seed_{seed}")
        
    def to_dict(self):
        """Converts configuration to a serializable dictionary."""
        return {
            'model_name': self.model_name,
            'fraction_pct': self.fraction_pct,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'device': self.device,
            'norm_type': self.norm_type,
            'norm_mean': self.norm_mean,
            'norm_std': self.norm_std,
            'master_split_csv': self.master_split_csv,
            'fraction_csv': self.fraction_csv,
            'run_dir': self.run_dir,
            'checkpoint_selection_criterion': 'Best Validation Macro F1-Score'
        }

    def save_json(self, filepath):
        """Saves configuration dictionary as JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
