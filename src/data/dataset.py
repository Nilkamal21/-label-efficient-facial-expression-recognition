"""
Modular PyTorch Dataset and DataLoader constructor for HAM10000.

Reads ham10000_master_split.csv and maps image_id to image files across
configurable image directories (e.g., HAM10000_images_part_1 and part_2).
"""

import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD

# Stable, deterministic class label mapping (alphabetical order)
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    """
    Custom PyTorch Dataset for HAM10000 Skin Lesion Classification.
    """
    def __init__(self, split_csv_path, image_dirs, split='train', transform=None):
        """
        Args:
            split_csv_path (str): Path to master split CSV file.
            image_dirs (str or list): Path or list of paths containing .jpg images.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): PyTorch torchvision transform.
        """
        self.split_csv_path = split_csv_path
        self.split = split
        self.transform = transform
        
        # Ensure image_dirs is a list
        if isinstance(image_dirs, str):
            self.image_dirs = [image_dirs]
        else:
            self.image_dirs = image_dirs
            
        # 1. Load split CSV
        df = pd.read_csv(split_csv_path)
        
        # Filter by split if column exists
        if 'split' in df.columns:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
            
        # 2. Build fast lookup map: image_id -> absolute file path
        self.image_path_map = {}
        for d in self.image_dirs:
            if os.path.exists(d):
                # Search directly or recursively for .jpg files
                jpg_files = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)
                for fpath in jpg_files:
                    fname = os.path.basename(fpath)
                    img_id = os.path.splitext(fname)[0]
                    self.image_path_map[img_id] = fpath

        # 3. Verify images exist for rows
        missing_count = 0
        valid_rows = []
        for _, row in self.df.iterrows():
            img_id = row['image_id']
            if img_id in self.image_path_map:
                valid_rows.append(row)
            else:
                missing_count += 1
                
        self.samples = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.missing_count = missing_count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_id = row['image_id']
        dx_label = row['dx']
        
        # Get image path
        img_path = self.image_path_map[img_id]
        
        # Load image in RGB
        image = Image.open(img_path).convert('RGB')
        
        # Map class label string to integer index
        label_idx = CLASS_TO_IDX[dx_label]
        
        # Apply torchvision transforms if provided
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image
            
        return {
            'image': image_tensor,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'image_id': img_id,
            'lesion_id': row['lesion_id'],
            'dx': dx_label
        }


def create_dataloaders(
    split_csv_path,
    image_dirs,
    batch_size=32,
    num_workers=0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    image_size=224
):
    """
    Helper function to create Train, Val, and Test DataLoaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, datasets_dict)
    """
    train_transform = get_transforms(is_training=True, image_size=image_size, mean=mean, std=std)
    eval_transform = get_transforms(is_training=False, image_size=image_size, mean=mean, std=std)
    
    train_dataset = HAM10000Dataset(split_csv_path, image_dirs, split='train', transform=train_transform)
    val_dataset = HAM10000Dataset(split_csv_path, image_dirs, split='val', transform=eval_transform)
    test_dataset = HAM10000Dataset(split_csv_path, image_dirs, split='test', transform=eval_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffling enabled for training
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Deterministic (no shuffle) for evaluation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # Deterministic (no shuffle) for test
        num_workers=num_workers,
        pin_memory=True
    )
    
    datasets_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    return train_loader, val_loader, test_loader, datasets_dict
