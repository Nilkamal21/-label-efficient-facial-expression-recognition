"""
Modular PyTorch Dataset and DataLoader constructor for RAF-DB (Real-world Affective Faces Database).

Maps image IDs / filenames to face images across configurable directories
and handles the 7 basic facial emotion classes.
"""

import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD

# RAF-DB 7 Basic Emotion Categories (1-indexed in RAF-DB raw files, 0-indexed in PyTorch)
CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# Mappings
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Raw RAF-DB 1-based numeric label mapping (1=Surprise ... 7=Neutral)
RAFDB_NUMERIC_MAP = {
    1: 0,  # Surprise
    2: 1,  # Fear
    3: 2,  # Disgust
    4: 3,  # Happiness
    5: 4,  # Sadness
    6: 5,  # Anger
    7: 6   # Neutral
}


class RAFDBDataset(Dataset):
    """
    Custom PyTorch Dataset for RAF-DB Facial Expression Recognition.
    """
    def __init__(self, split_csv_path, image_dirs, split='train', transform=None):
        """
        Args:
            split_csv_path (str): Path to split CSV file (or txt label file).
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
        elif image_dirs is None:
            self.image_dirs = []
        else:
            self.image_dirs = list(image_dirs)

        # 1. Load split CSV
        df = pd.read_csv(split_csv_path)

        # Filter by split if split column exists
        if 'split' in df.columns:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        # Determine image_id / filename column
        if 'image_id' in self.df.columns:
            self.id_col = 'image_id'
        elif 'filename' in self.df.columns:
            self.id_col = 'filename'
        elif 'image' in self.df.columns:
            self.id_col = 'image'
        else:
            self.id_col = self.df.columns[0]

        # Determine label column
        if 'label' in self.df.columns:
            self.label_col = 'label'
        elif 'emotion' in self.df.columns:
            self.label_col = 'emotion'
        elif 'dx' in self.df.columns:
            self.label_col = 'dx'
        else:
            self.label_col = self.df.columns[1]

        # 2. Build fast lookup map: image_id / basename -> absolute file path
        self.image_path_map = {}
        for d in self.image_dirs:
            if os.path.exists(d):
                jpg_files = (
                    glob.glob(os.path.join(d, "*.jpg"))
                    + glob.glob(os.path.join(d, "*.png"))
                    + glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)
                    + glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
                )
                for fpath in jpg_files:
                    fname = os.path.basename(fpath)
                    name_no_ext = os.path.splitext(fname)[0]
                    # Map both with and without extension
                    self.image_path_map[fname] = fpath
                    self.image_path_map[name_no_ext] = fpath
                    # Also map aligned variations (e.g. train_00001 -> train_00001_aligned)
                    if name_no_ext.endswith('_aligned'):
                        self.image_path_map[name_no_ext.replace('_aligned', '')] = fpath
                    else:
                        self.image_path_map[f"{name_no_ext}_aligned"] = fpath

        # 3. Match rows with existing images
        missing_count = 0
        valid_rows = []
        for _, row in self.df.iterrows():
            img_key = str(row[self.id_col]).strip()
            key_no_ext = os.path.splitext(img_key)[0]
            if img_key in self.image_path_map:
                valid_rows.append(row)
            elif key_no_ext in self.image_path_map:
                valid_rows.append(row)
            elif f"{key_no_ext}_aligned" in self.image_path_map:
                valid_rows.append(row)
            else:
                missing_count += 1

        self.samples = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.missing_count = missing_count

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, img_key):
        key = str(img_key).strip()
        key_no_ext = os.path.splitext(key)[0]
        if key in self.image_path_map:
            return self.image_path_map[key]
        if key_no_ext in self.image_path_map:
            return self.image_path_map[key_no_ext]
        if f"{key_no_ext}_aligned" in self.image_path_map:
            return self.image_path_map[f"{key_no_ext}_aligned"]
        raise FileNotFoundError(f"Image '{key}' not found across specified image directories.")

    def _parse_label(self, raw_label):
        """Converts raw label (string, 1-based int, or 0-based int) to 0..6 integer."""
        if isinstance(raw_label, (int, float, torch.Tensor)):
            val = int(raw_label)
            if val in RAFDB_NUMERIC_MAP:
                return RAFDB_NUMERIC_MAP[val]
            if 0 <= val < len(CLASSES):
                return val
            raise ValueError(f"Unrecognized numeric label: {val}")

        label_str = str(raw_label).strip().capitalize()
        if label_str in CLASS_TO_IDX:
            return CLASS_TO_IDX[label_str]

        # Try case-insensitive lookup
        lower_map = {k.lower(): v for k, v in CLASS_TO_IDX.items()}
        if str(raw_label).strip().lower() in lower_map:
            return lower_map[str(raw_label).strip().lower()]

        # Try int parsing
        try:
            val = int(raw_label)
            if val in RAFDB_NUMERIC_MAP:
                return RAFDB_NUMERIC_MAP[val]
            if 0 <= val < len(CLASSES):
                return val
        except ValueError:
            pass

        raise ValueError(f"Unknown RAF-DB label: '{raw_label}'. Expected one of {CLASSES}")

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_id = str(row[self.id_col]).strip()
        raw_label = row[self.label_col]

        img_path = self._resolve_image_path(img_id)
        image = Image.open(img_path).convert('RGB')
        label_idx = self._parse_label(raw_label)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return {
            'image': image_tensor,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'image_id': img_id
        }


# Alias for backward compatibility
HAM10000Dataset = RAFDBDataset
