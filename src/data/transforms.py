"""
Modular PyTorch Data Transforms for Skin Lesion Classification.

Supports:
- Models A, B, C (EfficientNet-B0, DINOv2): ImageNet normalization
- Model D (BiomedCLIP): OpenCLIP / BiomedCLIP normalization
- Training transforms (with light, biologically valid augmentations)
- Validation & Test transforms (deterministic resizing & normalization)
"""

from torchvision import transforms

# Pretrained Model Normalization Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BIOMEDCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMEDCLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_transforms(is_training=False, image_size=224, mean=None, std=None):
    """
    Returns torchvision transforms for training or evaluation.
    
    Args:
        is_training (bool): If True, includes random flips, rotation, and mild jitter.
        image_size (int): Target height and width (default 224).
        mean (list): Normalization channel means (defaults to ImageNet).
        std (list): Normalization channel stds (defaults to ImageNet).
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    transform_list = []

    # 1. Spatial Resizing
    transform_list.append(transforms.Resize((image_size, image_size)))

    # 2. Training Data Augmentations (Only applied during training)
    if is_training:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=20))
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))

    # 3. Convert PIL Image to PyTorch Tensor [C, H, W] in range [0.0, 1.0]
    transform_list.append(transforms.ToTensor())

    # 4. Apply Channel Normalization
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
