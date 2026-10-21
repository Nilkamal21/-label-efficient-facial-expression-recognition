"""
Modular PyTorch Data Transforms for Facial Expression Recognition (RAF-DB).

Supports:
- Models A, B, C (ResNet/EfficientNet, DINOv2): ImageNet normalization
- Model D (ArcFace): Standard Face normalization (mean=0.5, std=0.5 -> [-1, 1])
- Training transforms (face-appropriate augmentations: horizontal flip, mild rotation, color jitter)
- Validation & Test transforms (deterministic resizing & normalization)
"""

from torchvision import transforms

# Pretrained Model Normalization Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard ArcFace / InsightFace Normalization (scales [0, 1] to [-1, 1])
ARCFACE_MEAN = [0.5, 0.5, 0.5]
ARCFACE_STD = [0.5, 0.5, 0.5]


def get_transforms(is_training=False, image_size=224, mean=None, std=None):
    """
    Returns torchvision transforms for training or evaluation on facial images.
    
    Args:
        is_training (bool): If True, applies face-appropriate augmentations (horizontal flip,
                            mild rotation, color jitter). Excludes vertical flip.
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
    # Face-appropriate augmentations (horizontal flip, mild rotation, mild color jitter)
    if is_training:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=10))
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))

    # 3. Convert PIL Image to PyTorch Tensor [C, H, W] in range [0.0, 1.0]
    transform_list.append(transforms.ToTensor())

    # 4. Apply Channel Normalization
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
