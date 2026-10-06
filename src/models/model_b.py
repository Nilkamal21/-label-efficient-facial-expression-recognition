"""
Model B: ImageNet-Pretrained EfficientNet-B0.

Architecture: EfficientNet-B0
Pretrained Weights: ImageNet-1k (EfficientNet_B0_Weights.DEFAULT)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for HAM10000)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model_b(num_classes=7):
    """
    Builds Model B: ImageNet-Pretrained EfficientNet-B0 adapted for HAM10000.
    
    Args:
        num_classes (int): Number of target diagnostic categories (default 7).
        
    Returns:
        torch.nn.Module: EfficientNet-B0 initialized with ImageNet weights and a new 7-class head.
    """
    # Load EfficientNet-B0 with pretrained ImageNet-1k weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features  # 1280
    
    # Replace final 1000-class linear head with 7-class linear head
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model
