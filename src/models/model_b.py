"""
Model B: ImageNet-Pretrained ResNet-50.

Architecture: ResNet-50
Pretrained Weights: ImageNet-1k (ResNet50_Weights.DEFAULT)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for RAF-DB 7 basic emotions)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model_b(num_classes=7):
    """
    Builds Model B: ImageNet-Pretrained ResNet-50 adapted for RAF-DB.
    
    Args:
        num_classes (int): Number of target emotion categories (default 7).
        
    Returns:
        torch.nn.Module: ResNet-50 initialized with ImageNet weights and a new 7-class head.
    """
    # Load ResNet-50 with pretrained ImageNet-1k weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    in_features = model.fc.in_features  # 2048
    
    # Replace final 1000-class linear head with 7-class linear head
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model
