"""
Model A: ResNet-50 with Random Initialization (Trained from Scratch).

Architecture: ResNet-50
Pretrained Weights: None (Random Initialization)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for RAF-DB 7 basic emotions)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50


def build_model_a(num_classes=7):
    """
    Builds Model A: ResNet-50 from scratch with random weights.
    
    Args:
        num_classes (int): Number of target emotion categories (default 7).
        
    Returns:
        torch.nn.Module: ResNet-50 with 7-class linear classifier and random weights.
    """
    # Instantiate ResNet-50 with weights=None (Random Initialization)
    model = resnet50(weights=None)
    in_features = model.fc.in_features  # 2048
    
    # Replace the final linear head with 7 output logits
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model
