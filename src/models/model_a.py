"""
Model A: EfficientNet-B0 with Random Initialization (Trained from Scratch).

Architecture: EfficientNet-B0
Pretrained Weights: None (Random Kaiming/He Normal Initialization)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for HAM10000)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model_a(num_classes=7):
    """
    Builds Model A: EfficientNet-B0 from scratch with random weights.
    
    Args:
        num_classes (int): Number of target diagnostic categories (default 7).
        
    Returns:
        torch.nn.Module: EfficientNet-B0 with 7-class linear classifier and random weights.
    """
    # Instantiate EfficientNet-B0 with weights=None (Random Initialization)
    model = efficientnet_b0(weights=None)
    
    # EfficientNet-B0 classifier structure in torchvision:
    # model.classifier is nn.Sequential(
    #     Dropout(p=0.2, inplace=True),
    #     Linear(in_features=1280, out_features=1000, bias=True)
    # )
    in_features = model.classifier[1].in_features  # 1280
    
    # Replace the final linear layer with 7 output logits
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model
