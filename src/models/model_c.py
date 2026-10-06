"""
Model C: DINOv2 (Self-Supervised Vision Transformer - ViT-S/14).

Architecture: Vision Transformer Small with 14x14 patch size (dinov2_vits14)
Pretrained Weights: Self-Supervised LVD-142M (Meta AI DINOv2)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for HAM10000)
"""

import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-S/14 classification model for HAM10000 skin lesion classification.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # Load DINOv2 Small backbone from Meta AI PyTorch Hub repository
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # DINOv2 ViT-S/14 outputs a 384-dimensional CLS feature vector
        self.embed_dim = 384
        
        # 7-class linear classification head
        self.classifier = nn.Linear(in_features=self.embed_dim, out_features=num_classes)

    def forward(self, x):
        # Extract 384-dim CLS feature representation from DINOv2 backbone
        cls_representation = self.backbone(x)  # [Batch_Size, 384]
        
        # Compute 7 class logits
        logits = self.classifier(cls_representation)  # [Batch_Size, 7]
        return logits


def build_model_c(num_classes=7):
    """
    Factory function to build Model C (DINOv2 ViT-S/14).
    
    Args:
        num_classes (int): Number of target diagnostic categories (default 7).
        
    Returns:
        torch.nn.Module: DINOv2 ViT-S/14 model with 7-class classification head.
    """
    return DINOv2Classifier(num_classes=num_classes)
