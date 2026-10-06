"""
Model D: BiomedCLIP (Biomedical Vision-Language Foundation Model).

Architecture: Vision Transformer Base with 16x16 patch size (ViT-B/16)
Pretrained Weights: PubMed 15M Image-Text Pairs (Microsoft BiomedCLIP)
Input Shape: [Batch_Size, 3, 224, 224]
Output Shape: [Batch_Size, 7] (Raw classification logits for HAM10000)
"""

import torch
import torch.nn as nn
import open_clip


class BiomedCLIPClassifier(nn.Module):
    """
    BiomedCLIP ViT-B/16 visual classification model for HAM10000 skin lesion classification.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # Load BiomedCLIP multimodal model from Hugging Face via open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Extract visual encoder only (text encoder is not used for image classification)
        self.visual_encoder = clip_model.visual
        
        # Verify actual visual embedding dimension (BiomedCLIP projection dimension is 512)
        self.embed_dim = 512
        
        # 7-class linear classification head
        self.classifier = nn.Linear(in_features=self.embed_dim, out_features=num_classes)

    def forward(self, x):
        # Extract 512-dim visual embedding vector from BiomedCLIP vision backbone
        # In open_clip ViT backbones, model.visual(x) outputs the projected image embedding
        image_embeddings = self.visual_encoder(x)  # [Batch_Size, 512]
        
        # Compute 7 class logits
        logits = self.classifier(image_embeddings)  # [Batch_Size, 7]
        return logits


def build_model_d(num_classes=7):
    """
    Factory function to build Model D (BiomedCLIP ViT-B/16).
    
    Args:
        num_classes (int): Number of target diagnostic categories (default 7).
        
    Returns:
        torch.nn.Module: BiomedCLIP model with 7-class classification head.
    """
    return BiomedCLIPClassifier(num_classes=num_classes)
