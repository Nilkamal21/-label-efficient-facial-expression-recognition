"""
Model D: ArcFace (Domain-Specific Face Recognition Pretrained Representation).

Architecture: iResNet-50 / ArcFace
Pretrained Weights: ArcFace trained on large-scale face datasets (MS1MV2 / Glint360k)
Input Shape: [Batch_Size, 3, 224, 224] (or [Batch_Size, 3, 112, 112])
Output Shape: [Batch_Size, 7] (Raw classification logits for RAF-DB 7 basic emotions)
"""

import os
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0.0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


ARCFACE_R50_WEIGHT_URLS = [
    # High-reliability HuggingFace CDN mirror
    "https://huggingface.co/AIRI-Institute/StyleFeatureEditor/resolve/main/pretrained_models/iresnet50-7f187506.pth",
    # Legacy mirror
    "https://sota.nizhib.ai/pytorch-insightface/iresnet50-7f187506.pth"
]


class ArcFaceClassifier(nn.Module):
    """
    ArcFace iResNet-50 visual representation model for RAF-DB 7-class facial emotion classification.
    Extracts a 512-dimensional facial embedding vector and classifies into 7 emotion logits.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = IResNet(IBasicBlock, [3, 4, 14, 3], dropout=0.0, num_features=512)

        if pretrained:
            loaded = False
            for url in ARCFACE_R50_WEIGHT_URLS:
                try:
                    print(f"[INFO] Downloading ArcFace pretrained weights from: {url}")
                    state_dict = torch.hub.load_state_dict_from_url(
                        url,
                        progress=True,
                        map_location='cpu'
                    )
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print("[INFO] Successfully loaded pretrained ArcFace iResNet-50 weights.")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[WARNING] Failed downloading from {url}: {e}")

            if not loaded:
                print("[WARNING] Could not download ArcFace pretrained weights from any source. Initializing with random weights.")

        self.embed_dim = 512
        # Classification head for 7 basic emotion classes
        self.classifier = nn.Linear(in_features=self.embed_dim, out_features=num_classes)

    def forward(self, x):
        # Extract 512-dim face embedding
        face_features = self.backbone(x)
        # Compute 7 emotion logits
        logits = self.classifier(face_features)
        return logits


def build_model_d(num_classes=7, pretrained=True):
    """
    Factory function to build Model D (ArcFace iResNet-50).
    
    Args:
        num_classes (int): Number of emotion categories (default 7).
        pretrained (bool): Whether to load ArcFace face-recognition pretrained weights.
        
    Returns:
        torch.nn.Module: ArcFace model with 7-class classification head.
    """
    return ArcFaceClassifier(num_classes=num_classes, pretrained=pretrained)
