"""Attention modules for Y-ZA YOLOv8n: Efficient Channel Attention (ECA) and Global Context (GC)."""
import math

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Global average pooling, then a 1D convolution across the channel dimension
    (no dimensionality reduction), then a sigmoid gate on each channel.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                                  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))    # (B, 1, C)
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))   # (B, C, 1, 1)
        return x * y


class GlobalContext(nn.Module):
    """Global Context block (Cao et al., 2019).

    1x1 conv + softmax over all spatial positions gives a global attention map,
    which pools the features into one context vector. A bottleneck transform
    (1x1 conv, LayerNorm, ReLU, 1x1 conv) is then added back to every position.
    """

    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 8)
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.LayerNorm([mid, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feats = x.view(b, c, h * w).unsqueeze(1)                  # (B, 1, C, HW)
        mask = self.softmax(self.conv_mask(x).view(b, 1, h * w))  # (B, 1, HW)
        context = torch.matmul(feats, mask.unsqueeze(-1))         # (B, 1, C, 1)
        context = context.view(b, c, 1, 1)
        return x + self.transform(context)


def add_attention(model, first_neck_index=10):
    """Append ECA then GC after every C2f layer in the neck of a YOLOv8 DetectionModel.

    YOLOv8n has four neck C2f layers (indices 12, 15, 18, 21). Each one is wrapped as
    nn.Sequential(C2f, ECA, GC). Shapes do not change, so the Detect head is untouched.
    """
    layers = model.model
    wrapped_indices = []
    for i, m in enumerate(layers):
        if i >= first_neck_index and isinstance(m, C2f):
            c = m.cv2.conv.out_channels
            wrapped = nn.Sequential(m, ECA(c), GlobalContext(c))
            # attributes the YOLO forward pass expects on every layer
            wrapped.f, wrapped.i, wrapped.type = m.f, m.i, "C2f+ECA+GC"
            wrapped.np = sum(p.numel() for p in wrapped.parameters())
            layers[i] = wrapped
            wrapped_indices.append(i)
    return wrapped_indices
