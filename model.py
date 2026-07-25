import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class SpatialAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionGate, self).__init__()
        # Lightweight 1x1 conv bottleneck for spatial attention
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn_map = self.conv(x)
        return x * attn_map, attn_map

class AdvancedTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedTumorCNN, self).__init__()
        
        # Load standard pre-trained ResNet-34
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Extract features up to layer4
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
        # The output of ResNet34 layer4 has 512 channels
        self.attention_gate = SpatialAttentionGate(in_channels=512)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        feat = self.features(x)
        
        # Apply Spatial Attention Gate
        feat_attended, attn_map = self.attention_gate(feat)
        
        out = self.global_pool(feat_attended)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        
        if return_attention:
            return logits, attn_map
        return logits

if __name__ == "__main__":
    model = AdvancedTumorCNN(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    logits, attn_map = model(x, return_attention=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention map shape: {attn_map.shape}")
