"""
=============================================================================
BRAIN TUMOR CLASSIFICATION — TEAM 8
File: brain_tumor_model.py
Role: Attention-Gated ResNet-34 Architecture Definition
=============================================================================

Author      : Team 8
Project     : Brain Tumor Classification using Deep Learning & Explainable AI
Architecture: Attention-Gated ResNet-34 + Monte Carlo Uncertainty Estimation

Description :
    Defines two core components:

    1. SpatialAttentionGate (Novel Module):
       A lightweight convolutional attention gate inserted between conv4_x and
       conv5_x feature maps of ResNet-34. Suppresses anatomically irrelevant
       activations (skull, orbit, background) and channels computational focus
       directly onto intracranial pathological tissue.

    2. BrainTumorClassifier (Main Model):
       Wraps a pre-trained ImageNet ResNet-34 backbone. Intercepts layer4
       output (512 channels, 7×7 spatial resolution) and injects the custom
       SpatialAttentionGate before global pooling and 2-layer classification
       head with Monte Carlo Dropout.

    Novelties:
        - Spatial Attention Gate: Acts as a neurological lens that isolates
          tumor-bearing parenchymal regions.
        - Monte Carlo Dropout Inference: Retaining active Bernoulli dropout
          during test-time enables stochastic uncertainty quantification via
          repeated forward passes, producing diagnostic confidence intervals.

Layer Count Summary:
    ┌──────────────────────────┬────────────────────┬──────────────┐
    │ Layer Group              │ Residual Blocks    │ Output Shape │
    ├──────────────────────────┼────────────────────┼──────────────┤
    │ conv1 + bn1 + relu + pool│ –                  │ 56×56 ×64    │
    │ conv2_x (layer1)         │ 3 Residual Blocks  │ 56×56 ×64    │
    │ conv3_x (layer2)         │ 4 Residual Blocks  │ 28×28 ×128   │
    │ conv4_x (layer3)         │ 6 Residual Blocks  │ 14×14 ×256   │
    │ conv5_x (layer4)         │ 3 Residual Blocks  │  7×7  ×512   │
    │ Spatial Attention Gate   │ [NOVEL INSERTION]  │  7×7  ×512   │
    │ AdaptiveAvgPool2d        │ –                  │  1×1  ×512   │
    │ FC-256 + BN + ReLU       │ –                  │  256         │
    │ Dropout(0.4) [MC-Active] │ –                  │  256         │
    │ FC-4 (output logits)     │ –                  │    4         │
    └──────────────────────────┴────────────────────┴──────────────┘
    Total backbone layers: 34 (matching standard ResNet-34 specification)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL MODULE 1: SPATIAL ATTENTION GATE
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttentionGate(nn.Module):
    """
    Lightweight Spatial Channel-Squeeze-and-Excitation Attention Gate.

    Mechanism:
        Given an input feature map F ∈ R^{C × H × W}, the gate computes a
        scalar spatial relevance map M_s ∈ R^{1 × H × W} using two 1×1
        convolutional layers with a bottleneck compression ratio of 1/8:

            M_s(F) = σ( Conv_{1×1} ( ReLU( BN( Conv_{1×1}(F) ) ) ) )

        The attended feature map is then: F_out = F ⊗ M_s(F)

    Clinical Interpretation:
        This mechanism suppresses activations in non-relevant regions such as
        skull bone, orbital fat, and scalp tissue while amplifying response to
        irregular necrotic cores, ring-enhancing margins, and peritumoral edema.

    Args:
        in_channels (int): Number of input feature channels (512 for ResNet-34 layer4).
    """

    def __init__(self, in_channels: int):
        super(SpatialAttentionGate, self).__init__()

        # Bottleneck 1×1 Conv: C → C/8 (lightweight compression of channel info)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            # Projection back to single-channel spatial attention map
            nn.Conv2d(in_channels // 8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()                   # Squash to [0,1] — soft spatial mask
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W].

        Returns:
            tuple:
                - attended_features (torch.Tensor): Attention-weighted features [B, C, H, W].
                - attention_map     (torch.Tensor): Spatial soft attention mask [B, 1, H, W].
        """
        attention_map      = self.gate_net(x)          # [B, 1, H, W]
        attended_features  = x * attention_map          # Element-wise spatial masking
        return attended_features, attention_map


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL: ATTENTION-GATED RESNET-34 BRAIN TUMOR CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumorClassifier(nn.Module):
    """
    Attention-Gated ResNet-34 for 4-Class Brain Tumor Classification.

    Architecture:
        Backbone  : Pre-trained ResNet-34 (ImageNet IMAGENET1K_V1 weights)
                    Feature extractor frozen or fine-tuned depending on strategy.
        Attention : Custom SpatialAttentionGate injected after layer4 (conv5_x).
        Pooling   : Adaptive Global Average Pooling → 1×1×512 feature vector.
        Classifier: 512 → 256 (FC + BN + ReLU) → Dropout(0.4) → 4 (logits).

    Monte Carlo Uncertainty:
        During inference (model.eval()), standard behaviour disables Dropout.
        To enable Bayesian uncertainty estimation, call mc_inference(x, n_passes=10)
        which explicitly re-activates Dropout during n forward passes.
        Variance of softmax outputs across passes quantifies predictive uncertainty.

    Args:
        num_classes (int): Number of classification outputs (default: 4).
        dropout_p   (float): Bernoulli dropout probability (default: 0.4).
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.4):
        super(BrainTumorClassifier, self).__init__()

        # ── Load pre-trained ResNet-34 backbone ───────────────────────────────
        # ImageNet pre-trained weights provide rich low-level edge detectors,
        # texture analyzers, and shape recognizers that transfer effectively
        # to medical imaging feature extraction.
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # ── Build sequential feature extractor from ResNet-34 stages ──────────
        # We extract layer-by-layer for Grad-CAM compatibility (target_layers).
        # This preserves spatial feature map accessibility required by Grad-CAM.
        self.stem    = nn.Sequential(
            backbone.conv1,    # [B, 64, 112, 112]  — 7×7 stride-2 conv
            backbone.bn1,      # Batch normalization
            backbone.relu,     # ReLU activation
            backbone.maxpool   # [B, 64, 56, 56]    — 3×3 stride-2 max pool
        )
        self.layer1  = backbone.layer1  # [B, 64,  56, 56]  — 3 residual blocks
        self.layer2  = backbone.layer2  # [B, 128, 28, 28]  — 4 residual blocks
        self.layer3  = backbone.layer3  # [B, 256, 14, 14]  — 6 residual blocks
        self.layer4  = backbone.layer4  # [B, 512,  7,  7]  — 3 residual blocks

        # ── Novel Spatial Attention Gate ──────────────────────────────────────
        # Injected AFTER layer4 — intercepts the highest-level semantic feature
        # maps just before global pooling. This ensures the gate acts on rich,
        # semantically meaningful tumour representations.
        self.attention_gate = SpatialAttentionGate(in_channels=512)

        # ── Global Average Pooling ────────────────────────────────────────────
        # Reduces spatial dimensions from [B, 512, 7, 7] → [B, 512, 1, 1]
        # More robust than max pooling for classification generalization.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classification Head ───────────────────────────────────────────────
        # Two-layer MLP: 512 → 256 (with BN + ReLU) → num_classes
        # Batch Normalization stabilizes gradient flow through the dense layers.
        # Monte Carlo Dropout (p=dropout_p) is kept active during inference for
        # stochastic uncertainty quantification.
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),       # MC-Dropout: remains active at inference
            nn.Linear(256, num_classes)    # Raw logits output
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Standard forward pass through the complete classification pipeline.

        Args:
            x                (torch.Tensor): Input batch [B, 3, 224, 224].
            return_attention (bool)        : If True, also return spatial attention map.

        Returns:
            logits      (torch.Tensor): Class logits [B, num_classes].
            attn_map    (torch.Tensor): Spatial attention map [B, 1, H, W] — only if return_attention=True.
        """
        # Stage 1: Initial stem (conv + BN + ReLU + MaxPool)
        x = self.stem(x)

        # Stage 2: Residual convolution stages (conv2_x through conv5_x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)     # Output: [B, 512, 7, 7]

        # Stage 3: Spatial Attention Gating (Novel Clinical Contribution)
        x, attn_map = self.attention_gate(x)  # Attended: [B, 512, 7, 7], Mask: [B, 1, 7, 7]

        # Stage 4: Global Average Pooling → flatten
        x = self.global_avg_pool(x)            # [B, 512, 1, 1]
        x = torch.flatten(x, start_dim=1)      # [B, 512]

        # Stage 5: Classification head (FC → BN → ReLU → Dropout → FC)
        logits = self.classifier(x)            # [B, num_classes]

        if return_attention:
            return logits, attn_map
        return logits

    def mc_inference(self, x: torch.Tensor, n_passes: int = 10):
        """
        Monte Carlo Dropout Inference for Bayesian Uncertainty Quantification.

        Mechanism:
            During standard model.eval(), PyTorch disables Dropout layers.
            This method explicitly re-enables Dropout for n_passes forward
            iterations through the input batch, collecting stochastic softmax
            probability distributions. The variance across passes measures
            predictive uncertainty (epistemic uncertainty proxy).

            Uncertainty formula:
                σ²(x) = (1/M) Σ_{m=1}^{M} (ŷ_m - ȳ)²

            A high σ² triggers a clinical alert: the model is uncertain and
            human radiological review is mandatory before patient intervention.

        Args:
            x        (torch.Tensor): Preprocessed input tensor [B, 3, 224, 224].
            n_passes (int)         : Number of stochastic forward passes (default: 10).

        Returns:
            dict containing:
                'mean_probs'   : Mean softmax probabilities across passes [B, num_classes].
                'variance'     : Predictive variance across passes [B, num_classes].
                'predicted_cls': Argmax of mean probability [B].
                'uncertainty'  : Scalar uncertainty flag — True if variance > 0.05.
        """
        # Force all Dropout submodules to active training mode for stochasticity
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x)
                probs  = F.softmax(logits, dim=1)   # [B, num_classes]
                all_probs.append(probs.unsqueeze(0)) # [1, B, num_classes]

        # Stack into [n_passes, B, num_classes]
        all_probs_tensor = torch.cat(all_probs, dim=0)

        mean_probs    = all_probs_tensor.mean(dim=0)       # [B, num_classes]
        variance      = all_probs_tensor.var(dim=0)        # [B, num_classes]
        predicted_cls = mean_probs.argmax(dim=1)           # [B]
        max_variance  = variance.max(dim=1).values         # [B] — worst-case feature variance

        # Clinical alert threshold: σ² > 0.05 indicates ambiguous prediction
        uncertainty_flag = (max_variance > 0.05)

        return {
            'mean_probs'    : mean_probs,
            'variance'      : variance,
            'predicted_cls' : predicted_cls,
            'uncertainty'   : uncertainty_flag
        }


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE VALIDATION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Attention-Gated ResNet-34 Architecture Validation — Team 8")
    print("=" * 65)

    model   = BrainTumorClassifier(num_classes=4, dropout_p=0.4)
    x_dummy = torch.randn(2, 3, 224, 224)    # Simulated mini-batch of 2 MRI scans

    # ── Standard forward pass ──────────────────────────────────────────────
    logits, attn_map = model(x_dummy, return_attention=True)
    print(f"\n[Standard Forward Pass]")
    print(f"  Input shape           : {list(x_dummy.shape)}")
    print(f"  Output logits shape   : {list(logits.shape)}")
    print(f"  Attention map shape   : {list(attn_map.shape)}")

    # ── Monte Carlo inference pass ─────────────────────────────────────────
    model.eval()
    mc_results = model.mc_inference(x_dummy, n_passes=10)
    print(f"\n[Monte Carlo Uncertainty (10 passes)]")
    print(f"  Mean softmax probs    : {mc_results['mean_probs'].numpy().round(4)}")
    print(f"  Predictive variance   : {mc_results['variance'].numpy().round(6)}")
    print(f"  Predicted classes     : {mc_results['predicted_cls'].tolist()}")
    print(f"  Uncertainty flag      : {mc_results['uncertainty'].tolist()}")

    # ── Parameter count ────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Summary]")
    print(f"  Total parameters      : {total_params:,}")
    print(f"  Trainable parameters  : {trainable_params:,}")
    print(f"\nArchitecture validation complete.")
