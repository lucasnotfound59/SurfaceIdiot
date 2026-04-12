"""
SurfaceIdiot - GraspPolicy Model

Architecture:
    Visual branch  : MobileNetV3-Small (pretrained ImageNet) → 128-dim feature
    Temporal branch: Small Transformer over joint history     → 64-dim feature
    Fusion         : MLP → 5-dim joint angle prediction (sigmoid, 0-1)

The Transformer for the temporal branch lets the model attend to which past
frames matter most, which works better than a flat MLP for variable-length
patterns (e.g. fast vs slow grasp motions).

Total params: ~2.5M — comfortable for a 4080S.
"""

import torch
import torch.nn as nn
import torchvision.models as models

JOINT_DIM = 5  # thumb, index, middle, ring, pinky


# ─── Temporal encoder (joint history) ───────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Encodes a sequence of joint angles using a small Transformer.

    Input : (B, H, joint_dim)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        joint_dim: int = JOINT_DIM,
        history_len: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        out_dim: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(joint_dim, d_model)

        # Learned positional embedding (one per history step)
        self.pos_embed = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,    # Pre-LN → more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Pool over sequence with a learned CLS-style aggregation
        self.pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),          # attention weights over history steps
        )
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, joint_dim)
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1)]  # (B, H, d_model)
        x = self.transformer(x)                                   # (B, H, d_model)

        # Weighted sum over history
        attn = self.pool(x)   # (B, H, 1)
        ctx  = (attn * x).sum(dim=1)   # (B, d_model)
        return self.out_proj(ctx)       # (B, out_dim)


# ─── Visual encoder ──────────────────────────────────────────────────────────

class VisualEncoder(nn.Module):
    """
    MobileNetV3-Small pretrained on ImageNet, adapted to output a compact vector.

    Input : (B, 3, 224, 224)
    Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 128, freeze_backbone: bool = False):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # Remove the classification head; keep features + adaptive pool
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small outputs 576 channels after pool
        backbone_out = 576
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, out_dim),
            nn.SiLU(),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # (B, 576, 7, 7)
        x = self.pool(x)        # (B, 576, 1, 1)
        return self.proj(x)     # (B, out_dim)


# ─── Full policy network ─────────────────────────────────────────────────────

class GraspPolicy(nn.Module):
    """
    Behavior cloning policy for hand grasping.

    Args:
        history_len     : number of past joint frames fed as context
        joint_dim       : number of finger DOFs (default 5)
        vis_dim         : visual feature size
        temp_dim        : temporal feature size
        freeze_backbone : freeze MobileNet weights (useful early in training)
    """

    def __init__(
        self,
        history_len: int = 6,
        joint_dim: int = JOINT_DIM,
        vis_dim: int = 128,
        temp_dim: int = 64,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.visual_encoder   = VisualEncoder(out_dim=vis_dim, freeze_backbone=freeze_backbone)
        self.temporal_encoder = TemporalEncoder(
            joint_dim=joint_dim,
            history_len=history_len,
            out_dim=temp_dim,
        )

        fused = vis_dim + temp_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, joint_dim),
            nn.Sigmoid(),   # output in [0, 1] — maps directly to servo command
        )

    def forward(
        self, image: torch.Tensor, joint_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image         : (B, 3, H, W)
            joint_history : (B, history_len, joint_dim)
        Returns:
            predicted joint angles : (B, joint_dim)  in [0, 1]
        """
        vis  = self.visual_encoder(image)
        temp = self.temporal_encoder(joint_history)
        return self.fusion(torch.cat([vis, temp], dim=1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick model inspection ──────────────────────────────────────────────────

if __name__ == "__main__":
    model = GraspPolicy(history_len=6)
    print(f"Trainable parameters: {model.count_parameters():,}")

    B = 4
    img  = torch.randn(B, 3, 224, 224)
    hist = torch.rand(B, 6, 5)
    out  = model(img, hist)
    print(f"Output shape: {tuple(out.shape)}")   # (4, 5)
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
