"""Visers Nano PyTorch architecture (`mobilenetv4_conv_small_050` trunk + MRL head) and BN folder."""
from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "mobilenetv4_conv_small_050.e3000_r224_in1k"


class TrunkMRL(nn.Module):
    """Pretrained `features_only` trunk + global average pooling + 256D MRL projection."""

    def __init__(self, model_name: str = MODEL_NAME, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            trunk_dim = int(self.backbone(dummy)[-1].shape[1])
        self.trunk_dim = trunk_dim
        self.head = nn.Linear(trunk_dim, 256)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[-1].mean(dim=(-2, -1)).float()

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(x), dim=-1)


def fold_all_bn(module: nn.Module) -> None:
    """Fold all `Conv2d + BatchNorm2d` pairs in-place (preserving attached activations)."""
    for mod in module.modules():
        children = list(mod.named_children())
        for i in range(len(children) - 1):
            n1, c1 = children[i]
            n2, c2 = children[i + 1]
            if isinstance(c1, nn.Conv2d) and isinstance(c2, nn.BatchNorm2d):
                setattr(mod, n1, nn.utils.fuse_conv_bn_eval(c1, c2))
                repl = getattr(c2, "act", None)
                setattr(mod, n2, repl if repl is not None else nn.Identity())
        for conv_attr, bn_attr in (
            ("conv_pw", "bn1"),
            ("conv_dw", "bn2"),
            ("conv_pwl", "bn3"),
            ("conv_exp", "bn1"),
        ):
            if hasattr(mod, conv_attr) and hasattr(mod, bn_attr):
                c1, c2 = getattr(mod, conv_attr), getattr(mod, bn_attr)
                if isinstance(c1, nn.Conv2d) and isinstance(c2, nn.BatchNorm2d):
                    setattr(mod, conv_attr, nn.utils.fuse_conv_bn_eval(c1, c2))
                    repl = getattr(c2, "act", None)
                    setattr(mod, bn_attr, repl if repl is not None else nn.Identity())
