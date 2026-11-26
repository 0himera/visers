"""Export folded W8 Visers Nano checkpoint to `VSN1` binary for Rust `.rodata` baking."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np
import timm
import torch
from torch import nn

from model import MODEL_NAME, TrunkMRL, fold_all_bn


def quantize_per_channel_i8(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    w = weight.detach().cpu().float()
    dims = tuple(range(1, w.ndim))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / 127.0
    q = (w / scale).round().clamp(-127.0, 127.0)
    max_diff = float((w - q * scale).abs().max().item())
    if max_diff > 1e-5:
        raise RuntimeError(f"Weight is not W8-quantized (max_diff={max_diff})")
    return q.to(torch.int8).numpy(), scale.reshape(-1).numpy().astype("<f4")


def collect_ops(model: TrunkMRL) -> list[dict]:
    ops = []
    bb = model.backbone

    def add_conv(conv: nn.Conv2d, relu: bool, save_skip: bool = False, add_skip: bool = False) -> None:
        dw = conv.groups == conv.in_channels and conv.groups == conv.out_channels
        ops.append(
            {
                "conv": conv,
                "ci": int(conv.in_channels),
                "co": int(conv.out_channels),
                "k": int(conv.kernel_size[0]),
                "s": int(conv.stride[0]),
                "p": int(conv.padding[0]),
                "dw": bool(dw),
                "relu": bool(relu),
                "save_skip": bool(save_skip),
                "add_skip": bool(add_skip),
            }
        )

    add_conv(bb.conv_stem, relu=isinstance(bb.act1, nn.ReLU))
    for stage in bb.blocks:
        for blk in stage:
            cls = type(blk).__name__
            if cls == "ConvBnAct":
                add_conv(blk.conv, relu=isinstance(blk.bn1, nn.ReLU))
            elif cls == "UniversalInvertedResidual":
                sub_convs = []
                for attr in ("dw_start", "pw_exp", "dw_mid", "pw_proj", "dw_end"):
                    sub = getattr(blk, attr)
                    if not isinstance(sub, nn.Identity):
                        sub_convs.append((sub.conv, isinstance(sub.bn, nn.ReLU)))
                has_skip = bool(getattr(blk, "has_skip", False))
                for idx, (conv, relu) in enumerate(sub_convs):
                    add_conv(
                        conv,
                        relu=relu,
                        save_skip=(has_skip and idx == 0),
                        add_skip=(has_skip and idx == len(sub_convs) - 1),
                    )
            else:
                raise ValueError(f"Unsupported block type: {cls}")
    return ops


def pack_vsn1(model: TrunkMRL, mean: tuple[float, ...], std: tuple[float, ...]) -> bytes:
    ops = collect_ops(model)
    head_in = int(model.head.in_features)
    head_out = int(model.head.out_features)
    buf = bytearray()
    buf += struct.pack("<4sIIII3f3f", b"VSN1", 1, len(ops), head_in, head_out, *mean, *std)

    for op in ops:
        conv: nn.Conv2d = op["conv"]
        q_w, scales = quantize_per_channel_i8(conv.weight)
        biases = conv.bias.detach().cpu().numpy().astype("<f4")
        if op["dw"]:
            packed_w = np.ascontiguousarray(q_w[:, 0, :, :].transpose(1, 2, 0))
        else:
            packed_w = np.ascontiguousarray(q_w.transpose(2, 3, 1, 0))
        buf += struct.pack(
            "<HHBBBBBBBBI",
            op["ci"], op["co"], op["k"], op["s"], op["p"],
            int(op["dw"]), int(op["relu"]), int(op["save_skip"]), int(op["add_skip"]),
            0, packed_w.size,
        )
        buf += scales.tobytes()
        buf += biases.tobytes()
        buf += packed_w.tobytes()

    q_hw, h_scales = quantize_per_channel_i8(model.head.weight)
    h_biases = model.head.bias.detach().cpu().numpy().astype("<f4")
    buf += h_scales.tobytes()
    buf += h_biases.tobytes()
    buf += np.ascontiguousarray(q_hw).tobytes()
    return bytes(buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/nano_v1.bin"))
    args = parser.parse_args()

    model = TrunkMRL(MODEL_NAME, pretrained=False).eval()
    fold_all_bn(model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    ref = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0).eval()
    cfg = timm.data.resolve_model_data_config(ref)
    mean = tuple(float(v) for v in cfg["mean"])
    std = tuple(float(v) for v in cfg["std"])

    payload = pack_vsn1(model, mean, std)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"Exported {len(payload)} bytes -> {args.out}")


if __name__ == "__main__":
    main()
