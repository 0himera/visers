# Visers Nano — Python Forge

Reference PyTorch architecture, `Conv+BN` folding, and `VSN1` (`W8` int8) binary exporter for `visers-nano`.

## Requirements

- `torch >= 2.2`
- `timm >= 1.0`
- `numpy`

## Export a trained checkpoint to `weights/nano_v1.bin`

```bash
python forge/export_vsn1.py --checkpoint /path/to/nano-folded-w8-best.pt --out weights/nano_v1.bin
```
