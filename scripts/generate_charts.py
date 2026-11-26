#!/usr/bin/env python3
"""Generate publication-quality SVG/PNG benchmark charts for visers-nano."""

from __future__ import annotations

import math
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

FONT_SANS = "Liberation Sans, DejaVu Sans, Arial, sans-serif"
FONT_MONO = "DejaVu Sans Mono, Liberation Mono, monospace"

# Editorial light palette (clean technical documentation aesthetic)
BG = "#FFFFFF"
CARD_BG = "#F8FAFC"
BORDER = "#E2E8F0"
GRID = "#E2E8F0"
AXIS = "#94A3B8"
TEXT = "#0F172A"
TEXT_MUTED = "#475569"
TEXT_LIGHT = "#64748B"

EMERALD = "#059669"
EMERALD_BRIGHT = "#10B981"
EMERALD_SOFT = "#D1FAE5"
INDIGO = "#4F46E5"
INDIGO_SOFT = "#E0E7FF"
AMBER = "#D97706"
AMBER_SOFT = "#FEF3C7"
SLATE = "#64748B"
SLATE_LIGHT = "#94A3B8"
ROSE = "#E11D48"


def render_svg_to_png(svg_content: str, out_png: Path, width: int = 2400) -> None:
    svg_path = out_png.with_suffix(".svg")
    svg_path.write_text(svg_content, encoding="utf-8")
    subprocess.run(
        ["rsvg-convert", "--width", str(width), str(svg_path), "-o", str(out_png)],
        check=True,
    )
    svg_path.unlink()
    print(f"Wrote {out_png}")


def svg_header(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f"  <title>{escape(title)}</title>",
        "  <defs>",
        '    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">',
        '      <feGaussianBlur stdDeviation="4" result="blur"/>',
        '      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        "    </filter>",
        "  </defs>",
        f'  <rect width="{w}" height="{h}" fill="{BG}"/>',
    ]


def build_pareto_chart() -> str:
    w, h = 1280, 740
    pad_l, pad_r, pad_t, pad_b = 96, 56, 118, 96
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    x_min, x_max = 0.30, 300.0
    y_min, y_max = 0.30, 0.70

    def sx(ms: float) -> float:
        return pad_l + (math.log10(ms) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * pw

    def sy(score: float) -> float:
        return pad_t + (1.0 - (score - y_min) / (y_max - y_min)) * ph

    lines = svg_header(w, h, "Visers Nano Speed vs Quality Pareto Frontier")

    # Header
    lines.append(
        f'  <text x="{pad_l}" y="46" font-family="{FONT_SANS}" font-size="26" font-weight="700" fill="{TEXT}">'
        "Speed vs. Quality Pareto Frontier (abstract-v2, 5,940 images)</text>"
    )
    lines.append(
        f'  <text x="{pad_l}" y="76" font-family="{FONT_SANS}" font-size="15" fill="{TEXT_MUTED}">'
        "1-thread CPU P50 latency (log scale, Intel Core i7-8700K) vs. combined Selection nDCG@10 (0.25·Style + 0.25·Genre + 0.50·Photo)</text>"
    )

    # Plot background
    lines.append(
        f'  <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" rx="8" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>'
    )

    # Subtle "Better: faster & higher quality" directional hint in top-left
    lines.append(
        f'  <rect x="{pad_l + 18}" y="{pad_t + 16}" width="250" height="32" rx="6" fill="{EMERALD_SOFT}" stroke="{EMERALD_BRIGHT}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{pad_l + 30}" y="{pad_t + 37}" font-family="{FONT_SANS}" font-size="13" font-weight="700" fill="{EMERALD}">'
        "↖ Top-left is better (faster + sharper)</text>"
    )

    # Y gridlines & labels
    for y_val in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        yy = sy(y_val)
        lines.append(
            f'  <line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{pad_l - 14}" y="{yy + 5:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="13" fill="{TEXT_LIGHT}">{y_val:.2f}</text>'
        )

    # X gridlines & labels (log)
    for x_val, lbl in [(0.5, "0.5 ms"), (1, "1 ms"), (2, "2 ms"), (5, "5 ms"), (10, "10 ms"), (20, "20 ms"), (50, "50 ms"), (100, "100 ms"), (200, "200 ms")]:
        xx = sx(x_val)
        lines.append(
            f'  <line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t + ph}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{xx:.1f}" y="{pad_t + ph + 26}" text-anchor="middle" font-family="{FONT_MONO}" font-size="13" fill="{TEXT_LIGHT}">{lbl}</text>'
        )

    # Axis titles
    lines.append(
        f'  <text x="{pad_l + pw / 2:.1f}" y="{h - 24}" text-anchor="middle" font-family="{FONT_SANS}" font-size="15" font-weight="600" fill="{TEXT_MUTED}">'
        "1-Thread CPU P50 Latency (ms, log scale)</text>"
    )
    lines.append(
        f'  <text x="28" y="{pad_t + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 28 {pad_t + ph / 2:.1f})" font-family="{FONT_SANS}" font-size="15" font-weight="600" fill="{TEXT_MUTED}">'
        "Combined Selection nDCG@10</text>"
    )

    # SigLIP2 Teacher ceiling line
    siglip_y = sy(0.6614)
    lines.append(
        f'  <line x1="{pad_l}" y1="{siglip_y:.1f}" x2="{pad_l + pw}" y2="{siglip_y:.1f}" stroke="{AMBER}" stroke-width="2" stroke-dasharray="8 6"/>'
    )
    lines.append(
        f'  <rect x="{pad_l + 300}" y="{siglip_y - 28:.1f}" width="322" height="24" rx="4" fill="{AMBER_SOFT}" stroke="{AMBER}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{pad_l + 310}" y="{siglip_y - 12:.1f}" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="#92400E">'
        "SigLIP2 Base Teacher Ceiling (224px): 0.6614</text>"
    )

    # Pareto frontier points (ordered by latency)
    pareto_pts = [
        ("Visers Nano (Rust AVX2)", 0.457, 0.5307),
        ("CPUBone Nano (224px)", 5.403, 0.5334),
        ("CPUBone B0 (224px)", 11.475, 0.5488),
        ("TinyCLIP 8M (224px)", 37.405, 0.5852),
        ("MobileCLIP2-S0 (256px)", 65.803, 0.6333),
        ("MobileCLIP2-S2 (256px)", 184.226, 0.6621),
    ]
    poly_coords = " ".join(f"{sx(ms):.1f},{sy(sc):.1f}" for _, ms, sc in pareto_pts)
    lines.append(
        f'  <polyline points="{poly_coords}" fill="none" stroke="{EMERALD}" stroke-width="2.5" stroke-dasharray="6 4" opacity="0.85"/>'
    )

    # Horizontal callout connecting Visers Nano to CPUBone Nano 224px (11.8x faster)
    nano_x, nano_y = sx(0.457), sy(0.5307)
    cpub224_x, cpub224_y = sx(5.403), sy(0.5334)
    mid_x = (nano_x + cpub224_x) / 2
    lines.append(
        f'  <line x1="{nano_x + 12:.1f}" y1="{nano_y + 22:.1f}" x2="{cpub224_x - 10:.1f}" y2="{nano_y + 22:.1f}" stroke="{EMERALD}" stroke-width="1.5" stroke-dasharray="3 3"/>'
    )
    lines.append(
        f'  <rect x="{mid_x - 115:.1f}" y="{nano_y + 10:.1f}" width="230" height="24" rx="12" fill="{EMERALD_SOFT}" stroke="{EMERALD}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{mid_x:.1f}" y="{nano_y + 26:.1f}" text-anchor="middle" font-family="{FONT_SANS}" font-size="12" font-weight="700" fill="{EMERALD}">'
        "11.8× faster at matching quality</text>"
    )

    # All points: (label, sublabel, ms, score, color, radius, dx, dy, anchor, is_pareto, is_hero)
    points = [
        ("Visers Nano (Rust AVX2)", "0.457 ms · 0.531 · 0.48 MiB", 0.457, 0.5307, EMERALD, 10, -4, -36, "start", True, True),
        ("Visers Nano (ORT FP32)", "0.748 ms · 0.531", 0.748, 0.5307, EMERALD_BRIGHT, 6.5, 22, -68, "start", False, False),
        ("Visers v0.2 (Rust)", "1.279 ms · 0.359", 1.279, 0.3585, SLATE, 6.5, -12, -12, "end", False, False),
        ("Visers v3 / v0.3 (Rust)", "1.916 ms · 0.383", 1.916, 0.3833, SLATE, 6.5, 12, 16, "start", False, False),
        ("CPUBone Nano (128px)", "3.580 ms · 0.495", 3.580, 0.4950, INDIGO, 6.5, 12, 16, "start", False, False),
        ("CPUBone Nano (224px)", "5.403 ms · 0.533", 5.403, 0.5334, INDIGO, 7.5, 0, -28, "middle", True, False),
        ("CPUBone B0 (224px)", "11.475 ms · 0.549", 11.475, 0.5488, INDIGO, 7.5, 12, 20, "start", True, False),
        ("TinyCLIP 8M (224px)", "37.41 ms · 0.585", 37.405, 0.5852, INDIGO, 7.5, 12, 16, "start", True, False),
        ("MobileCLIP2-S0 (256px)", "65.80 ms · 0.633", 65.803, 0.6333, INDIGO, 7.5, -14, -10, "end", True, False),
        ("MobileCLIP2-S2 (256px)", "184.2 ms · 0.662", 184.226, 0.6621, INDIGO, 7.5, -12, -26, "end", True, False),
    ]

    # Small leader line for ORT FP32
    lines.append(
        f'  <line x1="{sx(0.748):.1f}" y1="{sy(0.5307):.1f}" x2="{sx(0.748) + 20:.1f}" y2="{sy(0.5307) - 58:.1f}" stroke="{EMERALD_BRIGHT}" stroke-width="1.2" opacity="0.75"/>'
    )

    for label, sub, ms, sc, color, r, dx, dy, anchor, is_pareto, is_hero in points:
        px, py = sx(ms), sy(sc)
        if is_hero:
            lines.append(
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="18" fill="{EMERALD_SOFT}" opacity="0.8"/>'
            )
            lines.append(
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" stroke="#FFFFFF" stroke-width="2.5" filter="url(#glow)"/>'
            )
        else:
            stroke_w = "2" if is_pareto else "1.5"
            lines.append(
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" stroke="#FFFFFF" stroke-width="{stroke_w}"/>'
            )

        tx = px + dx
        ty = py + dy
        weight = "700" if is_hero else ("600" if is_pareto else "500")
        lbl_color = TEXT if (is_hero or is_pareto) else TEXT_MUTED
        lines.append(
            f'  <text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{FONT_SANS}" font-size="{"14" if is_hero else "12.5"}" font-weight="{weight}" fill="{lbl_color}">{escape(label)}</text>'
        )
        lines.append(
            f'  <text x="{tx:.1f}" y="{ty + 15:.1f}" text-anchor="{anchor}" font-family="{FONT_MONO}" font-size="11.5" fill="{EMERALD if is_hero else TEXT_LIGHT}">{escape(sub)}</text>'
        )

    # Legend box (bottom-right)
    lx, ly = pad_l + pw - 270, pad_t + ph - 120
    lines.append(
        f'  <rect x="{lx}" y="{ly}" width="254" height="104" rx="8" fill="#FFFFFF" stroke="{BORDER}" stroke-width="1.2"/>'
    )
    legend_items = [
        (EMERALD, "Visers Nano (Pareto Frontier)", True),
        (INDIGO, "External Baselines (ORT FP32)", False),
        (SLATE, "Legacy Visers (v0.2 / v0.3)", False),
    ]
    for idx, (col, txt, dash) in enumerate(legend_items):
        iy = ly + 26 + idx * 28
        lines.append(f'  <circle cx="{lx + 22}" cy="{iy - 4}" r="6" fill="{col}"/>')
        if dash:
            lines.append(
                f'  <line x1="{lx + 8}" y1="{iy - 4}" x2="{lx + 36}" y2="{iy - 4}" stroke="{col}" stroke-width="2" stroke-dasharray="4 3"/>'
            )
        lines.append(
            f'  <text x="{lx + 44}" y="{iy}" font-family="{FONT_SANS}" font-size="12.5" font-weight="600" fill="{TEXT_MUTED}">{escape(txt)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def build_domain_comparison_chart() -> str:
    w, h = 1280, 620
    lines = svg_header(w, h, "Per-Domain Quality & Strict Artist-Disjoint Generalization")

    lines.append(
        f'  <text x="56" y="44" font-family="{FONT_SANS}" font-size="25" font-weight="700" fill="{TEXT}">'
        "Retrieval Quality Across Domains &amp; Strict Artist-Disjoint Split</text>"
    )
    lines.append(
        f'  <text x="56" y="72" font-family="{FONT_SANS}" font-size="14.5" fill="{TEXT_MUTED}">'
        "nDCG@10 on abstract-v2 held-out test (5,940 images) and strict artist-disjoint visers-v6 test (12,075 images)</text>"
    )

    # Left panel: abstract-v2 per-domain grouped bars
    p1x, p1y, p1w, p1h = 56, 108, 690, 456
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{p1x + 24}" y="{p1y + 34}" font-family="{FONT_SANS}" font-size="17" font-weight="700" fill="{TEXT}">abstract-v2 Held-Out Test (nDCG@10)</text>')

    chart_l, chart_r, chart_t, chart_b = p1x + 64, p1x + p1w - 28, p1y + 92, p1y + p1h - 54
    cw = chart_r - chart_l
    ch = chart_b - chart_t

    def sy1(val: float) -> float:
        return chart_t + (1.0 - val / 0.90) * ch

    for v in [0.0, 0.2, 0.4, 0.6, 0.8]:
        yy = sy1(v)
        lines.append(f'  <line x1="{chart_l}" y1="{yy:.1f}" x2="{chart_r}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'  <text x="{chart_l - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="12" fill="{TEXT_LIGHT}">{v:.1f}</text>')

    models = [
        ("Visers Nano (0.457 ms)", EMERALD, [0.7686, 0.5291, 0.4126]),
        ("CPUBone Nano 128px (3.58 ms)", INDIGO, [0.7160, 0.4675, 0.3982]),
        ("Visers v3 / v0.3 (1.92 ms)", SLATE, [0.6688, 0.4107, 0.2269]),
        ("Visers v0.2 (1.28 ms)", SLATE_LIGHT, [0.6451, 0.3724, 0.2082]),
    ]
    domains = ["Art Style", "Art Genre", "Photo (COCO)"]
    teacher_vals = [0.8380, 0.5907, 0.6084]

    # Legend row at top of panel 1
    legend_xs = [p1x + 24, p1x + 175, p1x + 382, p1x + 545]
    ly = p1y + 62
    for lx, (name, col, _) in zip(legend_xs, models):
        lines.append(f'  <rect x="{lx}" y="{ly - 10}" width="10" height="10" rx="2" fill="{col}"/>')
        lines.append(f'  <text x="{lx + 14}" y="{ly}" font-family="{FONT_SANS}" font-size="11" font-weight="600" fill="{TEXT_MUTED}">{escape(name)}</text>')

    group_w = cw / len(domains)
    bar_w = 32
    gap = 6
    total_bars_w = len(models) * bar_w + (len(models) - 1) * gap

    for d_idx, d_name in enumerate(domains):
        gx = chart_l + d_idx * group_w + (group_w - total_bars_w) / 2
        # Teacher dashed cap across group
        ty = sy1(teacher_vals[d_idx])
        lines.append(
            f'  <line x1="{gx - 6:.1f}" y1="{ty:.1f}" x2="{gx + total_bars_w + 6:.1f}" y2="{ty:.1f}" stroke="{AMBER}" stroke-width="2" stroke-dasharray="5 3"/>'
        )
        lines.append(
            f'  <text x="{gx + total_bars_w / 2:.1f}" y="{ty - 6:.1f}" text-anchor="middle" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="#92400E">SigLIP2 {teacher_vals[d_idx]:.3f}</text>'
        )

        for m_idx, (_, col, vals) in enumerate(models):
            val = vals[d_idx]
            bx = gx + m_idx * (bar_w + gap)
            by = sy1(val)
            bh = chart_b - by
            lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="{col}"/>')
            lines.append(
                f'  <text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-family="{FONT_MONO}" font-size="10.5" font-weight="{"700" if m_idx == 0 else "500"}" fill="{TEXT if m_idx == 0 else TEXT_MUTED}">{val:.3f}</text>'
            )

        lines.append(
            f'  <text x="{gx + total_bars_w / 2:.1f}" y="{chart_b + 24}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14" font-weight="700" fill="{TEXT}">{escape(d_name)}</text>'
        )

    # Right panel: visers-v6 strict artist-disjoint + MRL dims
    p2x, p2y, p2w, p2h = 770, 108, 454, 456
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{p2x + 24}" y="{p2y + 34}" font-family="{FONT_SANS}" font-size="17" font-weight="700" fill="{TEXT}">Strict Artist-Disjoint (visers-v6)</text>')
    lines.append(f'  <text x="{p2x + 24}" y="{p2y + 56}" font-family="{FONT_SANS}" font-size="12.5" fill="{TEXT_MUTED}">12,075 unseen test images · Visers Nano vs. Visers v6</text>')

    c2_l, c2_r, c2_t, c2_b = p2x + 58, p2x + p2w - 24, p2y + 104, p2y + 318
    c2w = c2_r - c2_l
    c2h = c2_b - c2_t

    def sy2(val: float) -> float:
        return c2_t + (1.0 - val / 0.70) * c2h

    for v in [0.0, 0.2, 0.4, 0.6]:
        yy = sy2(v)
        lines.append(f'  <line x1="{c2_l}" y1="{yy:.1f}" x2="{c2_r}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'  <text x="{c2_l - 8}" y="{yy + 4:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="11.5" fill="{TEXT_LIGHT}">{v:.1f}</text>')

    v6_metrics = [
        ("Style", 0.3587, 0.2575),
        ("Genre", 0.6096, 0.4323),
        ("Photo", 0.4181, 0.2444),
        ("Select", 0.4511, 0.2946),
    ]
    g2w = c2w / len(v6_metrics)
    bw2 = 28
    for idx, (lbl, nano_v, v6_v) in enumerate(v6_metrics):
        gx = c2_l + idx * g2w + (g2w - (bw2 * 2 + 6)) / 2
        for j, (val, col) in enumerate([(nano_v, EMERALD), (v6_v, SLATE)]):
            bx = gx + j * (bw2 + 6)
            by = sy2(val)
            lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw2}" height="{c2_b - by:.1f}" rx="4" fill="{col}"/>')
            lines.append(f'  <text x="{bx + bw2 / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-family="{FONT_MONO}" font-size="10.5" font-weight="{"700" if j == 0 else "500"}" fill="{TEXT if j == 0 else TEXT_MUTED}">{val:.3f}</text>')
        lines.append(f'  <text x="{gx + bw2 + 3:.1f}" y="{c2_b + 20}" text-anchor="middle" font-family="{FONT_SANS}" font-size="12.5" font-weight="600" fill="{TEXT}">{lbl}</text>')

    # Legend inside panel 2
    lines.append(f'  <rect x="{p2x + 24}" y="{p2y + 72}" width="10" height="10" rx="2" fill="{EMERALD}"/>')
    lines.append(f'  <text x="{p2x + 39}" y="{p2y + 81}" font-family="{FONT_SANS}" font-size="11.5" font-weight="600" fill="{TEXT_MUTED}">Visers Nano (0.457 ms)</text>')
    lines.append(f'  <rect x="{p2x + 195}" y="{p2y + 72}" width="10" height="10" rx="2" fill="{SLATE}"/>')
    lines.append(f'  <text x="{p2x + 210}" y="{p2y + 81}" font-family="{FONT_SANS}" font-size="11.5" font-weight="600" fill="{TEXT_MUTED}">Visers v6 (0.854 ms)</text>')

    # MRL summary box at bottom of panel 2
    mx, my, mw, mh = p2x + 24, p2y + 356, p2w - 48, 80
    lines.append(f'  <rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="8" fill="#FFFFFF" stroke="{BORDER}" stroke-width="1.2"/>')
    lines.append(f'  <text x="{mx + 16}" y="{my + 24}" font-family="{FONT_SANS}" font-size="13" font-weight="700" fill="{TEXT}">Matryoshka (MRL) Dimension Scaling (visers-v6)</text>')
    lines.append(
        f'  <text x="{mx + 16}" y="{my + 48}" font-family="{FONT_MONO}" font-size="12" fill="{EMERALD}" font-weight="700">256D: 0.4511 (489 KiB)</text>'
        f'  <text x="{mx + 205}" y="{my + 48}" font-family="{FONT_MONO}" font-size="12" fill="{TEXT_MUTED}">128D: 0.4501 (428 KiB)</text>'
    )
    lines.append(
        f'  <text x="{mx + 16}" y="{my + 68}" font-family="{FONT_MONO}" font-size="12" fill="{TEXT_MUTED}"> 64D: 0.4435 (397 KiB, 0.441 ms P50)</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


def build_runtime_profile_chart() -> str:
    w, h = 1280, 500
    lines = svg_header(w, h, "Visers Nano Throughput Scaling & Per-Stage Execution Profile")

    lines.append(
        f'  <text x="56" y="44" font-family="{FONT_SANS}" font-size="25" font-weight="700" fill="{TEXT}">'
        "Zero-Alloc Rust Engine: Throughput &amp; Stage Profile</text>"
    )
    lines.append(
        f'  <text x="56" y="72" font-family="{FONT_SANS}" font-size="14.5" fill="{TEXT_MUTED}">'
        "Measured on Intel Core i7-8700K (AVX2 + FMA, 128×128 RGB input, 0 heap allocations per call)</text>"
    )

    # Left panel: Throughput (img/s)
    p1x, p1y, p1w, p1h = 56, 106, 560, 350
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{p1x + 24}" y="{p1y + 34}" font-family="{FONT_SANS}" font-size="17" font-weight="700" fill="{TEXT}">Inference Throughput (images / sec)</text>')
    lines.append(f'  <text x="{p1x + 24}" y="{p1y + 56}" font-family="{FONT_SANS}" font-size="12.5" fill="{TEXT_MUTED}">Single-thread vs. 6-thread NanoSpinPool (batch=64)</text>')

    bars = [
        ("6-Thread NanoSpinPool (256D)", 10083.2, "10,083 img/s · 0.094 ms/img", EMERALD),
        ("1-Thread AVX2 (64D)", 1900.5, "1,901 img/s · 0.441 ms P50", EMERALD_BRIGHT),
        ("1-Thread AVX2 (256D)", 1838.8, "1,839 img/s · 0.457 ms P50", INDIGO),
    ]
    bx0, by0, max_bw = p1x + 24, p1y + 92, p1w - 235
    for idx, (label, val, note, col) in enumerate(bars):
        yy = by0 + idx * 82
        bw = (val / 10500.0) * max_bw
        lines.append(f'  <text x="{bx0}" y="{yy}" font-family="{FONT_SANS}" font-size="13.5" font-weight="700" fill="{TEXT}">{escape(label)}</text>')
        lines.append(f'  <rect x="{bx0}" y="{yy + 10}" width="{max_bw}" height="28" rx="6" fill="#E2E8F0" opacity="0.5"/>')
        lines.append(f'  <rect x="{bx0}" y="{yy + 10}" width="{bw:.1f}" height="28" rx="6" fill="{col}"/>')
        lines.append(f'  <text x="{bx0 + bw + 10:.1f}" y="{yy + 29}" font-family="{FONT_MONO}" font-size="12" font-weight="700" fill="{TEXT}">{escape(note)}</text>')

    # Right panel: Per-Stage Latency Breakdown (--profile)
    p2x, p2y, p2w, p2h = 644, 106, 580, 350
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{p2x + 24}" y="{p2y + 34}" font-family="{FONT_SANS}" font-size="17" font-weight="700" fill="{TEXT}">Per-Stage Latency Profile (594.5 µs total)</text>')
    lines.append(f'  <text x="{p2x + 24}" y="{p2y + 56}" font-family="{FONT_SANS}" font-size="12.5" fill="{TEXT_MUTED}">Instrumented 1-thread breakdown (--profile)</text>')

    stages = [
        ("Pointwise 1×1 Convs", 228.7, 38.5, INDIGO),
        ("Fused RGB Stem + Block 0.0", 181.4, 30.5, EMERALD),
        ("Depthwise 3×3 / 5×5 Convs", 107.0, 18.0, EMERALD_BRIGHT),
        ("Conv 3×3 (Block 1.0)", 61.7, 10.4, AMBER),
        ("GAP + Head + L2 Norm", 15.8, 2.7, SLATE),
    ]
    # Stacked overview bar
    sx0, sy0, sw = p2x + 24, p2y + 78, p2w - 48
    cur_x = sx0
    for _, us, _, col in stages:
        seg_w = (us / 594.6) * sw
        lines.append(f'  <rect x="{cur_x:.1f}" y="{sy0}" width="{seg_w:.1f}" height="22" fill="{col}"/>')
        cur_x += seg_w

    # Individual stage rows
    for idx, (name, us, pct, col) in enumerate(stages):
        ry = p2y + 128 + idx * 42
        bar_len = (pct / 45.0) * 165
        lines.append(f'  <rect x="{p2x + 24}" y="{ry - 11}" width="12" height="12" rx="3" fill="{col}"/>')
        lines.append(f'  <text x="{p2x + 44}" y="{ry}" font-family="{FONT_SANS}" font-size="13" font-weight="600" fill="{TEXT}">{escape(name)}</text>')
        lines.append(f'  <rect x="{p2x + 255}" y="{ry - 12}" width="{bar_len:.1f}" height="14" rx="4" fill="{col}" opacity="0.85"/>')
        lines.append(f'  <text x="{p2x + 556}" y="{ry}" text-anchor="end" font-family="{FONT_MONO}" font-size="12.5" font-weight="700" fill="{TEXT_MUTED}">{us:5.1f} µs ({pct:4.1f}%)</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    render_svg_to_png(build_pareto_chart(), ASSETS / "pareto-speed-quality.png")
    render_svg_to_png(build_domain_comparison_chart(), ASSETS / "domain-comparison.png")
    render_svg_to_png(build_runtime_profile_chart(), ASSETS / "nano-throughput-profile.png")


if __name__ == "__main__":
    main()
