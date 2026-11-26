#!/usr/bin/env python3
"""Generate publication-quality PNG benchmark charts for visers-nano.

See BENCHMARKS.md for the abstract-v2 retrieval scores and measured CPU throughput.
"""

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

BG = "#FFFFFF"
CARD_BG = "#F8FAFC"
BORDER = "#E2E8F0"
GRID = "#E2E8F0"
TEXT = "#0F172A"
TEXT_MUTED = "#475569"
TEXT_LIGHT = "#64748B"

EMERALD = "#059669"
EMERALD_BRIGHT = "#10B981"
EMERALD_SOFT = "#D1FAE5"
INDIGO = "#4F46E5"
CPUBONE_NANO_224_COLOR = "#818CF8"
SKY = "#0EA5E9"
VIOLET = "#8B5CF6"
PINK = "#DB2777"
AMBER = "#D97706"
AMBER_SOFT = "#FEF3C7"
SLATE = "#64748B"
SLATE_LIGHT = "#94A3B8"

# Measured on the abstract-v2 validation RGB corpus (see BENCHMARKS.md).
NANO_P50_MS = 0.430906
NANO_THROUGHPUT_1T = 2268.251
NANO_THROUGHPUT_6T = 13224.2
NANO_ORT_THROUGHPUT_1T = 1752.062
NANO_ORT_THROUGHPUT_6T = 8518.156
MOBILECLIP2_S0_THROUGHPUT_1T = 16.136
MOBILECLIP2_S0_THROUGHPUT_6T = 77.960
CPUBONE_NANO = {
    128: {"p50_ms": 2.222260, "throughput_1t": 445.063, "throughput_6t": 1230.948},
    224: {"p50_ms": 5.726606, "throughput_1t": 173.565, "throughput_6t": 727.018},
}

# ---------------------------------------------------------------------------
# Retrieval scores (abstract-v2 test) and 1-thread CPU P50 latency (ms)
# name, short label, ms, style, genre, photo, selection, color
# ---------------------------------------------------------------------------
MODELS = [
    ("Visers Nano", "Visers Nano", NANO_P50_MS, 0.7686, 0.5291, 0.4126, 0.5307, EMERALD),
    ("Visers v0.2", "v0.2", 1.279, 0.6451, 0.3724, 0.2082, 0.3585, SLATE_LIGHT),
    ("Visers v3 (v0.3)", "v0.3", 1.916, 0.6688, 0.4107, 0.2269, 0.3833, SLATE),
    ("CPUBone Nano 128px", "CPUBone Nano 128", CPUBONE_NANO[128]["p50_ms"], 0.7160, 0.4675, 0.3982, 0.4950, INDIGO),
    ("CPUBone Nano 224px", "CPUBone Nano 224", CPUBONE_NANO[224]["p50_ms"], 0.7332, 0.4883, 0.4560, 0.5334, INDIGO),
    ("CPUBone B0 224px", "CPUBone B0", 11.475, 0.7304, 0.4843, 0.4903, 0.5488, INDIGO),
    ("TinyCLIP 8M", "TinyCLIP 8M", 37.405, 0.7741, 0.5363, 0.5152, 0.5852, SKY),
    ("MobileCLIP2-S0", "MobileCLIP2-S0", 65.803, 0.7718, 0.5707, 0.5953, 0.6333, VIOLET),
    ("MobileCLIP2-S2", "MobileCLIP2-S2", 184.226, 0.8019, 0.5916, 0.6274, 0.6621, PINK),
]
TEACHER = {"style": 0.8380, "genre": 0.5907, "photo": 0.6084, "sel": 0.6614}


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
        f'  <rect width="{w}" height="{h}" fill="{BG}"/>',
    ]


def text(x, y, s, size=13, weight="400", fill=TEXT, anchor="start", mono=False, halo=False) -> str:
    fam = FONT_MONO if mono else FONT_SANS
    extra = f'stroke="{CARD_BG}" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"' if halo else ""
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{fam}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" {extra}>{escape(s)}</text>'
    )


def title_block(lines: list[str], title: str, subtitle: str, x: int = 56) -> None:
    lines.append(text(x, 44, title, size=25, weight="700"))
    lines.append(text(x, 72, subtitle, size=14.5, fill=TEXT_MUTED))


# ---------------------------------------------------------------------------
# 1. Pareto: 2x2 small multiples (selection / style / genre / photo)
# ---------------------------------------------------------------------------
def pareto_panel(lines, px, py, pw, ph, title, note, idx, key, y_lo, y_hi, y_step):
    lines.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(px + 22, py + 30, title, size=17, weight="700"))
    lines.append(text(px + 22, py + 52, note, size=12.5, fill=EMERALD, weight="600"))

    l, r, t, b = px + 60, px + pw - 22, py + 76, py + ph - 50
    x_lo, x_hi = 0.3, 420.0

    def sx(ms):
        return l + (math.log10(ms) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * (r - l)

    def sy(v):
        return t + (1 - (v - y_lo) / (y_hi - y_lo)) * (b - t)

    n = int(round((y_hi - y_lo) / y_step))
    for i in range(n + 1):
        v = y_lo + i * y_step
        lines.append(f'  <line x1="{l}" y1="{sy(v):.1f}" x2="{r}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(l - 8, sy(v) + 4, f"{v:.2f}", size=11.5, fill=TEXT_LIGHT, anchor="end", mono=True))
    for v in [0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
        lines.append(f'  <line x1="{sx(v):.1f}" y1="{t}" x2="{sx(v):.1f}" y2="{b}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(sx(v), b + 18, f"{v:g}", size=11.5, fill=TEXT_LIGHT, anchor="middle", mono=True))
    lines.append(text((l + r) / 2, b + 38, "CPU P50 latency, ms (log, 1 thread)", size=12, weight="600", fill=TEXT_MUTED, anchor="middle"))

    # teacher ceiling
    ty = sy(TEACHER[key])
    lines.append(f'  <line x1="{l}" y1="{ty:.1f}" x2="{r}" y2="{ty:.1f}" stroke="{AMBER}" stroke-width="1.8" stroke-dasharray="7 5"/>')
    lines.append(text(l + 8, ty - 6, f"SigLIP2 Base teacher {TEACHER[key]:.3f}", size=11, weight="700", fill="#92400E", anchor="start", halo=True))

    pts = sorted(MODELS, key=lambda m: m[2])
    # pareto frontier: strictly better score than every faster model
    frontier, best = [], -1.0
    for m in pts:
        if m[idx] > best:
            frontier.append(m)
            best = m[idx]
    poly = " ".join(f"{sx(m[2]):.1f},{sy(m[idx]):.1f}" for m in frontier)
    lines.append(f'  <polyline points="{poly}" fill="none" stroke="{EMERALD}" stroke-width="2.2" stroke-dasharray="6 4" opacity="0.8"/>')

    for i, m in enumerate(pts):
        name, short, ms, *_rest = m
        v = m[idx]
        col = m[7]
        x, y = sx(ms), sy(v)
        hero = name == "Visers Nano"
        on_f = m in frontier
        if hero:
            lines.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="17" fill="{EMERALD_SOFT}" opacity="0.9"/>')
        lines.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{9 if hero else 6.5}" fill="{col}" stroke="#FFFFFF" stroke-width="2"/>')
        # (dx, dy, anchor) per model to avoid collisions
        place = {
            "Visers Nano": (-8, 36, "start"),
            "Visers v0.2": (-10, 4, "end"),
            "Visers v3 (v0.3)": (0, -12, "middle"),
            "CPUBone Nano 128px": (0, 22, "middle"),
            "CPUBone Nano 224px": (-9, -11, "end"),
            "CPUBone B0 224px": (0, 22, "middle"),
            "TinyCLIP 8M": (0, 22, "middle"),
            "MobileCLIP2-S0": (-10, -10, "end"),
            "MobileCLIP2-S2": (0, -13, "middle"),
        }[name]
        if key in ("style", "genre", "photo") and hero:
            place = (-8, -16, "start")
        if key == "style" and name == "TinyCLIP 8M":
            place = (0, -13, "middle")
        if key == "style" and name == "MobileCLIP2-S0":
            place = (0, 22, "middle")
        label = f"{short} {v:.3f}" if hero else short
        lines.append(text(x + place[0], y + place[1], label, size=12.5 if hero else 11, weight="700" if (hero or on_f) else "500", fill=EMERALD if hero else (TEXT if on_f else TEXT_MUTED), anchor=place[2], halo=True))


def build_pareto_chart() -> str:
    w, h = 1360, 1070
    lines = svg_header(w, h, "Visers Nano speed vs quality (abstract-v2)")
    title_block(
        lines,
        "Speed vs. Quality: Visers Nano owns the sub-millisecond corner",
        "abstract-v2 held-out test (5,940 images) · 1-thread CPU P50 latency, Intel i7-8700K · per-task nDCG@10 and combined Selection score",
    )
    pw, ph = 620, 420
    gx, gy = 56, 100
    models_by_name = {model[0]: model for model in MODELS}
    sel_ratio = models_by_name["Visers Nano"][6] / TEACHER["sel"] * 100
    tinyclip_ratio = models_by_name["TinyCLIP 8M"][2] / NANO_P50_MS
    cpubone_ratio = CPUBONE_NANO[128]["p50_ms"] / NANO_P50_MS
    spec = [
        ("Combined Selection score", f"{sel_ratio:.0f}% of the SigLIP2 teacher score at {NANO_P50_MS:.3f} ms", 6, "sel", 0.30, 0.70, 0.10),
        ("Art Style nDCG@10", f"0.769 ≈ TinyCLIP 8M (0.774) at {tinyclip_ratio:.0f}× lower latency", 3, "style", 0.60, 0.85, 0.05),
        ("Art Genre nDCG@10", "0.529 vs CPUBone Nano / B0 (0.468–0.488)", 4, "genre", 0.35, 0.65, 0.05),
        ("Photo (COCO) nDCG@10", f"0.413 vs CPUBone Nano 128px (0.398) at {cpubone_ratio:.1f}× lower latency", 5, "photo", 0.15, 0.70, 0.10),
    ]
    for i, (ttl, note, idx, key, lo, hi, step) in enumerate(spec):
        col, row = i % 2, i // 2
        pareto_panel(lines, gx + col * (pw + 24), gy + row * (ph + 24), pw, ph, ttl, note, idx, key, lo, hi, step)

    # legend
    ly = gy + 2 * ph + 24 + 38
    items = [
        (EMERALD, "Visers Nano (Rust AVX2, 128px)"),
        (INDIGO, "CPUBone Nano / B0 (ORT FP32)"),
        (SKY, "TinyCLIP 8M"),
        (VIOLET, "MobileCLIP2-S0"),
        (PINK, "MobileCLIP2-S2"),
        (SLATE, "Visers v0.2 / v0.3"),
    ]
    lx = 56
    for col, lbl in items:
        lines.append(f'  <circle cx="{lx + 6}" cy="{ly - 4}" r="6" fill="{col}"/>')
        lines.append(text(lx + 18, ly, lbl, size=12.5, weight="600", fill=TEXT_MUTED))
        lx += 18 + len(lbl) * 6.9 + 22
    lines.append(text(56, ly + 26, "Dashed green line = Pareto frontier of that task (a model is on it only if it beats every faster model). Dashed amber = teacher ceiling.", size=12, fill=TEXT_LIGHT))
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Domain comparison + throughput of the same models
# ---------------------------------------------------------------------------
def build_domain_comparison_chart() -> str:
    w, h = 1360, 640
    lines = svg_header(w, h, "Quality per domain and throughput vs larger models")
    title_block(
        lines,
        "Retrieval Quality and CPU Throughput",
        "abstract-v2 held-out test (5,940 images) · single-thread CPU throughput on Intel Core i7-8700K",
    )
    shown_labels = {
        "Visers Nano": "Visers Nano",
        "CPUBone Nano 224px": "CPUBone Nano (224px)",
        "TinyCLIP 8M": "TinyCLIP 8M",
        "MobileCLIP2-S0": "MobileCLIP2-S0",
        "MobileCLIP2-S2": "MobileCLIP2-S2",
    }
    shown = [m for m in MODELS if m[0] in shown_labels]
    colors = {m[0]: m[7] for m in shown}
    colors["CPUBone Nano 224px"] = CPUBONE_NANO_224_COLOR

    # ---- left panel
    p1x, p1y, p1w, p1h = 56, 100, 800, 500
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p1x + 24, p1y + 34, "Retrieval quality (nDCG@10, higher is better)", size=17, weight="700"))
    lx = p1x + 24
    for m in shown:
        lbl = shown_labels[m[0]]
        lines.append(f'  <rect x="{lx}" y="{p1y + 50}" width="11" height="11" rx="2" fill="{colors[m[0]]}"/>')
        lines.append(text(lx + 16, p1y + 60, lbl, size=11.5, weight="600", fill=TEXT_MUTED))
        lx += 16 + len(lbl) * 6.4 + 20

    cl, cr, ct, cb = p1x + 56, p1x + p1w - 24, p1y + 100, p1y + p1h - 50
    ch = cb - ct

    def sy(v):
        return ct + (1 - v / 0.90) * ch

    for v in [0.0, 0.2, 0.4, 0.6, 0.8]:
        lines.append(f'  <line x1="{cl}" y1="{sy(v):.1f}" x2="{cr}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(cl - 10, sy(v) + 4, f"{v:.1f}", size=12, fill=TEXT_LIGHT, anchor="end", mono=True))

    domains = [("Art Style", 3), ("Art Genre", 4), ("Photo (COCO)", 5)]
    gw = (cr - cl) / 3
    bw, gap = 28, 6
    total = len(shown) * bw + (len(shown) - 1) * gap
    for di, (dname, idx) in enumerate(domains):
        gx = cl + di * gw + (gw - total) / 2
        for mi, m in enumerate(shown):
            v = m[idx]
            bx = gx + mi * (bw + gap)
            by = sy(v)
            lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{cb - by:.1f}" rx="4" fill="{colors[m[0]]}"/>')
            lines.append(text(bx + bw / 2, by - 6, f"{v:.3f}", size=10.5, weight="700" if mi == 0 else "500", fill=TEXT if mi == 0 else TEXT_MUTED, anchor="middle", mono=True, halo=True))
        lines.append(text(gx + total / 2, cb + 26, dname, size=14, weight="700", anchor="middle"))

    # ---- right panel: throughput (log bars)
    p2x, p2y, p2w, p2h = 880, 100, 424, 500
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p2x + 24, p2y + 34, "1-thread throughput (img/s, log scale)", size=17, weight="700"))
    lines.append(text(p2x + 24, p2y + 56, "Measured from in-memory RGB to embedding", size=12.5, fill=TEXT_MUTED))

    throughput = {
        "Visers Nano": NANO_THROUGHPUT_1T,
        "CPUBone Nano 224px": CPUBONE_NANO[224]["throughput_1t"],
        # TinyCLIP and S2 use earlier measurements on the same corpus.
        "TinyCLIP 8M": 26.582,
        "MobileCLIP2-S0": MOBILECLIP2_S0_THROUGHPUT_1T,
        "MobileCLIP2-S2": 5.355,
    }

    bl, br = p2x + 24, p2x + p2w - 24
    log_lo, log_hi = 0.0, math.log10(5000.0)
    for i, m in enumerate(shown):
        yy = p2y + 96 + i * 64
        v = throughput[m[0]]
        bar = (math.log10(v) - log_lo) / (log_hi - log_lo) * (br - bl)
        lines.append(text(bl, yy, shown_labels[m[0]], size=13, weight="700"))
        lines.append(f'  <rect x="{bl}" y="{yy + 8}" width="{br - bl}" height="22" rx="5" fill="#E2E8F0" opacity="0.55"/>')
        lines.append(f'  <rect x="{bl}" y="{yy + 8}" width="{bar:.1f}" height="22" rx="5" fill="{colors[m[0]]}"/>')
        label = f"{v:,.0f} img/s" if v >= 100 else f"{v:,.1f} img/s"
        if m[0] == "Visers Nano":
            lines.append(text(bl + bar - 8, yy + 24, label, size=12, weight="700", fill="#FFFFFF", anchor="end", mono=True))
        else:
            lines.append(text(bl + bar + 8, yy + 24, label, size=11.5, weight="700", fill=TEXT_MUTED, mono=True))
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Runtime throughput + per-stage profile
# ---------------------------------------------------------------------------
def build_runtime_profile_chart() -> str:
    w, h = 1360, 666
    lines = svg_header(w, h, "Visers Nano throughput and per-stage profile")
    title_block(
        lines,
        "CPU Throughput and Per-Stage Profile",
        "Intel Core i7-8700K · 128 in-memory RGB validation frames · includes model-specific preprocessing",
    )

    p1x, p1y, p1w, p1h = 56, 106, 640, 526
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p1x + 24, p1y + 34, "Measured inference throughput (img/s)", size=17, weight="700"))
    lines.append(text(p1x + 24, p1y + 56, "CPU affinity restricted to 1 or 6 physical cores", size=12.5, fill=TEXT_MUTED))

    groups = [
        ("Visers Nano · 128px (Rust AVX2/FMA)", NANO_THROUGHPUT_1T, NANO_THROUGHPUT_6T, EMERALD),
        ("Visers Nano · 128px (ORT FP32)", NANO_ORT_THROUGHPUT_1T, NANO_ORT_THROUGHPUT_6T, EMERALD_BRIGHT),
        ("CPUBone Nano · 224px (ORT FP32)", CPUBONE_NANO[224]["throughput_1t"], CPUBONE_NANO[224]["throughput_6t"], CPUBONE_NANO_224_COLOR),
        ("MobileCLIP2-S0 · 256px (ORT FP32)", MOBILECLIP2_S0_THROUGHPUT_1T, MOBILECLIP2_S0_THROUGHPUT_6T, VIOLET),
    ]
    max_throughput = math.ceil(max(max(one, six) for _, one, six, _ in groups) / 1000) * 1000
    label_x, bar_x, bar_width = p1x + 24, p1x + 102, p1w - 220
    for i, (label, one_core, six_cores, color) in enumerate(groups):
        group_y = p1y + 96 + i * 110
        lines.append(text(label_x, group_y, label, size=14, weight="700"))
        for j, (core_label, value) in enumerate((("1 core", one_core), ("6 cores", six_cores))):
            row_y = group_y + 14 + j * 30
            width = value / max_throughput * bar_width
            lines.append(text(label_x, row_y + 15, core_label, size=12, fill=TEXT_MUTED))
            lines.append(f'  <rect x="{bar_x}" y="{row_y}" width="{bar_width}" height="20" rx="4" fill="{GRID}"/>')
            lines.append(f'  <rect x="{bar_x}" y="{row_y}" width="{width:.1f}" height="20" rx="4" fill="{color}"/>')
            formatted = f"{value:,.0f}" if value >= 100 else f"{value:.1f}"
            lines.append(text(p1x + p1w - 24, row_y + 15, formatted, size=13, weight="700", anchor="end", mono=True))

    p2x, p2y, p2w, p2h = 724, 106, 580, 526
    stages = [
        ("Pointwise 1×1 Convs", 171.11, INDIGO),
        ("Fused RGB Stem + Block 0.0", 147.72, EMERALD),
        ("Depthwise 3×3 / 5×5 Convs", 73.63, EMERALD_BRIGHT),
        ("Conv 3×3 (Block 1.0)", 44.50, AMBER),
        ("GAP + Head + L2 Norm", 7.60, SLATE),
    ]
    total_us = sum(us for _, us, _ in stages)
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p2x + 24, p2y + 34, f"Visers Nano stage profile ({total_us:.1f} µs total)", size=17, weight="700"))
    lines.append(text(p2x + 24, p2y + 56, "Mean stage times, one thread (--profile)", size=12.5, fill=TEXT_MUTED))
    sx0, sy0, sw = p2x + 24, p2y + 82, p2w - 48
    cur = sx0
    for _, us, col in stages:
        seg = us / total_us * sw
        lines.append(f'  <rect x="{cur:.1f}" y="{sy0}" width="{seg:.1f}" height="22" fill="{col}"/>')
        cur += seg
    for i, (name, us, col) in enumerate(stages):
        pct = us / total_us * 100
        ry = p2y + 150 + i * 57
        lines.append(f'  <rect x="{p2x + 24}" y="{ry - 11}" width="12" height="12" rx="3" fill="{col}"/>')
        lines.append(text(p2x + 44, ry, name, size=13, weight="600"))
        lines.append(f'  <rect x="{p2x + 255}" y="{ry - 12}" width="{us / total_us * 360:.1f}" height="14" rx="4" fill="{col}" opacity="0.85"/>')
        lines.append(text(p2x + 556, ry, f"{us:5.1f} µs ({pct:4.1f}%)", size=12.5, weight="700", fill=TEXT_MUTED, anchor="end", mono=True))
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    render_svg_to_png(build_pareto_chart(), ASSETS / "pareto-speed-quality.png")
    render_svg_to_png(build_domain_comparison_chart(), ASSETS / "domain-comparison.png")
    render_svg_to_png(build_runtime_profile_chart(), ASSETS / "nano-throughput-profile.png")


if __name__ == "__main__":
    main()
