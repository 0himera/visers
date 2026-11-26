#!/usr/bin/env python3
"""Generate publication-quality PNG benchmark charts for visers-nano.

All numbers come from reports in the source repo (see BENCHMARKS.md):
  * abstract-v2 held-out test (5,940 images), 1-thread CPU P50 latency (i7-8700K)
  * Visers Nano throughput measured with `nano_rgb --bench`
Throughput of baselines is derived as 1000 / P50 (batch=1, 1 thread).
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
SKY = "#0EA5E9"
VIOLET = "#8B5CF6"
PINK = "#DB2777"
AMBER = "#D97706"
AMBER_SOFT = "#FEF3C7"
SLATE = "#64748B"
SLATE_LIGHT = "#94A3B8"

# ---------------------------------------------------------------------------
# Verified data (abstract-v2 test; latency = 1-thread CPU P50, ms)
# name, short label, ms, style, genre, photo, selection, color
# ---------------------------------------------------------------------------
MODELS = [
    ("Visers Nano", "Visers Nano", 0.457, 0.7686, 0.5291, 0.4126, 0.5307, EMERALD),
    ("Visers v0.2", "v0.2", 1.279, 0.6451, 0.3724, 0.2082, 0.3585, SLATE_LIGHT),
    ("Visers v3 (v0.3)", "v0.3", 1.916, 0.6688, 0.4107, 0.2269, 0.3833, SLATE),
    ("CPUBone Nano 128px", "CPUBone Nano 128", 3.580, 0.7160, 0.4675, 0.3982, 0.4950, INDIGO),
    ("CPUBone Nano 224px", "CPUBone Nano 224", 5.403, 0.7332, 0.4883, 0.4560, 0.5334, INDIGO),
    ("CPUBone B0 224px", "CPUBone B0", 11.475, 0.7304, 0.4843, 0.4903, 0.5488, INDIGO),
    ("TinyCLIP 8M", "TinyCLIP 8M", 37.405, 0.7741, 0.5363, 0.5152, 0.5852, SKY),
    ("MobileCLIP2-S0", "MobileCLIP2-S0", 65.803, 0.7718, 0.5707, 0.5953, 0.6333, VIOLET),
    ("MobileCLIP2-S2", "MobileCLIP2-S2", 184.226, 0.8019, 0.5916, 0.6274, 0.6621, PINK),
]
TEACHER = {"style": 0.8380, "genre": 0.5907, "photo": 0.6084, "sel": 0.6614}
NANO_THROUGHPUT_1T = 1838.8  # measured, nano_rgb --bench (dim-256)
NANO_THROUGHPUT_6T = 10083.2  # measured, NanoSpinPool 6 threads


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
        if key in ("style", "genre") and hero:
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
    sel_ratio = 0.5307 / TEACHER["sel"] * 100
    spec = [
        ("Combined Selection score", f"{sel_ratio:.0f}% of the SigLIP2 teacher score at 0.457 ms", 6, "sel", 0.30, 0.70, 0.10),
        ("Art Style nDCG@10", "0.769 ≈ TinyCLIP 8M (0.774) at 82× lower latency", 2, "style", 0.60, 0.85, 0.05),
        ("Art Genre nDCG@10", "above CPUBone Nano / B0 (0.468–0.488), 8–25× faster", 3, "genre", 0.35, 0.65, 0.05),
        ("Photo (COCO) nDCG@10", "weakest task: 0.413 vs CPUBone Nano 128px 0.398 at 7.8× lower latency", 4, "photo", 0.15, 0.70, 0.10),
    ]
    # idx in MODELS tuple: style=3, genre=4, photo=5, sel=6
    idx_map = {"sel": 6, "style": 3, "genre": 4, "photo": 5}
    for i, (ttl, note, _, key, lo, hi, step) in enumerate(spec):
        col, row = i % 2, i // 2
        pareto_panel(lines, gx + col * (pw + 24), gy + row * (ph + 24), pw, ph, ttl, note, idx_map[key], key, lo, hi, step)

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
        "Quality per Task vs. Larger Models — and What It Costs in Throughput",
        "Left: nDCG@10 on abstract-v2 held-out test (5,940 images). Right: 1-thread throughput of the same models (i7-8700K).",
    )
    shown = [m for m in MODELS if m[0] in ("Visers Nano", "CPUBone B0 224px", "TinyCLIP 8M", "MobileCLIP2-S0", "MobileCLIP2-S2")]
    shown_labels = {
        "Visers Nano": "Visers Nano",
        "CPUBone B0 224px": "CPUBone B0",
        "TinyCLIP 8M": "TinyCLIP 8M",
        "MobileCLIP2-S0": "MobileCLIP2-S0",
        "MobileCLIP2-S2": "MobileCLIP2-S2",
    }

    # ---- left panel
    p1x, p1y, p1w, p1h = 56, 100, 800, 500
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p1x + 24, p1y + 34, "Retrieval quality (nDCG@10, higher is better)", size=17, weight="700"))
    lx = p1x + 24
    for m in shown:
        lbl = shown_labels[m[0]]
        lines.append(f'  <rect x="{lx}" y="{p1y + 50}" width="11" height="11" rx="2" fill="{m[7]}"/>')
        lines.append(text(lx + 16, p1y + 60, lbl, size=11.5, weight="600", fill=TEXT_MUTED))
        lx += 16 + len(lbl) * 6.4 + 20

    cl, cr, ct, cb = p1x + 56, p1x + p1w - 24, p1y + 100, p1y + p1h - 62
    ch = cb - ct

    def sy(v):
        return ct + (1 - v / 0.90) * ch

    for v in [0.0, 0.2, 0.4, 0.6, 0.8]:
        lines.append(f'  <line x1="{cl}" y1="{sy(v):.1f}" x2="{cr}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(cl - 10, sy(v) + 4, f"{v:.1f}", size=12, fill=TEXT_LIGHT, anchor="end", mono=True))

    domains = [("Art Style", 3, "style"), ("Art Genre", 4, "genre"), ("Photo (COCO)", 5, "photo")]
    gw = (cr - cl) / 3
    bw, gap = 28, 6
    total = len(shown) * bw + (len(shown) - 1) * gap
    for di, (dname, idx, key) in enumerate(domains):
        gx = cl + di * gw + (gw - total) / 2
        ty = sy(TEACHER[key])
        lines.append(f'  <line x1="{gx - 8:.1f}" y1="{ty:.1f}" x2="{gx + total + 8:.1f}" y2="{ty:.1f}" stroke="{AMBER}" stroke-width="2" stroke-dasharray="5 3"/>')
        for mi, m in enumerate(shown):
            v = m[idx]
            bx = gx + mi * (bw + gap)
            by = sy(v)
            lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{cb - by:.1f}" rx="4" fill="{m[7]}"/>')
            lines.append(text(bx + bw / 2, by - 6, f"{v:.3f}", size=10.5, weight="700" if mi == 0 else "500", fill=TEXT if mi == 0 else TEXT_MUTED, anchor="middle", mono=True, halo=True))
        lines.append(text(gx + total / 2, cb + 26, dname, size=14, weight="700", anchor="middle"))
        lines.append(text(gx + total / 2, cb + 44, f"SigLIP2 teacher (dashed): {TEACHER[key]:.3f}", size=11, weight="700", fill="#92400E", anchor="middle", mono=True))

    # ---- right panel: throughput (log bars)
    p2x, p2y, p2w, p2h = 880, 100, 424, 500
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p2x + 24, p2y + 34, "1-thread throughput (img/s, log scale)", size=17, weight="700"))
    lines.append(text(p2x + 24, p2y + 56, "higher is better · same models and colors", size=12.5, fill=TEXT_MUTED))

    def thr(m):
        return NANO_THROUGHPUT_1T if m[0] == "Visers Nano" else 1000.0 / m[2]

    bl, br = p2x + 24, p2x + p2w - 24
    log_lo, log_hi = 0.0, math.log10(5000.0)
    for i, m in enumerate(shown):
        yy = p2y + 100 + i * 66
        v = thr(m)
        bar = (math.log10(v) - log_lo) / (log_hi - log_lo) * (br - bl)
        lines.append(text(bl, yy, shown_labels[m[0]], size=13, weight="700"))
        lines.append(f'  <rect x="{bl}" y="{yy + 8}" width="{br - bl}" height="22" rx="5" fill="#E2E8F0" opacity="0.55"/>')
        lines.append(f'  <rect x="{bl}" y="{yy + 8}" width="{bar:.1f}" height="22" rx="5" fill="{m[7]}"/>')
        label = f"{v:,.0f} img/s" if v >= 100 else f"{v:,.1f} img/s"
        if i == 0:
            lines.append(text(bl + bar - 8, yy + 24, label, size=12, weight="700", fill="#FFFFFF", anchor="end", mono=True))
        else:
            lines.append(text(bl + bar + 8, yy + 24, f"{label} · {NANO_THROUGHPUT_1T / v:,.0f}× slower", size=11.5, weight="700", fill=TEXT_MUTED, mono=True))
    lines.append(text(p2x + 24, p2y + p2h - 40, "Baselines: 1000 / P50 (ORT FP32, batch=1). Visers Nano: measured", size=11, fill=TEXT_LIGHT))
    lines.append(text(p2x + 24, p2y + p2h - 24, "1,839 img/s (conservative; 1000 / 0.457 ms would be 2,188).", size=11, fill=TEXT_LIGHT))
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Runtime throughput vs CPUBone Nano + per-stage profile
# ---------------------------------------------------------------------------
def build_runtime_profile_chart() -> str:
    w, h = 1360, 560
    lines = svg_header(w, h, "Visers Nano throughput and per-stage profile")
    title_block(
        lines,
        "Zero-Alloc Rust Engine: Throughput vs. CPUBone Nano & Stage Profile",
        "Intel Core i7-8700K (AVX2 + FMA), 128×128 RGB input · Visers Nano: 0 heap allocations per call",
    )

    p1x, p1y, p1w, p1h = 56, 106, 640, 420
    lines.append(f'  <rect x="{p1x}" y="{p1y}" width="{p1w}" height="{p1h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p1x + 24, p1y + 34, "Inference throughput (images / sec)", size=17, weight="700"))
    lines.append(text(p1x + 24, p1y + 56, "Visers Nano vs. CPUBone Nano (ORT FP32), same CPU", size=12.5, fill=TEXT_MUTED))

    cpub128 = 1000.0 / 3.580
    cpub224 = 1000.0 / 5.403
    bars = [
        ("Visers Nano · 6 threads (NanoSpinPool)", NANO_THROUGHPUT_6T, EMERALD, "0.094 ms/img"),
        ("Visers Nano · 1 thread (AVX2, 0.457 ms)", NANO_THROUGHPUT_1T, EMERALD_BRIGHT, f"{NANO_THROUGHPUT_1T / cpub128:.1f}× CPUBone Nano 128px"),
        ("CPUBone Nano 128px · 1 thread (3.580 ms)", cpub128, INDIGO, ""),
        ("CPUBone Nano 224px · 1 thread (5.403 ms)", cpub224, "#818CF8", ""),
    ]
    bx0, by0, max_bw = p1x + 24, p1y + 96, p1w - 48
    for i, (label, val, col, note) in enumerate(bars):
        yy = by0 + i * 74
        bw = val / 10500.0 * max_bw
        lines.append(text(bx0, yy, label, size=13.5, weight="700"))
        lines.append(f'  <rect x="{bx0}" y="{yy + 10}" width="{max_bw}" height="28" rx="6" fill="#E2E8F0" opacity="0.5"/>')
        lines.append(f'  <rect x="{bx0}" y="{yy + 10}" width="{bw:.1f}" height="28" rx="6" fill="{col}"/>')
        main = f"{val:,.0f} img/s"
        if bw > 0.6 * max_bw:
            lines.append(text(bx0 + bw - 10, yy + 29, f"{main} · {note}", size=12, weight="700", fill="#FFFFFF", anchor="end", mono=True))
        else:
            tail = f" · {note}" if note else ""
            lines.append(text(bx0 + bw + 10, yy + 29, main + tail, size=12, weight="700", fill=TEXT, mono=True))
    lines.append(text(p1x + 24, p1y + p1h - 22, "CPUBone throughput = 1000 / P50 (1 thread, batch=1). Visers Nano 1-thread is the measured figure.", size=11, fill=TEXT_LIGHT))

    p2x, p2y, p2w, p2h = 724, 106, 580, 420
    lines.append(f'  <rect x="{p2x}" y="{p2y}" width="{p2w}" height="{p2h}" rx="10" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1.5"/>')
    lines.append(text(p2x + 24, p2y + 34, "Per-stage latency profile (594.5 µs total)", size=17, weight="700"))
    lines.append(text(p2x + 24, p2y + 56, "Instrumented 1-thread breakdown (--profile)", size=12.5, fill=TEXT_MUTED))
    stages = [
        ("Pointwise 1×1 Convs", 228.7, 38.5, INDIGO),
        ("Fused RGB Stem + Block 0.0", 181.4, 30.5, EMERALD),
        ("Depthwise 3×3 / 5×5 Convs", 107.0, 18.0, EMERALD_BRIGHT),
        ("Conv 3×3 (Block 1.0)", 61.7, 10.4, AMBER),
        ("GAP + Head + L2 Norm", 15.8, 2.7, SLATE),
    ]
    sx0, sy0, sw = p2x + 24, p2y + 82, p2w - 48
    cur = sx0
    for _, us, _, col in stages:
        seg = us / 594.6 * sw
        lines.append(f'  <rect x="{cur:.1f}" y="{sy0}" width="{seg:.1f}" height="22" fill="{col}"/>')
        cur += seg
    for i, (name, us, pct, col) in enumerate(stages):
        ry = p2y + 150 + i * 46
        lines.append(f'  <rect x="{p2x + 24}" y="{ry - 11}" width="12" height="12" rx="3" fill="{col}"/>')
        lines.append(text(p2x + 44, ry, name, size=13, weight="600"))
        lines.append(f'  <rect x="{p2x + 255}" y="{ry - 12}" width="{pct / 45.0 * 165:.1f}" height="14" rx="4" fill="{col}" opacity="0.85"/>')
        lines.append(text(p2x + 556, ry, f"{us:5.1f} µs ({pct:4.1f}%)", size=12.5, weight="700", fill=TEXT_MUTED, anchor="end", mono=True))
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    render_svg_to_png(build_pareto_chart(), ASSETS / "pareto-speed-quality.png")
    render_svg_to_png(build_domain_comparison_chart(), ASSETS / "domain-comparison.png")
    render_svg_to_png(build_runtime_profile_chart(), ASSETS / "nano-throughput-profile.png")


if __name__ == "__main__":
    main()
