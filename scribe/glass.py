"""Scribe's overlay artwork: glass pills drawn with Pillow for per-pixel-alpha layered windows.

Everything here is pure. A frame is an RGBA image in physical pixels, drawn at SS times its size and
reduced, so curves and hairlines stay smooth at any display scale; ui.py puts frames on screen with
UpdateLayeredWindow. The palette is sampled from a wave artwork (near-black teal ground,
verdigris, copper, gilt hairlines). Labels, counts and symbols only; light type.

The widget is one pill at the bottom centre that changes size and content with its view:
  rest       small and faint; carries a breathing copper dot while a recording runs
  start      round button with the record dot (the pointer is over the resting pill)
  rec        dot, elapsed time and stop button (recording just started, or the pointer is over it)
  listening  level bars while Ctrl+Win is held
  working    a wave travelling across low bars while Whisper and the clean-up run
  error      a gilt ✕
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 3                          # supersampling factor

GROUND = (10, 19, 20)           # #0a1314
GROUND_2 = (14, 28, 28)         # #0e1c1c
GROUND_3 = (18, 33, 33)         # #122121
VERDIGRIS = (33, 63, 58)        # #213f3a
VERDIGRIS_2 = (50, 84, 76)      # #32544c
SAGE = (120, 137, 119)          # #788977
MINT = (215, 244, 234)          # #d7f4ea
COPPER = (176, 122, 76)         # #b07a4c
COPPER_2 = (210, 154, 99)       # #d29a63
GILT = (240, 207, 159)          # #f0cf9f

GLASS_ALPHA = 0.80              # body opacity: the desktop shows through a little
HAIRLINE_ALPHA = 0.30           # gilt edge
SHEEN_ALPHA = 0.10              # mint line along the top inside edge
SHADOW_ALPHA = 0.42
MARGIN = 12                     # logical px of transparent margin around a pill, for its shadow
CLICK_ALPHA = 6                 # pixels fainter than this are left fully transparent, so clicks pass through

# Widget pill sizes per view, logical px (w, h).
SIZES = {"rest": (44, 9), "start": (36, 36), "rec": (128, 36), "listening": (116, 34),
         "working": (116, 34), "error": (116, 34)}
WIDGET_W = max(w for w, _ in SIZES.values())
WIDGET_H = max(h for _, h in SIZES.values())
REST_OPACITY = 0.62
BARS = 11
BAR_W, BAR_GAP = 3, 3.5


def ease(t: float) -> float:
    """Cubic ease-out on 0..1."""
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@lru_cache(maxsize=16)
def font(px: int, weight: str = "semilight") -> ImageFont.FreeTypeFont:
    files = {"light": "segoeuil.ttf", "semilight": "segoeuisl.ttf", "regular": "segoeui.ttf"}
    try:
        return ImageFont.truetype(files.get(weight, "segoeuisl.ttf"), px)
    except OSError:
        return ImageFont.load_default()


def _reduce(big: Image.Image, size: tuple[int, int]) -> Image.Image:
    return big.resize(size, Image.Resampling.LANCZOS)


def _pill_mask(size: tuple[int, int], rect: tuple[float, float, float, float], radius: float) -> Image.Image:
    """Antialiased L mask of a rounded rectangle (physical px), drawn at SS and reduced."""
    w, h = size
    big = Image.new("L", (w * SS, h * SS), 0)
    x0, y0, x1, y1 = (v * SS for v in rect)
    ImageDraw.Draw(big).rounded_rectangle((x0, y0, x1 - 1, y1 - 1), radius=radius * SS, fill=255)
    return _reduce(big, size)


def _gradient(size: tuple[int, int], y0: float, y1: float, top: tuple, bottom: tuple) -> Image.Image:
    w, h = size
    t = np.clip((np.arange(h, dtype=np.float32) - y0) / max(1.0, y1 - y0), 0.0, 1.0)[:, None]
    rgb = np.array(top, np.float32) * (1 - t[..., None]) + np.array(bottom, np.float32) * t[..., None]
    return Image.fromarray(np.repeat(rgb, w, axis=1).astype(np.uint8), "RGB")


def _solid(size: tuple[int, int], colour: tuple, alpha_mask: Image.Image, opacity: float) -> Image.Image:
    layer = Image.new("RGBA", size, colour + (0,))
    a = np.asarray(alpha_mask, np.float32) * opacity
    layer.putalpha(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L"))
    return layer


@lru_cache(maxsize=96)
def body(size: tuple[int, int], rect: tuple[int, int, int, int], scale: float) -> Image.Image:
    """Shadow, glass and hairline for one pill: a canvas of `size` with the pill at `rect` (physical px).
    Cached per geometry; callers copy before drawing on it."""
    w, h = size
    x0, y0, x1, y1 = rect
    radius = min(x1 - x0, y1 - y0) / 2
    mask = _pill_mask(size, rect, radius)

    shadow_mask = mask.filter(ImageFilter.GaussianBlur(5 * scale))
    shadow_mask = Image.fromarray(np.roll(np.asarray(shadow_mask), int(round(2 * scale)), axis=0), "L")
    out = _solid(size, (0, 0, 0), shadow_mask, SHADOW_ALPHA)

    fill = _gradient(size, y0, y1, GROUND_3, GROUND).convert("RGBA")
    fill.putalpha(Image.fromarray((np.asarray(mask, np.float32) * GLASS_ALPHA).astype(np.uint8), "L"))
    out.alpha_composite(fill)

    hair = max(1.0, round(scale))
    inner = _pill_mask(size, (x0 + hair, y0 + hair, x1 - hair, y1 - hair), max(0.0, radius - hair))
    ring = np.clip(np.asarray(mask, np.int16) - np.asarray(inner, np.int16), 0, 255).astype(np.uint8)
    out.alpha_composite(_solid(size, GILT, Image.fromarray(ring, "L"), HAIRLINE_ALPHA))

    # sheen: the top half of a second ring one hairline further in
    inner2 = _pill_mask(size, (x0 + 2 * hair, y0 + 2 * hair, x1 - 2 * hair, y1 - 2 * hair),
                        max(0.0, radius - 2 * hair))
    sheen = np.clip(np.asarray(inner, np.int16) - np.asarray(inner2, np.int16), 0, 255).astype(np.float32)
    fade = np.clip(1.0 - (np.arange(h, dtype=np.float32) - y0) / max(1.0, (y1 - y0) * 0.5), 0, 1)[:, None]
    out.alpha_composite(_solid(size, MINT, Image.fromarray((sheen * fade).astype(np.uint8), "L"), SHEEN_ALPHA))
    return out


def _layer(size: tuple[int, int]):
    """A blank SS-sized RGBA layer and its drawing context, for content drawn over a body."""
    big = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    return big, ImageDraw.Draw(big)


def _put(img: Image.Image, big: Image.Image) -> None:
    img.alpha_composite(_reduce(big, img.size))


def fade(img: Image.Image, opacity: float) -> Image.Image:
    """The frame with its alpha scaled by opacity (0..1)."""
    if opacity >= 0.999:
        return img
    a = (np.asarray(img.getchannel("A"), np.float32) * max(0.0, opacity)).astype(np.uint8)
    out = img.copy()
    out.putalpha(Image.fromarray(a, "L"))
    return out


# ---------------------------------------------------------------- the widget (bottom centre)


def widget_canvas(scale: float) -> tuple[int, int]:
    """Fixed window size for the widget: its largest pill plus the shadow margin."""
    return int(math.ceil((WIDGET_W + 2 * MARGIN) * scale)), int(math.ceil((WIDGET_H + 2 * MARGIN) * scale))


def widget_rect(scale: float, w: float, h: float) -> tuple[int, int, int, int]:
    """The pill's rectangle (physical px) for a pill of w x h logical px: centred, bottom edge fixed."""
    cw, ch = widget_canvas(scale)
    pw, ph = round(w * scale), round(h * scale)
    bottom = ch - round(MARGIN * scale)
    x0 = (cw - pw) // 2
    return x0, bottom - ph, x0 + pw, bottom


def listening_heights(level: float, phase: float, n: int = BARS) -> list[float]:
    """Bar heights 0..1 for a mic level 0..1: tallest in the middle, swaying a little so speech looks alive,
    and never flat while listening."""
    c = (n - 1) / 2
    out = []
    for i in range(n):
        shape = 1.0 - 0.55 * (abs(i - c) / c) ** 1.4
        sway = 0.78 + 0.22 * math.sin(phase * 7.0 + i * 1.3) * math.sin(phase * 3.1 + i * 0.7)
        out.append(max(0.0, min(1.0, (0.12 + 0.88 * level) * shape * sway)))
    return out


def working_heights(phase: float, n: int = BARS) -> list[float]:
    """A soft wave travelling left to right across low bars while the words are worked out."""
    return [0.5 + 0.5 * math.sin(phase * 6.0 - i * 0.75) for i in range(n)]


def rec_layout(rect: tuple[int, int, int, int], scale: float) -> dict:
    """Where the rec view puts its dot, clock and stop button inside the pill (physical px)."""
    x0, y0, x1, y1 = rect
    cy = (y0 + y1) / 2
    return {"dot": (x0 + 17 * scale, cy), "clock": (x0 + 30 * scale, cy),
            "stop": (x1 - 18 * scale, cy), "stop_r": 13 * scale}


def widget(scale: float, w: float, h: float, view: str, *, content: float = 1.0, opacity: float = 1.0,
           heights: list[float] | None = None, pulse: float = 0.0, clock: str = "", recording: bool = False,
           hover: str | None = None) -> Image.Image:
    """One frame of the widget: a w x h (logical px) pill showing `view`, its content faded in by `content`
    (0..1, for the moment the pill is still growing) and the whole frame by `opacity`.
    hover names the control under the pointer ("start" or "stop")."""
    size = widget_canvas(scale)
    rect = widget_rect(scale, w, h)
    img = body(size, rect, round(scale, 3)).copy()
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = max(0.0, min(1.0, content))
    big, d = _layer(size)
    S = SS

    if view == "rest" and recording:
        r = min(2.6 * scale, (y1 - y0) * 0.32)
        d.ellipse(((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S),
                  fill=COPPER_2 + (int(255 * (0.55 + 0.45 * pulse)),))
    elif view == "start" and k > 0:
        rr = (min(x1 - x0, y1 - y0) / 2 - 3 * scale) * S
        if hover == "start":
            d.ellipse((cx * S - rr, cy * S - rr, cx * S + rr, cy * S + rr), fill=VERDIGRIS_2 + (int(200 * k),))
        r = 6 * scale * S
        d.ellipse((cx * S - r, cy * S - r, cx * S + r, cy * S + r), fill=COPPER_2 + (int(255 * k),))
    elif view == "rec" and k > 0:
        L = rec_layout(rect, scale)
        dx, dy = L["dot"]
        halo = (4.5 + 3.5 * pulse) * scale
        d.ellipse(((dx - halo) * S, (dy - halo) * S, (dx + halo) * S, (dy + halo) * S),
                  fill=COPPER + (int(60 * (1 - pulse) * k),))
        r = 4.5 * scale
        d.ellipse(((dx - r) * S, (dy - r) * S, (dx + r) * S, (dy + r) * S),
                  fill=COPPER_2 + (int(255 * (0.7 + 0.3 * pulse) * k),))
        tx, ty = L["clock"]
        d.text((tx * S, ty * S), clock, font=font(round(14 * scale * S)), fill=MINT + (int(235 * k),), anchor="lm")
        sx, sy = L["stop"]
        sr = L["stop_r"] * S
        ring = VERDIGRIS_2 + (int(210 * k),) if hover == "stop" else VERDIGRIS + (int(150 * k),)
        d.ellipse((sx * S - sr, sy * S - sr, sx * S + sr, sy * S + sr), fill=ring)
        q = 4.4 * scale * S
        d.rounded_rectangle((sx * S - q, sy * S - q, sx * S + q, sy * S + q), radius=1.6 * scale * S,
                            fill=MINT + (int(240 * k),))
    elif view in ("listening", "working") and k > 0 and heights:
        bw, gap = BAR_W * scale * S, BAR_GAP * scale * S
        n = len(heights)
        left = cx * S - (n * bw + (n - 1) * gap) / 2
        full = (y1 - y0 - 14 * scale) * S
        for i, v in enumerate(heights):
            if view == "working":
                hh = bw + v * (full * 0.28 - bw)
                colour, a = SAGE, 0.55 + 0.45 * v
            else:
                hh = bw + v * (full - bw)
                colour, a = MINT, 0.92
            bx = left + i * (bw + gap)
            d.rounded_rectangle((bx, cy * S - hh / 2, bx + bw, cy * S + hh / 2), radius=bw / 2,
                                fill=colour + (int(255 * a * k),))
    elif view == "error" and k > 0:
        r, lw = 6 * scale * S, max(1.0, 1.6 * scale) * S
        a = int(255 * k)
        d.line((cx * S - r, cy * S - r, cx * S + r, cy * S + r), fill=GILT + (a,), width=round(lw))
        d.line((cx * S - r, cy * S + r, cx * S + r, cy * S - r), fill=GILT + (a,), width=round(lw))
    _put(img, big)
    return fade(img, opacity)


def widget_hit(scale: float, w: float, h: float, view: str, x: float, y: float) -> str | None:
    """The control at window point (x, y): "start", "stop" or None."""
    rect = widget_rect(scale, w, h)
    x0, y0, x1, y1 = rect
    if view == "start":
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 2 * scale
        return "start" if (x - cx) ** 2 + (y - cy) ** 2 <= r * r else None
    if view == "rec":
        L = rec_layout(rect, scale)
        sx, sy = L["stop"]
        r = L["stop_r"] + 3 * scale
        return "stop" if (x - sx) ** 2 + (y - sy) ** 2 <= r * r else None
    return None


# ---------------------------------------------------------------- vault picker (just above the widget)

PICK_H = 40
CHIP_H = 28


def picker_layout(scale: float, names: list[str]) -> dict:
    """Canvas size, pill rect, one rect per vault chip, and the countdown ring's centre (physical px)."""
    m, h = MARGIN * scale, PICK_H * scale
    pad, gap, cpad = 6 * scale, 4 * scale, 13 * scale
    px = round(13 * scale * SS)
    widths = [font(px).getlength(n) / SS + 2 * cpad for n in names]
    ring = 26 * scale
    w = pad + sum(widths) + gap * (len(names) - 1) + 10 * scale + ring + 7 * scale
    cw, ch = int(math.ceil(w + 2 * m)), int(math.ceil(h + 2 * m))
    rect = (round(m), round(m), round(m + w), round(m + h))
    cy = (rect[1] + rect[3]) / 2
    chips, x = [], rect[0] + pad
    for cwid in widths:
        chips.append((x, cy - CHIP_H * scale / 2, x + cwid, cy + CHIP_H * scale / 2))
        x += cwid + gap
    return {"size": (cw, ch), "rect": rect, "chips": chips, "ring": (rect[2] - 7 * scale - ring / 2, cy),
            "ring_r": ring / 2}


def picker(scale: float, names: list[str], default: str, hover: int | None, remaining: int,
           fraction: float) -> Image.Image:
    """The vault picker: one chip per vault (the default one lit), and the seconds left inside a ring
    that empties as the countdown runs."""
    L = picker_layout(scale, names)
    img = body(L["size"], L["rect"], round(scale, 3)).copy()
    big, d = _layer(L["size"])
    f = font(round(13 * scale * SS))
    for i, (name, (x0, y0, x1, y1)) in enumerate(zip(names, L["chips"])):
        box = (x0 * SS, y0 * SS, x1 * SS, y1 * SS)
        rad = (y1 - y0) / 2 * SS
        if i == hover:
            d.rounded_rectangle(box, radius=rad, fill=VERDIGRIS_2 + (235,))
        elif name == default:
            d.rounded_rectangle(box, radius=rad, fill=VERDIGRIS + (215,))
        if name == default:
            d.rounded_rectangle(box, radius=rad, outline=GILT + (110,), width=max(1, round(scale * SS * 0.8)))
        d.text(((x0 + x1) / 2 * SS, (y0 + y1) / 2 * SS), name, font=f,
               fill=MINT + (245 if (i == hover or name == default) else 200,), anchor="mm")
    rx, ry = (v * SS for v in L["ring"])
    rr = L["ring_r"] * SS
    lw = max(1, round(1.6 * scale * SS))
    d.ellipse((rx - rr, ry - rr, rx + rr, ry + rr), outline=VERDIGRIS_2 + (200,), width=lw)
    if fraction > 0:
        d.arc((rx - rr, ry - rr, rx + rr, ry + rr), start=-90, end=-90 + 360 * min(1.0, fraction),
              fill=GILT + (220,), width=lw)
    d.text((rx, ry), str(remaining), font=font(round(11.5 * scale * SS)), fill=MINT + (210,), anchor="mm")
    _put(img, big)
    return img


def hit_chip(scale: float, names: list[str], x: float, y: float) -> int | None:
    for i, (x0, y0, x1, y1) in enumerate(picker_layout(scale, names)["chips"]):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
    return None


# ---------------------------------------------------------------- to the screen


def premultiplied_bgra(img: Image.Image, min_alpha: int = CLICK_ALPHA) -> bytes:
    """Top-down BGRA bytes with colour premultiplied by alpha, as UpdateLayeredWindow's AC_SRC_ALPHA wants.
    Pixels fainter than min_alpha become fully transparent, so the faint edge of a shadow never takes a click."""
    a = np.asarray(img.convert("RGBA"), np.uint16)
    alpha = a[..., 3:4].copy()
    alpha[alpha < min_alpha] = 0
    rgb = (a[..., :3] * alpha + 127) // 255
    return np.concatenate([rgb[..., ::-1], alpha], axis=2).astype(np.uint8).tobytes()
