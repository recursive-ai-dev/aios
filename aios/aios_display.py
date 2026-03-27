#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Display Manager                                                      ║
║  aios_display.py                                                             ║
║                                                                              ║
║  "Light is information. Every pixel is a decision. Every frame, a proof."   ║
║                                                                              ║
║  Rendering Pipeline:                                                         ║
║    InputDriver → EventQueue → WindowManager → Compositor → DisplayDriver    ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Enums   — DisplayMode, PixelFormat, EventType, KeyCode   ║
║    §1  Math Primitives     — clamp, lerp, Bresenham, Porter-Duff, bilinear  ║
║    §2  Color System        — Color, VGA16 palette, ANSI256 mapping          ║
║    §3  Font Engine         — CP437 8×8 bitmap font (embedded), glyph ops   ║
║    §4  Surface             — ARGB32 pixel buffer, draw ops, double-buffer   ║
║    §5  Compositor          — Z-order, DirtyRect AABB, Porter-Duff blending ║
║    §6  Widget System       — Label, Panel, ProgressBar, TextInput, Border   ║
║    §7  Event System        — EventQueue, InputDriver, VT200 mouse, raw tty  ║
║    §8  Display Drivers     — ANSITerminal (half-block), VGAAdapter, FB      ║
║    §9  Window Manager      — Window lifecycle, focus, z-order, decorations  ║
║    §10 Display Manager     — Top-level orchestrator, @agent_method hooks    ║
║    §11 Self-Test           — Surface, compositor, window, render pipeline   ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Bresenham    : err = 2·dy − dx, step by sign(dx)  [Bresenham 1965]       ║
║    Porter-Duff  : C_out = αₛ·Cₛ + (1−αₛ)·αd·Cd  [Porter & Duff 1984]      ║
║    Bilinear     : f(x,y) = Σᵢⱼ f(i,j)·max(0,1−|x−i|)·max(0,1−|y−j|)      ║
║    AABB union   : R' = (min(x₀,x₁), min(y₀,y₁), max(x₂,x₃), max(y₂,y₃))  ║
║    ANSI half-blk: pair rows (2r, 2r+1), fg=top-pixel, bg=bot-pixel [▀ U+2580]║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Every computation traceable to a named equation above.                  ║
║    • Thread-safe: all mutable state guarded by threading.RLock.              ║
║    • Zero external dependencies. Pure Python 3.9+ stdlib only.              ║
║    • Standalone: runs without aios_core if not on path.                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import time
import struct
import array
import select
import threading
import functools
import hashlib
import termios
import tty
import fcntl
import traceback
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, auto
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
)

# ── AIOS kernel integration (optional) ───────────────────────────────────────
try:
    from aios_core import agent_method, AgentPriority, AgentTrace, AgentContext
    _AIOS_INTEGRATED: bool = True
except ImportError:
    _AIOS_INTEGRATED = False

    class AgentPriority(IntEnum):  # type: ignore[no-redef]
        CRITICAL = 0; HIGH = 1; NORMAL = 2; LOW = 3

    def agent_method(  # type: ignore[no-redef]
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict] = None,
        returns: str = "Any",
        priority: Any = None,
        owner: str = "display",
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("_ctx", None)
                return fn(*args, **kwargs)
            return wrapper
        return decorator


# ════════════════════════════════════════════════════════════════════════════
#  §0 — CONSTANTS & ENUMS
#  All numeric constants derive from their respective standards.
#  Color indices follow ANSI X3.64, VGA text-mode EGA 4-bit palette,
#  and the xterm-256color table (8+8 named + 216 6×6×6 cube + 24 grays).
# ════════════════════════════════════════════════════════════════════════════

DISPLAY_VERSION   = (1, 0, 0)
FONT_W            = 8          # CP437 glyph width  in pixels
FONT_H            = 8          # CP437 glyph height in pixels
MAX_WINDOWS       = 64
DIRTY_POOL_MAX    = 256        # max dirty rects before full-screen flush
HALF_BLOCK        = '\u2580'   # ▀ U+2580: used for half-block rendering
FULL_BLOCK        = '\u2588'   # █ U+2588

# ANSI escape sequences
_ESC  = '\x1b'
_CSI  = '\x1b['
_RESET = '\x1b[0m'


class DisplayMode(IntEnum):
    """Output rendering mode, chosen at driver construction."""
    ANSI_TERMINAL  = 0   # default: ANSI/VT100 escapes to stdout
    VGA_TEXT       = 1   # direct VGA text-mode buffer (via aios_core.VGATextDriver)
    FRAMEBUFFER    = 2   # mmap-backed /dev/fb0 pixel buffer


class PixelFormat(IntEnum):
    """In-memory pixel layout for Surface buffers."""
    ARGB32 = 0   # 4 bytes: A(7:0) R(7:0) G(7:0) B(7:0) — native
    RGBA32 = 1   # 4 bytes: R G B A (OpenGL convention)
    BGR24  = 2   # 3 bytes: B G R   (BMP/Windows convention)
    RGB16  = 3   # 2 bytes: R(4:0) G(5:0) B(4:0) (RGB565)


class EventType(IntEnum):
    KEY_DOWN    = 0
    KEY_UP      = 1
    MOUSE_MOVE  = 2
    MOUSE_DOWN  = 3
    MOUSE_UP    = 4
    MOUSE_WHEEL = 5
    RESIZE      = 6
    FOCUS_IN    = 7
    FOCUS_OUT   = 8
    QUIT        = 9


class KeyCode(IntEnum):
    """Extended key codes for non-printable keys."""
    UNKNOWN   = 0
    UP        = 256
    DOWN      = 257
    LEFT      = 258
    RIGHT     = 259
    HOME      = 260
    END       = 261
    PAGE_UP   = 262
    PAGE_DOWN = 263
    INSERT    = 264
    DELETE    = 265
    F1        = 271
    F2        = 272
    F3        = 273
    F4        = 274
    F5        = 275
    F6        = 276
    F7        = 277
    F8        = 278
    F9        = 279
    F10       = 280
    F11       = 281
    F12       = 282
    ENTER     = 13
    BACKSPACE = 127
    ESCAPE    = 27
    TAB       = 9


class MouseButton(IntEnum):
    LEFT   = 0
    MIDDLE = 1
    RIGHT  = 2
    WHEEL_UP   = 64
    WHEEL_DOWN = 65


class WindowState(IntEnum):
    NORMAL    = 0
    MINIMIZED = 1
    MAXIMIZED = 2
    HIDDEN    = 3


# ════════════════════════════════════════════════════════════════════════════
#  §1 — MATH PRIMITIVES
#  No import of math, numpy, or any numeric library.
#  All formulas are cited by name and year.
# ════════════════════════════════════════════════════════════════════════════

def _clamp(x: int, lo: int, hi: int) -> int:
    """Clamp integer x to [lo, hi] inclusive."""
    return lo if x < lo else (hi if x > hi else x)


def _clampf(x: float, lo: float, hi: float) -> float:
    """Clamp float x to [lo, hi] inclusive."""
    return lo if x < lo else (hi if x > hi else x)


def _lerp(a: float, b: float, t: float) -> float:
    """
    Eq. LERP-1: Linear interpolation.
      f(a, b, t) = a + t·(b − a),  t ∈ [0,1]
    """
    return a + t * (b - a)


def _ilerp(a: int, b: int, t256: int) -> int:
    """
    Integer linear interpolation with t scaled to [0, 256].
    Avoids float division for blending inner loops.
    Eq. ILERP-1: f = a + (t256·(b−a)) >> 8
    """
    return a + ((t256 * (b - a)) >> 8)


def _isqrt(n: int) -> int:
    """
    Integer square root via Newton-Raphson iteration.
    Eq. ISQRT-NR: x_{k+1} = (x_k + n/x_k) // 2
    Converges in O(log log n) steps from initial estimate.
    """
    if n < 0:
        raise ValueError("isqrt requires non-negative integer")
    if n == 0:
        return 0
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x


def _alpha_blend(
    sr: int, sg: int, sb: int, sa: int,
    dr: int, dg: int, db: int, da: int,
) -> Tuple[int, int, int, int]:
    """
    Porter-Duff 'Over' compositing operator [Porter & Duff 1984].
    Eq. PD-OVER:
      C_out = αₛ·Cₛ + (1 − αₛ)·αd·Cd
      α_out = αₛ + (1 − αₛ)·αd

    All channels pre-divided by 255 are handled via integer arithmetic
    with a 256-unit fixed-point scale for performance.
    """
    if sa == 255:
        return sr, sg, sb, 255
    if sa == 0:
        return dr, dg, db, da
    # Scale: s/255 ≈ s * 257 >> 16  (accurate for 0–255)
    inv_sa = 255 - sa
    out_a  = sa + ((inv_sa * da) >> 8)
    if out_a == 0:
        return 0, 0, 0, 0
    # Eq. PD-OVER applied per channel, with fixed-point 1/255 ≈ 1/256
    out_r = _clamp((sa * sr + inv_sa * dr) >> 8, 0, 255)
    out_g = _clamp((sa * sg + inv_sa * dg) >> 8, 0, 255)
    out_b = _clamp((sa * sb + inv_sa * db) >> 8, 0, 255)
    return out_r, out_g, out_b, _clamp(out_a, 0, 255)


def _bresenham_line(
    x0: int, y0: int, x1: int, y1: int
) -> List[Tuple[int, int]]:
    """
    Bresenham's integer line algorithm [Bresenham 1965].
    Eq. BRESENHAM-1965:
      error_init = 2·|dy| − |dx|
      At each x-step: if error >= 0: y += sign(dy), error -= 2·|dx|
                                     error += 2·|dy|
    Returns list of (x, y) pixel coordinates — no floats, no divisions.
    """
    pts: List[Tuple[int, int]] = []
    dx  = abs(x1 - x0)
    dy  = abs(y1 - y0)
    sx  = 1 if x0 < x1 else -1
    sy  = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy
    return pts


def _bilinear_sample(
    buf: bytearray, w: int, h: int, bpp: int,
    fx: float, fy: float,
) -> Tuple[int, int, int, int]:
    """
    Bilinear interpolation for sub-pixel sampling.
    Eq. BILINEAR-2D: f(x,y) = Σᵢ∈{0,1} Σⱼ∈{0,1} f(x0+i, y0+j)·wᵢ·wⱼ
      where w₀ = 1 − frac,  w₁ = frac
    Returns (r, g, b, a) as uint8 tuple.
    bpp must be 4 (ARGB32 or RGBA32).
    """
    x0 = int(fx);  x1 = min(x0 + 1, w - 1)
    y0 = int(fy);  y1 = min(y0 + 1, h - 1)
    tx = fx - x0;  ty = fy - y0
    # Four corner samples
    def px(xi: int, yi: int) -> Tuple[int, int, int, int]:
        off = (yi * w + xi) * bpp
        return buf[off], buf[off+1], buf[off+2], buf[off+3]
    tl = px(x0, y0); tr = px(x1, y0)
    bl = px(x0, y1); br = px(x1, y1)
    tx256 = int(tx * 256)
    ty256 = int(ty * 256)
    # Bilinear blend per channel
    def bl_ch(tl_c: int, tr_c: int, bl_c: int, br_c: int) -> int:
        top = _ilerp(tl_c, tr_c, tx256)
        bot = _ilerp(bl_c, br_c, tx256)
        return _ilerp(top, bot, ty256)
    return (
        bl_ch(tl[0], tr[0], bl[0], br[0]),
        bl_ch(tl[1], tr[1], bl[1], br[1]),
        bl_ch(tl[2], tr[2], bl[2], br[2]),
        bl_ch(tl[3], tr[3], bl[3], br[3]),
    )


@dataclass
class Rect:
    """Axis-aligned bounding box used for dirty-rect tracking and clipping."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int: return self.x + self.w
    @property
    def y2(self) -> int: return self.y + self.h

    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def intersects(self, other: "Rect") -> bool:
        return (self.x < other.x2 and self.x2 > other.x and
                self.y < other.y2 and self.y2 > other.y)

    def clip_to(self, bounds: "Rect") -> "Rect":
        """Return sub-rect clipped to bounds. Eq. AABB-CLIP."""
        nx = max(self.x, bounds.x)
        ny = max(self.y, bounds.y)
        nx2 = min(self.x2, bounds.x2)
        ny2 = min(self.y2, bounds.y2)
        return Rect(nx, ny, max(0, nx2 - nx), max(0, ny2 - ny))

    @staticmethod
    def union(a: "Rect", b: "Rect") -> "Rect":
        """
        AABB union. Eq. AABB-UNION:
          R' = (min(a.x, b.x), min(a.y, b.y),
                max(a.x2, b.x2) − min(a.x, b.x),
                max(a.y2, b.y2) − min(a.y, b.y))
        """
        if a.is_empty(): return b
        if b.is_empty(): return a
        nx  = min(a.x,  b.x)
        ny  = min(a.y,  b.y)
        nx2 = max(a.x2, b.x2)
        ny2 = max(a.y2, b.y2)
        return Rect(nx, ny, nx2 - nx, ny2 - ny)


# ════════════════════════════════════════════════════════════════════════════
#  §2 — COLOR SYSTEM
#  VGA 16-color palette: EGA standard bit-coded RGB (IBM Technical Reference).
#  ANSI 256-color: 8+8 named + 216-color 6×6×6 RGB cube + 24-step grayscale.
#  RGB→ANSI256 mapping uses min-distance in RGB³ Euclidean space.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Color:
    """32-bit ARGB color. All channels in [0, 255]."""
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255

    def to_argb32(self) -> int:
        """Pack to 32-bit ARGB integer."""
        return ((self.a & 0xFF) << 24 | (self.r & 0xFF) << 16 |
                (self.g & 0xFF) << 8  | (self.b & 0xFF))

    @staticmethod
    def from_argb32(v: int) -> "Color":
        return Color(
            r=(v >> 16) & 0xFF,
            g=(v >>  8) & 0xFF,
            b=(v      ) & 0xFF,
            a=(v >> 24) & 0xFF,
        )

    @staticmethod
    def from_hex(h: str) -> "Color":
        """Parse '#RRGGBB' or '#RRGGBBAA'."""
        h = h.lstrip('#')
        if len(h) == 6:
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return Color(r, g, b)
        r, g, b, a = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), int(h[6:8],16)
        return Color(r, g, b, a)

    def blend(self, other: "Color") -> "Color":
        """Porter-Duff Over: self over other."""
        r, g, b, a = _alpha_blend(
            self.r, self.g, self.b, self.a,
            other.r, other.g, other.b, other.a,
        )
        return Color(r, g, b, a)

    def __repr__(self) -> str:
        return f"Color(#{self.r:02X}{self.g:02X}{self.b:02X}:{self.a:02X})"


# VGA 16-color palette (EGA standard — IBM Technical Reference Manual, 1984)
_VGA16: List[Color] = [
    Color(  0,   0,   0),  # 0  BLACK
    Color(  0,   0, 170),  # 1  BLUE
    Color(  0, 170,   0),  # 2  GREEN
    Color(  0, 170, 170),  # 3  CYAN
    Color(170,   0,   0),  # 4  RED
    Color(170,   0, 170),  # 5  MAGENTA
    Color(170,  85,   0),  # 6  BROWN
    Color(170, 170, 170),  # 7  LIGHT GRAY
    Color( 85,  85,  85),  # 8  DARK GRAY
    Color( 85,  85, 255),  # 9  LIGHT BLUE
    Color( 85, 255,  85),  # 10 LIGHT GREEN
    Color( 85, 255, 255),  # 11 LIGHT CYAN
    Color(255,  85,  85),  # 12 LIGHT RED
    Color(255,  85, 255),  # 13 LIGHT MAGENTA
    Color(255, 255,  85),  # 14 YELLOW
    Color(255, 255, 255),  # 15 WHITE
]

# Named palette for common AIOS UI colors
class Palette:
    BLACK         = Color(  0,   0,   0)
    WHITE         = Color(255, 255, 255)
    DARK_BG       = Color( 20,  20,  30)
    PANEL_BG      = Color( 30,  30,  45)
    BORDER        = Color( 80,  80, 120)
    BORDER_ACTIVE = Color(100, 140, 220)
    TEXT          = Color(220, 220, 230)
    TEXT_DIM      = Color(120, 120, 140)
    ACCENT        = Color( 80, 160, 255)
    SUCCESS       = Color( 80, 200, 100)
    WARNING       = Color(220, 180,  60)
    ERROR         = Color(220,  80,  80)
    SELECTION     = Color( 50,  80, 130)
    TRANSPARENT   = Color(0, 0, 0, 0)


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """
    Map (r,g,b) to nearest xterm-256 color index.
    Strategy:
      1. Check 24-step grayscale ramp (indices 232–255): values 8,18,...,238
      2. Check 6×6×6 RGB cube (indices 16–231): each channel ∈ {0,95,135,175,215,255}
      3. Return index with minimum squared Euclidean distance in RGB³.
    """
    # Grayscale ramp: index k → value 8 + 10*(k-232), k in 232..255
    def _gray_val(k: int) -> int:
        return 8 + 10 * (k - 232)

    # 6-step cube levels
    _CUBE = (0, 95, 135, 175, 215, 255)

    def _sq(x: int) -> int: return x * x

    best_idx  = 0
    best_dist = 0x7FFFFFFF

    # Check all 256 entries
    # Named 0–