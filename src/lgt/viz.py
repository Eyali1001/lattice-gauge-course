"""Shared visual language for the course.

Conventions used in every chapter (keep them consistent!):

* Vertices are small dark dots. Edges are drawn with a small arrow showing the
  reference orientation s -> t.
* Edge labels: Z2 -> edge colour (+1 dark, -1 gold and thick); circle groups (U1, ZN)
  -> a small *clock hand* at the edge midpoint; everything else -> text.
* Faces are filled by their face product: pale = identity, coral = defect (Z2/S3),
  cyclic colour map for angles (U1/ZN), diverging colour map for exchange rates.
* Spanning-tree edges are drawn in light grey; highlighted walks in teal.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch

from .board import Board
from .groups import Group, Z2, ZN, U1, S3, SO3, PositiveReals, TAU
from .labeling import face_products, reframe

INK = "#2b2d42"
MUTED = "#8d99ae"
PALE = "#eef3f6"
TEAL = "#1b998b"
CORAL = "#e4572e"
GOLD = "#f3a712"
TREE = "#c9ced6"
PALETTE = dict(ink=INK, muted=MUTED, pale=PALE, teal=TEAL, coral=CORAL, gold=GOLD, tree=TREE)


def use_style():
    """Matplotlib defaults for the course."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[TEAL, CORAL, GOLD, INK, MUTED]),
        "figure.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

def format_label(G: Group, x, digits: int = 2) -> str:
    if isinstance(G, Z2):
        return "+1" if int(x) == 1 else "−1"
    if isinstance(G, ZN):
        return str(int(x))
    if isinstance(G, U1):
        return f"{float(x) / np.pi:.{digits}f}π"
    if isinstance(G, PositiveReals):
        return f"{float(x):.{digits + 1}g}"
    if isinstance(G, S3):
        return perm_cycle_notation(G.perm(x))
    if isinstance(G, SO3):
        return f"{np.degrees(G.rotation_angle(x)):.0f}°"
    return str(x)


def perm_cycle_notation(p) -> str:
    """Cycle notation with symbols 1,2,3, e.g. (0,2,1) -> '(23)'. Identity -> 'e'."""
    seen, cycles = set(), []
    for i in range(len(p)):
        if i in seen:
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(j + 1)
            j = p[j]
        if len(c) > 1:
            cycles.append("(" + "".join(map(str, c)) + ")")
    return "".join(cycles) or "e"


# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------

def draw_clock(ax, xy, angle, r=0.11, color=INK, face=PALE, lw=1.6, zorder=6):
    """A small clock: circle with one hand pointing at ``angle`` (0 = right, CCW)."""
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=color, lw=0.8, zorder=zorder))
    ax.plot([xy[0], xy[0] + 0.85 * r * np.cos(angle)],
            [xy[1], xy[1] + 0.85 * r * np.sin(angle)],
            color=color, lw=lw, solid_capstyle="round", zorder=zorder + 1)


def _arrow(ax, p, q, color, lw, zorder=3, frac=0.62, size=11):
    """Line p->q with a small arrowhead at ``frac`` of the way."""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round")
    if size:
        a = p + (frac - 0.08) * (q - p)
        b = p + frac * (q - p)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=size,
                                     color=color, lw=0, zorder=zorder + 0.5))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _edge_segments(board: Board):
    """Drawing segments for each edge: list of [(p, q), ...] (two stubs for wrapped edges)."""
    pos = board.pos[:, :2]
    segs = []
    emu = {}
    if "edge_index" in board.meta:
        emu = {e: mu for (v, mu), e in board.meta["edge_index"].items()}
    for e, (s, t) in enumerate(board.edges):
        p, q = pos[s], pos[t]
        # wrapped edge on a periodic grid: it steps "forward" but lands at a smaller coordinate
        if e in emu and board.pos[t][emu[e]] < board.pos[s][emu[e]]:
            d = np.eye(board.pos.shape[1])[emu[e]][:2]
            segs.append([(p, p + 0.5 * d), (q - 0.5 * d, q)])
        else:
            segs.append([(p, q)])
    return segs


def _face_colors(G: Group, fp, face_style="auto", cmap=None, vmax=None):
    """Colours for faces from their face products."""
    if isinstance(G, (Z2, S3)):
        ident = G.is_identity(fp)
        return [PALE if i else CORAL for i in ident], None
    if isinstance(G, (U1, ZN)):
        cm = plt.get_cmap(cmap or "twilight")
        ang = G.angle(fp)
        return [cm(a / TAU) for a in ang], (cm, mpl.colors.Normalize(0, TAU))
    if isinstance(G, PositiveReals):
        cm = plt.get_cmap(cmap or "RdBu_r")
        lg = np.log(fp)
        m = vmax or max(1e-9, np.abs(lg).max())
        norm = mpl.colors.TwoSlopeNorm(0, -m, m)
        return [cm(norm(v)) if abs(v) > 1e-9 else PALE for v in lg], (cm, norm)
    if isinstance(G, SO3):
        cm = plt.get_cmap(cmap or "magma_r")
        a = G.rotation_angle(fp)
        norm = mpl.colors.Normalize(0, np.pi)
        return [cm(norm(v)) for v in a], (cm, norm)
    return [PALE] * len(fp), None


def draw_board(ax, board: Board, U=None, G: Group | None = None, *,
               edge_style: str = "auto", faces: bool = True, face_values=None,
               face_cmap=None, face_norm=None, tree_edges=(), highlight_walk=None,
               highlight_color=TEAL, edge_labels=None, vertex_labels=None,
               show_arrows: bool = True, vertex_size: float = 28, clock_r: float = 0.11,
               label_fontsize: float = 9, pad: float = 0.35, title: str | None = None):
    """Draw a 2D board, optionally with a labeling U of group G.

    edge_style: 'auto' | 'clock' | 'text' | 'color' | 'none'
    face_values: optional numbers per face (overrides colouring by face product).
    tree_edges: edges drawn in light grey (e.g. a spanning tree in normal form).
    highlight_walk: list of (e, sign) steps drawn in teal on top.
    edge_labels: optional list/dict of strings per edge (overrides formatting).
    vertex_labels: optional list of strings per vertex (default: none).
    Returns the colour-map info (cmap, norm) used for faces, or None.
    """
    pos = board.pos[:, :2]
    tree_edges = set(int(e) for e in tree_edges)
    segs = _edge_segments(board)
    cinfo = None

    # faces
    if faces and board.face_polys is not None and board.n_faces:
        polys = [np.asarray(p)[:, :2] for p in board.face_polys]
        if face_values is not None:
            cm = plt.get_cmap(face_cmap or "viridis")
            norm = face_norm or mpl.colors.Normalize(np.min(face_values), np.max(face_values))
            cols = [cm(norm(v)) for v in face_values]
            cinfo = (cm, norm)
        elif U is not None and G is not None:
            cols, cinfo = _face_colors(G, face_products(G, board, U), cmap=face_cmap)
        else:
            cols = [PALE] * len(polys)
        # periodic grids: also draw each face shifted back by one period, so the window
        # (cut at the half-edge, see below) shows wrapped faces on every side
        shifts = [np.zeros(2)]
        per = board.meta.get("periodic", ())
        for mu in range(min(2, len(per))):
            if per[mu] and board.meta["dims"][mu] >= 2:
                d = np.zeros(2)
                d[mu] = -board.meta["dims"][mu]
                shifts = shifts + [s + d for s in shifts]
        shrink, scols = [], []
        for p, col in zip(polys, cols):
            c = p.mean(axis=0)
            for sh in shifts:
                shrink.append(c + 0.9 * (p - c) + sh)
                scols.append(col)
        ax.add_collection(PolyCollection(shrink, facecolors=scols, edgecolors="none",
                                         zorder=1, alpha=0.9))

    # edges
    if edge_style == "auto" and G is not None:
        edge_style = ("color" if isinstance(G, Z2) else
                      "clock" if isinstance(G, (U1, ZN)) else "text")
    for e, pieces in enumerate(segs):
        color, lw = (TREE, 2.2) if e in tree_edges else (INK, 1.4)
        if U is not None and edge_style == "color" and e not in tree_edges:
            if int(U[e]) == -1:
                color, lw = GOLD, 3.2
        for k, (p, q) in enumerate(pieces):
            _arrow(ax, np.asarray(p), np.asarray(q), color, lw,
                   size=11 if (show_arrows and k == 0) else 0,
                   frac=0.62 if len(pieces) == 1 else 0.7)
        p0, q0 = pieces[0]
        mid = (np.asarray(p0) + np.asarray(q0)) / 2 if len(pieces) == 1 else np.asarray(q0)
        if U is not None and edge_style == "clock":
            draw_clock(ax, mid if len(pieces) == 1 else (np.asarray(p0) + np.asarray(q0)) / 2,
                       float(G.angle(U[e])), r=clock_r,
                       color=MUTED if e in tree_edges else INK)
        txt = None
        if edge_labels is not None:
            txt = edge_labels[e] if not isinstance(edge_labels, dict) else edge_labels.get(e)
        elif U is not None and edge_style == "text":
            txt = format_label(G, U[e])
        if txt:
            pm = (np.asarray(p0) + np.asarray(q0)) / 2
            ax.text(pm[0], pm[1], txt, ha="center", va="center", fontsize=label_fontsize,
                    color=MUTED if e in tree_edges else INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

    # highlighted walk
    if highlight_walk:
        for e, sgn in highlight_walk:
            for p, q in segs[e]:
                p, q = np.asarray(p), np.asarray(q)
                if sgn < 0:
                    p, q = q, p
                _arrow(ax, p, q, highlight_color, 4.5, zorder=4, size=16, frac=0.55)

    # vertices
    ax.scatter(pos[:, 0], pos[:, 1], s=vertex_size, color=INK, zorder=9)
    if vertex_labels is not None:
        c = pos.mean(axis=0)
        for v, lab in enumerate(vertex_labels):
            d = pos[v] - c
            n = np.linalg.norm(d)
            d = d / n if n > 1e-6 else np.array([0.0, -1.0])  # vertex at the centre: label below
            ax.text(*(pos[v] + 0.16 * d + np.array([0, 0.0])), lab, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=INK, zorder=10)

    pts = [pos]
    if faces and board.face_polys is not None and board.n_faces:
        pts += [np.asarray(p)[:, :2] for p in board.face_polys]
    for pieces in segs:
        for p, q in pieces:
            pts.append(np.array([p, q]))
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
    # Periodic grids: cut the picture at the half-edge, so wrapped faces and edges read as
    # "continuing around" instead of floating past the border.
    per = board.meta.get("periodic", ())
    for mu in range(min(2, len(per))):
        if per[mu] and board.meta["dims"][mu] >= 2:
            lo[mu], hi[mu] = pos[:, mu].min() - 0.5, pos[:, mu].max() + 0.5
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return cinfo


def animate_reframing(board: Board, G: Group, U, n_frames: int = 90, seed: int = 0,
                      figsize=(5, 5), interval: int = 70, h_fn=None, **draw_kwargs):
    """Animation: random, smoothly varying reframings applied to U.

    Every edge label changes, every face colour stays put. Returns a
    matplotlib FuncAnimation; in a notebook show it with
    ``HTML(anim.to_jshtml())``.

    ``h_fn(k)`` may supply the reframing for frame k (required for groups
    without a uniform distribution, such as PositiveReals).
    """
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(seed)
    if h_fn is not None:
        h_at = h_fn
    elif isinstance(G, PositiveReals):
        amp = rng.uniform(0.5, 1.2, board.n_vertices)
        phase = rng.uniform(0, TAU, board.n_vertices)
        def h_at(k):
            return np.exp(amp * np.sin(TAU * k / n_frames + phase))
    elif isinstance(G, U1):
        amp = rng.uniform(0.5, 1.0, board.n_vertices) * np.pi
        phase = rng.uniform(0, TAU, board.n_vertices)
        def h_at(k):
            return np.mod(amp * np.sin(TAU * k / n_frames + phase), TAU)
    else:
        hs = [G.uniform(rng, board.n_vertices) for _ in range(n_frames // 10 + 1)]
        def h_at(k):
            return hs[k // 10]

    fig, ax = plt.subplots(figsize=figsize)

    def frame(k):
        ax.clear()
        draw_board(ax, board, reframe(G, board, U, h_at(k)), G, **draw_kwargs)

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=interval)
    plt.close(fig)
    return anim
