"""Labelings, walk products, reframings and classes (glossary sections 2-3).

A labeling ``U`` is an array of shape ``(*batch, n_edges, *G.elem_shape)``: one
label per edge, for the edge's reference orientation s -> t. The label of the
reverse direction is the inverse (glossary 2.2). Leading ``batch`` axes are
optional and let you process many labelings at once.

A reframing ``h`` has shape ``(*batch, n_vertices, *G.elem_shape)``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .board import Board
from .groups import Group


def _axis(G: Group, X) -> int:
    """Index of the edge (or vertex) axis of X."""
    return np.ndim(X) - 1 - len(G.elem_shape)


def _take(G: Group, X, idx):
    return np.take(np.asarray(X), np.asarray(idx), axis=_axis(G, X))


def _step_labels(G: Group, U, walk):
    """Labels met along a walk, inverted on reversed steps. Shape (*batch, k, *elem)."""
    edges = [e for e, _ in walk]
    signs = np.array([s for _, s in walk])
    x = _take(G, U, edges)
    xinv = G.inv(x)
    mask = (signs < 0).reshape((-1,) + (1,) * len(G.elem_shape))
    return np.where(mask, xinv, x)


# ---------------------------------------------------------------------------
# Products (2.3 - 2.5)
# ---------------------------------------------------------------------------


def walk_product(G: Group, board: Board, U, walk):
    """U(d1) U(d2) ... U(dk) for a walk given as (e, sign) steps or as a vertex list."""
    if len(walk) and not isinstance(walk[0], tuple):
        walk = board.walk(list(walk))
    x = _step_labels(G, U, walk)
    return G.prod(x, axis=_axis(G, x))


def loop_product(G: Group, board: Board, U, closed_walk):
    """Walk product of a closed walk (glossary 2.4). Same as walk_product, but checks closure."""
    if len(closed_walk) and not isinstance(closed_walk[0], tuple):
        closed_walk = board.walk(list(closed_walk))
    vs = board.walk_vertices(closed_walk)
    if vs[0] != vs[-1]:
        raise ValueError("walk is not closed")
    return walk_product(G, board, U, closed_walk)


def face_products(G: Group, board: Board, U):
    """All face products, shape (*batch, n_faces, *elem) (glossary 2.5)."""
    U = np.asarray(U)
    lengths = {len(f) for f in board.faces}
    if len(lengths) == 1:  # vectorized fast path (e.g. grids)
        k = lengths.pop()
        edges = np.array([[e for e, _ in f] for f in board.faces])  # (F, k)
        signs = np.array([[s for _, s in f] for f in board.faces])
        ax = _axis(G, U)
        x = np.take(U, edges, axis=ax)  # (*batch, F, k, *elem)
        mask = (signs < 0).reshape(signs.shape + (1,) * len(G.elem_shape))
        x = np.where(mask, G.inv(x), x)
        return G.prod(x, axis=ax + 1)
    out = [walk_product(G, board, U, f) for f in board.faces]
    return np.stack(out, axis=_axis(G, U))


def defects(G: Group, board: Board, U, atol=1e-9):
    """Boolean array: which faces are defects (face product != identity)."""
    return ~G.is_identity(face_products(G, board, U), atol=atol)


# ---------------------------------------------------------------------------
# Reframings (3.1)
# ---------------------------------------------------------------------------


def reframe(G: Group, board: Board, U, h):
    """Apply the reframing h: U(s->t) -> h(s) U(s->t) h(t)^{-1}."""
    hs = _take(G, h, board.edges[:, 0])
    ht = _take(G, h, board.edges[:, 1])
    return G.mul(G.mul(hs, U), G.inv(ht))


def random_labeling(G: Group, board: Board, rng, batch=()):
    batch = (batch,) if isinstance(batch, int) else tuple(batch)
    return G.uniform(rng, batch + (board.n_edges,))


def random_reframing(G: Group, board: Board, rng, batch=()):
    batch = (batch,) if isinstance(batch, int) else tuple(batch)
    return G.uniform(rng, batch + (board.n_vertices,))


def identity_labeling(G: Group, board: Board):
    return G.identity((board.n_edges,))


# ---------------------------------------------------------------------------
# Tree normal form and consistency (3.8, 2.6)
# ---------------------------------------------------------------------------


def tree_normal_form(G: Group, board: Board, U, root: int = 0):
    """Reframe U so that every spanning-tree edge carries the identity.

    Returns (U_normal, h, tree_edges) with U_normal == reframe(G, board, U, h).
    h(root) = identity and h(v) = walk product from the root to v along the tree,
    so the label left on a non-tree edge e is the loop product of e's fundamental
    cycle (root -> s, e, t -> root).
    """
    U = np.asarray(U)
    tree, order, parent_step = board.spanning_tree(root)
    batch = U.shape[: _axis(G, U)]
    h = G.identity(batch + (board.n_vertices,)).copy()
    ax = _axis(G, h)
    for v in order[1:]:
        e, sign = parent_step[v]
        p = board.step_endpoints(e, sign)[0]
        lab = np.take(U, e, axis=_axis(G, U))
        lab = lab if sign > 0 else G.inv(lab)
        hv = G.mul(np.take(h, p, axis=ax), lab)
        idx = [slice(None)] * h.ndim
        idx[ax] = v
        h[tuple(idx)] = hv
    # Tree edge p->v becomes h(p) U h(v)^{-1} = h(p) U (h(p) U)^{-1} = identity.
    return reframe(G, board, U, h), h, tree


def is_consistent(G: Group, board: Board, U, atol=1e-9):
    """True iff every loop product is the identity (glossary 2.6).

    Uses the tree normal form: U is consistent iff its tree normal form is the
    identity labeling.
    """
    Un, _, _ = tree_normal_form(G, board, U)
    ok = G.is_identity(Un, atol=atol)
    return np.all(ok, axis=-1)


# ---------------------------------------------------------------------------
# Finite groups: enumeration and classes (3.3)
# ---------------------------------------------------------------------------


def all_labelings(G: Group, board: Board, max_count: int = 2**22):
    """Every labeling of a board with a finite group, shape (|G|^E, E)."""
    els = G.elements()
    count = len(els) ** board.n_edges
    if count > max_count:
        raise ValueError(f"{count} labelings is too many to enumerate")
    idx = np.array(list(itertools.product(range(len(els)), repeat=board.n_edges)))
    return els[idx] if len(idx) else np.empty((1, 0), dtype=els.dtype)


def all_reframings(G: Group, board: Board, max_count: int = 2**22):
    els = G.elements()
    count = len(els) ** board.n_vertices
    if count > max_count:
        raise ValueError(f"{count} reframings is too many to enumerate")
    idx = np.array(list(itertools.product(range(len(els)), repeat=board.n_vertices)))
    return els[idx]


def class_ids(G: Group, board: Board, Us):
    """Integer class label for each labeling in a batch (finite groups only).

    Two labelings get the same id iff some reframing maps one to the other.
    Abelian: the tree normal form is a complete invariant. Non-abelian: the tree
    normal form is determined up to one global conjugation, so we take the
    lexicographically smallest conjugate.
    """
    Un, _, _ = tree_normal_form(G, board, np.asarray(Us))
    els = np.sort(G.elements())
    n = len(els)
    if n ** board.n_edges >= 2**62:
        raise ValueError("board too large to encode classes")
    weights = n ** np.arange(board.n_edges, dtype=np.int64)

    def encode(X):
        return (np.searchsorted(els, X).astype(np.int64) * weights).sum(axis=-1)

    codes = encode(Un)
    if not G.is_abelian:
        codes = np.stack([encode(G.conj(g, Un)) for g in G.elements()], axis=1).min(axis=1)
    _, ids = np.unique(codes, return_inverse=True)
    return ids.reshape(-1)
