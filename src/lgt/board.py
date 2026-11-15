"""Boards: a graph plus a chosen list of closed walks called faces (glossary 1.1-1.3).

Edges are stored with a fixed reference orientation ``edges[e] = (s, t)``. A walk
is a sequence of *steps* ``(e, sign)``: sign=+1 walks edge e from s to t, sign=-1
walks it from t to s (the reverse direction, glossary 1.2).

Multiple edges between the same two vertices are allowed (they occur on small
periodic grids), which is why faces are stored as steps and not as vertex lists.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Board:
    n_vertices: int
    edges: np.ndarray  # (E, 2) int: reference orientation s -> t
    faces: list  # list of faces; each face is a list of (e, sign) steps
    pos: np.ndarray | None = None  # (V, 2 or 3) drawing positions
    face_polys: list | None = None  # optional polygons for drawing faces
    name: str = "board"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.faces = [[(int(e), int(s)) for e, s in f] for f in self.faces]
        for f in self.faces:
            self._check_closed(f)
        if self.pos is not None:
            self.pos = np.asarray(self.pos, dtype=float)

    # --- basic counts -------------------------------------------------------
    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def cycle_rank(self) -> int:
        """|E| - |V| + 1: the number of independent cycles of a connected graph."""
        return self.n_edges - self.n_vertices + 1

    def __repr__(self) -> str:
        return (
            f"Board({self.name!r}: {self.n_vertices} vertices, "
            f"{self.n_edges} edges, {self.n_faces} faces)"
        )

    # --- walks ----------------------------------------------------------------
    def step_endpoints(self, e: int, sign: int) -> tuple[int, int]:
        s, t = self.edges[e]
        return (int(s), int(t)) if sign > 0 else (int(t), int(s))

    def walk_vertices(self, walk) -> list[int]:
        """Vertices visited by a walk given as (e, sign) steps."""
        vs = [self.step_endpoints(*walk[0])[0]]
        for e, sign in walk:
            a, b = self.step_endpoints(e, sign)
            if a != vs[-1]:
                raise ValueError(f"walk is broken at step {(e, sign)}")
            vs.append(b)
        return vs

    def _check_closed(self, walk):
        vs = self.walk_vertices(walk)
        if vs[0] != vs[-1]:
            raise ValueError(f"face {walk} is not closed")

    def edge_between(self, a: int, b: int) -> tuple[int, int]:
        """The unique step from vertex a to vertex b. Errors if 0 or >1 edges join them."""
        fwd = np.flatnonzero((self.edges[:, 0] == a) & (self.edges[:, 1] == b))
        bwd = np.flatnonzero((self.edges[:, 0] == b) & (self.edges[:, 1] == a))
        if len(fwd) + len(bwd) != 1:
            raise ValueError(f"need exactly one edge between {a} and {b}")
        return (int(fwd[0]), 1) if len(fwd) else (int(bwd[0]), -1)

    def walk(self, vertices) -> list[tuple[int, int]]:
        """Convert a vertex sequence [v0, v1, ..., vk] into (e, sign) steps."""
        return [self.edge_between(a, b) for a, b in zip(vertices[:-1], vertices[1:])]

    def reverse_walk(self, walk):
        return [(e, -s) for e, s in reversed(walk)]

    def neighbors(self, v: int) -> list[tuple[int, int, int]]:
        """Steps leaving v, as (e, sign, other_vertex)."""
        out = []
        for e, (s, t) in enumerate(self.edges):
            if s == v:
                out.append((e, 1, int(t)))
            if t == v:
                out.append((e, -1, int(s)))
        return out

    def faces_of_edge(self) -> list[list[int]]:
        """For each edge, the faces whose walk uses it."""
        out = [[] for _ in range(self.n_edges)]
        for fi, f in enumerate(self.faces):
            for e, _ in f:
                if fi not in out[e]:
                    out[e].append(fi)
        return out

    # --- trees ----------------------------------------------------------------
    def spanning_tree(self, root: int = 0):
        """Breadth-first spanning tree.

        Returns (tree_edges, order, parent_step) where ``order`` lists vertices in
        BFS order and ``parent_step[v] = (e, sign)`` is the step from v's parent to v
        (None for the root).
        """
        parent_step: list = [None] * self.n_vertices
        seen = [False] * self.n_vertices
        seen[root] = True
        order = [root]
        q = deque([root])
        tree = []
        nbrs = [self.neighbors(v) for v in range(self.n_vertices)]
        while q:
            v = q.popleft()
            for e, sign, w in nbrs[v]:
                if not seen[w]:
                    seen[w] = True
                    parent_step[w] = (e, sign)
                    tree.append(e)
                    order.append(w)
                    q.append(w)
        if not all(seen):
            raise ValueError("board is not connected")
        return sorted(tree), order, parent_step

    def tree_path(self, v: int, root: int = 0, tree=None):
        """Steps from the root to v along the spanning tree."""
        _, _, parent_step = tree or self.spanning_tree(root)
        steps = []
        while v != root:
            e, sign = parent_step[v]
            steps.append((e, sign))
            v = self.step_endpoints(e, sign)[0]
        return steps[::-1]

    def fundamental_cycles(self, root: int = 0):
        """One closed walk at the root per non-tree edge: root -> s, then e, then t -> root.

        Returns a dict {non_tree_edge: walk}. There are exactly ``cycle_rank`` of them.
        """
        tree_info = self.spanning_tree(root)
        tree = set(tree_info[0])
        out = {}
        for e in range(self.n_edges):
            if e in tree:
                continue
            s, t = (int(x) for x in self.edges[e])
            to_s = self.tree_path(s, root, tree_info)
            from_t = self.reverse_walk(self.tree_path(t, root, tree_info))
            out[e] = to_s + [(e, 1)] + from_t
        return out

    # --- constructors -----------------------------------------------------------
    @classmethod
    def grid(cls, *dims, periodic=False) -> "Board":
        """Hypercubic grid with side lengths ``dims`` (any dimension >= 1).

        Faces are the unit squares, one per vertex and per pair of directions mu < nu,
        walked x -> x+mu -> x+mu+nu -> x+nu -> x. ``periodic`` may be a bool or a
        tuple of bools (wrap each direction).

        Vertex index of coordinate (x0, x1, ...) is x0 + L0*(x1 + L1*(...)).
        ``meta['edge_index'][(v, mu)]`` is the edge leaving v in direction mu.
        """
        dims = tuple(int(d) for d in dims)
        d = len(dims)
        per = (periodic,) * d if isinstance(periodic, bool) else tuple(periodic)
        coords = list(itertools.product(*[range(L) for L in reversed(dims)]))
        coords = [c[::-1] for c in coords]  # x0 fastest

        def vid(c):
            v, mult = 0, 1
            for x, L in zip(c, dims):
                v += x * mult
                mult *= L
            return v

        edges, eidx = [], {}
        for c in coords:
            for mu in range(d):
                nxt = list(c)
                nxt[mu] += 1
                if nxt[mu] == dims[mu]:
                    if not per[mu] or dims[mu] < 2:
                        continue
                    nxt[mu] = 0
                eidx[(vid(c), mu)] = len(edges)
                edges.append((vid(c), vid(nxt)))

        faces, polys, anchors = [], [], []
        for c in coords:
            v = vid(c)
            for mu, nu in itertools.combinations(range(d), 2):
                c_mu = list(c)
                c_mu[mu] = (c_mu[mu] + 1) % dims[mu]
                c_nu = list(c)
                c_nu[nu] = (c_nu[nu] + 1) % dims[nu]
                keys = [(v, mu), (vid(c_mu), nu), (vid(c_nu), mu), (v, nu)]
                if not all(k in eidx for k in keys):
                    continue
                e1, e2, e3, e4 = (eidx[k] for k in keys)
                faces.append([(e1, 1), (e2, 1), (e3, -1), (e4, -1)])
                anchors.append((tuple(c), mu, nu))
                base = np.array(c, dtype=float)
                emu, enu = np.eye(d)[mu], np.eye(d)[nu]
                polys.append(np.array([base, base + emu, base + emu + enu, base + enu]))

        pos = np.array(coords, dtype=float)
        name = "x".join(map(str, dims)) + (" periodic" if any(per) else "") + " grid"
        return cls(
            len(coords), np.array(edges), faces, pos=pos, face_polys=polys, name=name,
            meta={"dims": dims, "periodic": per, "edge_index": eidx,
                  "face_anchor": anchors, "coord": coords},
        )

    @classmethod
    def from_vertex_cycles(cls, n_vertices, edges, face_cycles=(), pos=None, name="board",
                           vertex_names=None) -> "Board":
        """Build a simple board from an edge list and faces given as vertex cycles
        [v0, v1, ..., v0]."""
        tmp = cls(n_vertices, np.asarray(edges), [], pos=pos)
        faces = [tmp.walk(list(c)) for c in face_cycles]
        polys = None
        if pos is not None:
            polys = [np.asarray(pos, dtype=float)[list(c[:-1])] for c in face_cycles]
        meta = {"vertex_names": list(vertex_names)} if vertex_names is not None else {}
        return cls(n_vertices, np.asarray(edges), faces, pos=pos, face_polys=polys,
                   name=name, meta=meta)

    @classmethod
    def triangle(cls) -> "Board":
        """Three vertices, edges 0->1, 1->2, 2->0, one face 0->1->2->0."""
        pos = [(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)]
        return cls.from_vertex_cycles(3, [(0, 1), (1, 2), (2, 0)], [[0, 1, 2, 0]],
                                      pos=pos, name="triangle")

    @classmethod
    def cycle(cls, n: int, with_face: bool = True) -> "Board":
        """A ring of n vertices; optionally the whole ring is one face."""
        ang = 2 * np.pi * np.arange(n) / n + np.pi / 2
        pos = np.c_[np.cos(ang), np.sin(ang)]
        edges = [(i, (i + 1) % n) for i in range(n)]
        faces = [list(range(n)) + [0]] if with_face else []
        return cls.from_vertex_cycles(n, edges, faces, pos=pos, name=f"{n}-cycle")

    @classmethod
    def complete(cls, n: int, names=None, triangle_faces: bool = True) -> "Board":
        """Complete graph K_n (edges i->j for i<j), faces = all triangles i->j->k->i."""
        ang = 2 * np.pi * np.arange(n) / n + np.pi / 2
        pos = np.c_[np.cos(ang), np.sin(ang)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        faces = ([[i, j, k, i] for i, j, k in itertools.combinations(range(n), 3)]
                 if triangle_faces else [])
        return cls.from_vertex_cycles(n, edges, faces, pos=pos, name=f"K{n}",
                                      vertex_names=names)
