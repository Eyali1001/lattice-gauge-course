"""Fast, vectorized samplers for models on periodic grids (chapter 4 onward).

The generic ``Model.metropolis`` loops over edges in Python and is fine for boards
with a few dozen edges. Experiments on big grids (say 8x8x8x8, 16384 edges) need
array code. ``GridModel`` stores a labeling of a periodic d-dimensional grid as an
array ``U`` of shape ``(d, L0, ..., L_{d-1}, *elem)``: ``U[mu][x]`` is the label of the
edge leaving vertex x in direction mu. Labels are stored as

* ``Z2``: real numbers +1 / -1,
* ``U1``: complex numbers e^{i theta} on the unit circle,
* ``SU2``: unit quaternions (w, x, y, z), trailing axis of length 4,

so products, inverses and (normalised) real traces are cheap array operations.
Only the standard face cost phi(x) = dim - Re tr(x) is supported.

Key idea (glossary 4.8, *face remainder*): the face product of a face through edge e,
walked starting along e, is U(e) r_f. The cost of e with everything else fixed is
const - Re tr(U(e) A), with A = sum of the remainders of the 2(d-1) faces through e.
Edges that share no face are updated at the same time ("checkerboard").
"""

from __future__ import annotations

import itertools

import numpy as np

from .board import Board
from .groups import Group, Z2, U1, SU2, qmul, qconj


class GridModel:
    """Model on a periodic grid with the standard face cost, stored as arrays."""

    def __init__(self, dims, G: Group, beta: float):
        self.dims = tuple(int(L) for L in dims)
        self.d = len(self.dims)
        if any(L % 2 for L in self.dims):
            raise ValueError("all side lengths must be even (needed for checkerboard updates)")
        if not isinstance(G, (Z2, U1, SU2)):
            raise ValueError("GridModel supports Z2, U1 and SU2")
        self.G = G
        self.beta = float(beta)
        self.kind = "su2" if isinstance(G, SU2) else "u1" if isinstance(G, U1) else "z2"
        self.complex = self.kind == "u1"  # kept for backwards compatibility
        coords = np.indices(self.dims)
        self._masks = []  # _masks[mu][p]: direction-mu edges whose other coords sum to p mod 2
        for mu in range(self.d):
            s = sum(coords[nu] for nu in range(self.d) if nu != mu) % 2
            self._masks.append([s == 0, s == 1])
        self.U = self.identity()

    # --- the algebra of stored labels ---------------------------------------------
    def _mul(self, a, b):
        return qmul(a, b) if self.kind == "su2" else a * b

    def _conj(self, x):
        if self.kind == "su2":
            return qconj(x)
        return np.conj(x) if self.kind == "u1" else x

    def _retr(self, x):
        """Re tr(x) / dim: 1 at the identity. Linear in x (also for sums of elements)."""
        if self.kind == "su2":
            return x[..., 0]
        return np.real(x)

    def _elem_mask(self, mask):
        return mask[..., None] if self.kind == "su2" else mask

    # --- states ---------------------------------------------------------------------
    def identity(self):
        if self.kind == "su2":
            U = np.zeros((self.d,) + self.dims + (4,))
            U[..., 0] = 1.0
            return U
        return np.ones((self.d,) + self.dims, dtype=complex if self.kind == "u1" else float)

    def cold_start(self):
        """All labels = identity (every face calm)."""
        self.U = self.identity()
        return self

    def hot_start(self, rng):
        """Every label independent and uniform."""
        shape = (self.d,) + self.dims
        if self.kind == "su2":
            self.U = self.G.uniform(rng, shape)
        elif self.kind == "u1":
            self.U = np.exp(1j * rng.uniform(0, 2 * np.pi, shape))
        else:
            self.U = rng.choice([-1.0, 1.0], size=shape)
        return self

    @property
    def n_edges(self) -> int:
        return self.d * int(np.prod(self.dims))

    @property
    def n_faces(self) -> int:
        return self.d * (self.d - 1) // 2 * int(np.prod(self.dims))

    @staticmethod
    def _shift(X, nu, k=1):
        """Y[x] = X[x + k e_nu] (periodic)."""
        return np.roll(X, -k, axis=nu)

    def remainder_sum(self, mu):
        """A[x] = sum of face remainders of the 2(d-1) faces through edge (x, mu)."""
        U, sh, cj, m = self.U, self._shift, self._conj, self._mul
        A = np.zeros_like(U[mu])
        for nu in range(self.d):
            if nu == mu:
                continue
            # face x -> x+mu -> x+mu+nu -> x+nu -> x ; remainder = U_nu(x+mu) U_mu(x+nu)^-1 U_nu(x)^-1
            A += m(m(sh(U[nu], mu), cj(sh(U[mu], nu))), cj(U[nu]))
            # face x -> x+mu -> x+mu-nu -> x-nu -> x
            A += m(m(cj(sh(sh(U[nu], mu), nu, -1)), cj(sh(U[mu], nu, -1))), sh(U[nu], nu, -1))
        return A

    # --- updates ----------------------------------------------------------------------
    def sweep(self, rng, n: int = 1, hits: int = 4, step: float = 1.0):
        """n full sweeps; every edge is updated once per sweep.

        Z2 and SU2: heat bath (draw the label from its exact conditional distribution).
        U1: Metropolis with ``hits`` proposals theta -> theta + uniform(-step*pi, step*pi).
        Returns the acceptance rate (U1) or None.
        """
        acc, tot = 0, 0
        for _ in range(n):
            for mu in range(self.d):
                for p in (0, 1):
                    mask = self._masks[mu][p]
                    A = self.remainder_sum(mu)
                    if self.kind == "u1":
                        a, t = self._metropolis_u1(mu, mask, A, rng, hits, step)
                        acc, tot = acc + a, tot + t
                    elif self.kind == "z2":
                        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * self.beta * A))
                        new = np.where(rng.random(self.dims) < prob_plus, 1.0, -1.0)
                        self.U[mu] = np.where(mask, new, self.U[mu])
                    else:
                        new = self._heatbath_su2(A, rng)
                        self.U[mu] = np.where(mask[..., None], new, self.U[mu])
        return acc / tot if tot else None

    def _metropolis_u1(self, mu, mask, A, rng, hits, step):
        acc, tot = 0, 0
        z = self.U[mu]
        for _ in range(hits):
            prop = z * np.exp(1j * rng.uniform(-np.pi, np.pi, self.dims) * step)
            dS = -np.real(prop * A) + np.real(z * A)  # change of total cost
            ok = mask & (rng.random(self.dims) < np.exp(np.minimum(0.0, -self.beta * dS)))
            z = np.where(ok, prop, z)
            acc += int(ok.sum())
            tot += int(mask.sum())
        self.U[mu] = z
        return acc, tot

    def _heatbath_su2(self, A, rng):
        """Draw U with density proportional to exp(beta * Re tr(U A)) = exp(2 beta (U A)_0).

        Write A = k V with k = |A| and V a unit quaternion. Then X = U V is distributed
        as exp(2 beta k X_0) times the uniform distribution on the 3-sphere, whose X_0
        marginal is proportional to sqrt(1 - X_0^2). Sample X_0 by Creutz's rejection
        method, the rest of X uniformly on a 2-sphere, and return U = X V^{-1}.
        """
        k = np.linalg.norm(A, axis=-1)
        V = A / np.where(k > 0, k, 1.0)[..., None]
        a = 2.0 * self.beta * np.maximum(k, 1e-12)
        x0 = np.empty(self.dims)
        todo = np.ones(self.dims, dtype=bool)
        while todo.any():
            y = rng.uniform(np.exp(-2.0 * a[todo]), 1.0)
            cand = 1.0 + np.log(y) / a[todo]
            ok = rng.random(cand.shape) < np.sqrt(np.clip(1.0 - cand ** 2, 0, None))
            idx = np.flatnonzero(todo)
            x0.flat[idx[ok]] = cand[ok]
            todo.flat[idx[ok]] = False
        r = np.sqrt(np.clip(1.0 - x0 ** 2, 0, None))
        n = rng.normal(size=self.dims + (3,))
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        X = np.concatenate([x0[..., None], r[..., None] * n], axis=-1)
        return qmul(X, qconj(V))

    # --- invariants ---------------------------------------------------------------------
    def face_products(self):
        """Array (n_pairs, *dims, *elem) of face products for direction pairs mu < nu."""
        U, sh, cj, m = self.U, self._shift, self._conj, self._mul
        out = [m(m(m(U[a], sh(U[b], a)), cj(sh(U[a], b))), cj(U[b]))
               for a, b in itertools.combinations(range(self.d), 2)]
        return np.array(out)

    def mean_face(self) -> float:
        """Average over faces of Re tr(face product)/dim: 1 = all calm, 0 = random."""
        return float(np.mean(self._retr(self.face_products())))

    def line(self, mu, length):
        """P[x] = product of `length` labels walking from x in direction mu."""
        P = np.ones_like(self.U[mu]) if self.kind != "su2" else self.identity()[0]
        for k in range(length):
            P = self._mul(P, self._shift(self.U[mu], mu, k))
        return P

    def rectangle(self, R, T, mu=0, nu=1):
        """Re tr(loop product)/dim of every R x T rectangle in the (mu, nu) plane."""
        cj, m = self._conj, self._mul
        a = self.line(mu, R)
        b = self._shift(self.line(nu, T), mu, R)
        c = self._shift(self.line(mu, R), nu, T)
        d_ = self.line(nu, T)
        return self._retr(m(m(m(a, b), cj(c)), cj(d_)))

    def mean_rectangle(self, R, T) -> float:
        """Average of Re tr(loop product)/dim over all R x T rectangles in all planes and
        both orientations (R along mu, T along nu and vice versa)."""
        vals = [self.rectangle(R, T, mu, nu).mean()
                for mu, nu in itertools.permutations(range(self.d), 2)]
        return float(np.mean(vals))

    # --- conversion to the generic Board representation --------------------------------
    def board(self) -> Board:
        return Board.grid(*self.dims, periodic=True)

    def to_labeling(self, board: Board | None = None):
        """Labels in the edge order of Board.grid(*dims, periodic=True).

        Z2 -> int8 +-1, U1 -> angles in [0, 2pi), SU2 -> quaternions. Returns (board, U).
        """
        board = board or self.board()
        if self.kind == "su2":
            out = np.empty((board.n_edges, 4))
        else:
            out = np.empty(board.n_edges, dtype=np.int8 if self.kind == "z2" else float)
        for (v, mu), e in board.meta["edge_index"].items():
            x = np.unravel_index(v, self.dims, order="F")
            val = self.U[(mu,) + tuple(x)]
            if self.kind == "u1":
                out[e] = np.mod(np.angle(val), 2 * np.pi)
            elif self.kind == "z2":
                out[e] = int(np.real(val))
            else:
                out[e] = val
        return board, out

    def from_labeling(self, U_edges, board: Board | None = None):
        board = board or self.board()
        for (v, mu), e in board.meta["edge_index"].items():
            x = np.unravel_index(v, self.dims, order="F")
            if self.kind == "u1":
                self.U[(mu,) + tuple(x)] = np.exp(1j * U_edges[e])
            elif self.kind == "z2":
                self.U[(mu,) + tuple(x)] = float(U_edges[e])
            else:
                self.U[(mu,) + tuple(x)] = U_edges[e]
        return self
