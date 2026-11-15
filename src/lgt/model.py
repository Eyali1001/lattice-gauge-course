"""Models: probability distributions on labelings (glossary section 4).

    mu(U) = exp(-beta * S(U)) / Z   relative to "every edge label uniform and independent",
    S(U)  = sum over faces f of phi(face product of f).

For finite groups every average can be computed *exactly* by enumerating all
labelings (``Model.exact``). For bigger boards or continuous groups use
``Model.metropolis`` (introduced properly in chapter 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .board import Board
from .groups import Group, U1, SO3, SU2, standard_face_cost
from .labeling import all_labelings, face_products, random_labeling, _axis


@dataclass
class Exact:
    """All labelings of a finite-group model with their probabilities."""

    U: np.ndarray  # (n_labelings, n_edges)
    p: np.ndarray  # (n_labelings,) probabilities, sum to 1
    cost: np.ndarray  # (n_labelings,) total cost S
    log_Z: float  # log of the normalizer (uniform reference: Z = mean of exp(-beta S))

    def average(self, f: Callable) -> float:
        """<f>: f takes a batch of labelings (n, E) and returns n numbers."""
        return float(np.sum(self.p * np.asarray(f(self.U), dtype=float)))

    def sample(self, rng, n: int):
        idx = rng.choice(len(self.p), size=n, p=self.p)
        return self.U[idx]


@dataclass
class Model:
    """A model = (board, label group, face cost phi, stiffness beta) (glossary 4.5)."""

    board: Board
    G: Group
    beta: float = 1.0
    phi: Callable | None = None
    _exact: Exact | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.phi is None:
            self.phi = standard_face_cost(self.G)
        if not self.G.is_compact:
            raise ValueError(
                f"{self.G} is not compact, so there is no uniform distribution to build "
                "a model on (see chapter 2)."
            )

    def with_beta(self, beta: float) -> "Model":
        return Model(self.board, self.G, beta, self.phi)

    # --- costs ------------------------------------------------------------------
    def face_costs(self, U):
        """phi(face product) for every face, shape (*batch, n_faces)."""
        return self.phi(face_products(self.G, self.board, U))

    def cost(self, U):
        """Total cost S(U) (glossary 4.3)."""
        return np.sum(self.face_costs(U), axis=-1)

    def log_weight(self, U):
        return -self.beta * self.cost(U)

    # --- exact enumeration (finite groups) -------------------------------------
    def exact(self, max_count: int = 2**22) -> Exact:
        if self._exact is None:
            U = all_labelings(self.G, self.board, max_count=max_count)
            S = self.cost(U)
            lw = -self.beta * S
            m = lw.max()
            w = np.exp(lw - m)
            Zrel = w.sum()
            self._exact = Exact(U=U, p=w / Zrel, cost=S,
                                log_Z=float(m + np.log(Zrel) - np.log(len(U))))
        return self._exact

    def average(self, f: Callable) -> float:
        return self.exact().average(f)

    # --- Metropolis sampling ------------------------------------------------------
    def metropolis(self, rng, n_samples: int, sweeps_between: int = 1, burn_in: int = 100,
                   U0=None, step: float = 1.0):
        """Single-edge Metropolis sampler. Returns (n_samples, n_edges, *elem) array.

        Proposal: a fresh uniform label (finite groups), or the current label
        multiplied by a random element near the identity of size ~``step``
        (U1, SO3).
        """
        G, board = self.G, self.board
        U = random_labeling(G, board, rng) if U0 is None else np.array(U0)
        f_of_e = board.faces_of_edge()
        faces = board.faces
        out = []

        def local_cost(U, fis):
            return sum(float(self.phi(_face_prod(G, U, faces[fi]))) for fi in fis)

        total = burn_in + n_samples * sweeps_between
        for sweep in range(total):
            for e in rng.permutation(board.n_edges):
                old = U[e].copy() if np.ndim(U[e]) else U[e]
                fis = f_of_e[e]
                c_old = local_cost(U, fis)
                U[e] = _propose(G, old, rng, step)
                dS = local_cost(U, fis) - c_old
                # accept with probability min(1, exp(-beta dS)); correct for either sign of beta
                if rng.random() >= np.exp(min(0.0, -self.beta * dS)):
                    U[e] = old
            if sweep >= burn_in and (sweep - burn_in) % sweeps_between == sweeps_between - 1:
                out.append(U.copy())
        return np.array(out)


def _face_prod(G, U, face):
    x = None
    for e, s in face:
        lab = U[e] if s > 0 else G.inv(U[e])
        x = lab if x is None else G.mul(x, lab)
    return x


def _propose(G, old, rng, step):
    if G.is_finite:
        return G.uniform(rng)
    if isinstance(G, U1):
        return G.mul(old, rng.uniform(-np.pi, np.pi) * min(step, 1.0))
    if isinstance(G, SU2):
        v = rng.normal(size=3)
        v *= rng.uniform(-np.pi, np.pi) * min(step, 1.0) / 2 / np.linalg.norm(v)
        return G.mul(G.exp(v), old)
    if isinstance(G, SO3):
        axis = rng.normal(size=3)
        return G.mul(G.about_axis(axis, rng.uniform(-np.pi, np.pi) * min(step, 1.0)), old)
    return G.uniform(rng)
