"""Label groups.

Every group stores its elements as numpy arrays with a fixed trailing
``elem_shape`` (``()`` for scalars, ``(3, 3)`` for 3x3 matrices). All operations
broadcast over leading axes, so ``U`` of shape ``(n_edges, *elem_shape)`` is a
whole labeling.

Glossary: 2.1 (label group), 3.6 (class function), 4.1 (uniform distribution).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

TAU = 2 * np.pi


def _tup(shape) -> tuple:
    return (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)


class Group:
    """Interface shared by all label groups."""

    name: str = "G"
    elem_shape: tuple = ()
    dtype = float
    is_abelian: bool = True
    is_compact: bool = True
    is_finite: bool = False
    dim: int = 1  # size of the matrices whose trace defines ``chi``

    # --- group structure -------------------------------------------------
    def identity(self, shape=()) -> np.ndarray:
        raise NotImplementedError

    def mul(self, a, b) -> np.ndarray:
        raise NotImplementedError

    def inv(self, a) -> np.ndarray:
        raise NotImplementedError

    def prod(self, xs, axis=0) -> np.ndarray:
        """Ordered product x_0 x_1 ... x_{k-1} along ``axis``."""
        xs = np.moveaxis(np.asarray(xs), axis, 0)
        out = xs[0]
        for x in xs[1:]:
            out = self.mul(out, x)
        return out

    def conj(self, h, x) -> np.ndarray:
        """h x h^{-1}."""
        return self.mul(self.mul(h, x), self.inv(h))

    def is_identity(self, x, atol=1e-9) -> np.ndarray:
        e = self.identity()
        d = np.abs(np.asarray(x) - e)
        if self.elem_shape:
            d = d.reshape(d.shape[: d.ndim - len(self.elem_shape)] + (-1,)).max(axis=-1)
        return d <= atol

    def equal(self, a, b, atol=1e-9) -> np.ndarray:
        return self.is_identity(self.mul(a, self.inv(b)), atol=atol)

    # --- randomness ------------------------------------------------------
    def uniform(self, rng: np.random.Generator, shape=()) -> np.ndarray:
        """Draw from the uniform (Haar) distribution. Only for compact groups."""
        raise ValueError(
            f"{self.name} is not compact: it has no uniform probability distribution."
        )

    # --- invariants ------------------------------------------------------
    def chi(self, x) -> np.ndarray:
        """Real part of the trace in the defining matrix representation.

        A class function: chi(h x h^-1) = chi(x). chi(identity) = dim.
        """
        raise NotImplementedError

    def angle(self, x) -> np.ndarray:
        """An angle in [0, 2pi) for drawing elements as clock hands (if meaningful)."""
        raise NotImplementedError

    # --- finite groups ---------------------------------------------------
    def elements(self) -> np.ndarray:
        raise ValueError(f"{self.name} is not finite.")

    @property
    def order(self) -> int:
        return len(self.elements())

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Abelian scalar groups
# ---------------------------------------------------------------------------


class Z2(Group):
    """{+1, -1} under multiplication: the coin model."""

    name = "Z2"
    dtype = np.int8
    is_finite = True

    def identity(self, shape=()):
        return np.ones(shape, dtype=np.int8)

    def mul(self, a, b):
        return (np.asarray(a) * np.asarray(b)).astype(np.int8)

    def inv(self, a):
        return np.asarray(a, dtype=np.int8)

    def prod(self, xs, axis=0):
        return np.prod(np.asarray(xs), axis=axis).astype(np.int8)

    def uniform(self, rng, shape=()):
        return rng.choice(np.array([1, -1], dtype=np.int8), size=shape)

    def chi(self, x):
        return np.asarray(x, dtype=float)

    def angle(self, x):
        return np.where(np.asarray(x) == 1, 0.0, np.pi)

    def elements(self):
        return np.array([1, -1], dtype=np.int8)


@dataclass(repr=False)
class ZN(Group):
    """Integers mod N under addition; element k is drawn as the angle 2 pi k / N."""

    N: int = 3
    dtype = np.int64
    is_finite = True

    def __post_init__(self):
        self.name = f"Z{self.N}"

    def identity(self, shape=()):
        return np.zeros(shape, dtype=np.int64)

    def mul(self, a, b):
        return (np.asarray(a) + np.asarray(b)) % self.N

    def inv(self, a):
        return (-np.asarray(a)) % self.N

    def prod(self, xs, axis=0):
        return np.sum(np.asarray(xs), axis=axis) % self.N

    def uniform(self, rng, shape=()):
        return rng.integers(0, self.N, size=shape)

    def chi(self, x):
        return np.cos(TAU * np.asarray(x) / self.N)

    def angle(self, x):
        return TAU * np.asarray(x) / self.N

    def elements(self):
        return np.arange(self.N)


class U1(Group):
    """The circle: angles in [0, 2pi) under addition mod 2pi."""

    name = "U1"

    def identity(self, shape=()):
        return np.zeros(shape)

    def mul(self, a, b):
        return np.mod(np.asarray(a) + np.asarray(b), TAU)

    def inv(self, a):
        return np.mod(-np.asarray(a), TAU)

    def prod(self, xs, axis=0):
        return np.mod(np.sum(np.asarray(xs), axis=axis), TAU)

    def is_identity(self, x, atol=1e-9):
        x = np.mod(np.asarray(x), TAU)
        return np.minimum(x, TAU - x) <= atol

    def uniform(self, rng, shape=()):
        return rng.uniform(0, TAU, size=shape)

    def chi(self, x):
        return np.cos(np.asarray(x))

    def angle(self, x):
        return np.mod(np.asarray(x), TAU)


class PositiveReals(Group):
    """(R_{>0}, x): exchange rates. Not compact, so no uniform distribution."""

    name = "R>0"
    is_compact = False

    def identity(self, shape=()):
        return np.ones(shape)

    def mul(self, a, b):
        return np.asarray(a, dtype=float) * np.asarray(b, dtype=float)

    def inv(self, a):
        return 1.0 / np.asarray(a, dtype=float)

    def prod(self, xs, axis=0):
        return np.prod(np.asarray(xs, dtype=float), axis=axis)

    def is_identity(self, x, atol=1e-9):
        return np.abs(np.log(np.asarray(x, dtype=float))) <= atol

    def chi(self, x):
        return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Non-abelian groups
# ---------------------------------------------------------------------------


class S3(Group):
    """Permutations of {0,1,2}. Element i is ``S3.PERMS[i]``; i=0 is the identity.

    chi = number of fixed points = trace of the 3x3 permutation matrix.
    """

    name = "S3"
    dtype = np.int64
    is_abelian = False
    is_finite = True
    dim = 3
    PERMS = [p for p in itertools.permutations(range(3))]  # (0,1,2) first

    def __init__(self):
        idx = {p: i for i, p in enumerate(self.PERMS)}
        n = len(self.PERMS)
        self._mul = np.zeros((n, n), dtype=np.int64)
        for i, p in enumerate(self.PERMS):
            for j, q in enumerate(self.PERMS):
                # (p q)(k) = p(q(k)) : apply q first, then p
                self._mul[i, j] = idx[tuple(p[q[k]] for k in range(3))]
        self._inv = np.array(
            [idx[tuple(np.argsort(p))] for p in self.PERMS], dtype=np.int64
        )
        self._chi = np.array(
            [sum(p[k] == k for k in range(3)) for p in self.PERMS], dtype=float
        )

    def identity(self, shape=()):
        return np.zeros(shape, dtype=np.int64)

    def mul(self, a, b):
        return self._mul[np.asarray(a), np.asarray(b)]

    def inv(self, a):
        return self._inv[np.asarray(a)]

    def is_identity(self, x, atol=0):
        return np.asarray(x) == 0

    def uniform(self, rng, shape=()):
        return rng.integers(0, 6, size=shape)

    def chi(self, x):
        return self._chi[np.asarray(x)]

    def elements(self):
        return np.arange(6)

    def perm(self, x) -> tuple:
        return self.PERMS[int(x)]


class SO3(Group):
    """Rotations of 3D space as 3x3 orthogonal matrices with determinant 1."""

    name = "SO3"
    elem_shape = (3, 3)
    is_abelian = False
    dim = 3

    def identity(self, shape=()):
        return np.broadcast_to(np.eye(3), _tup(shape) + (3, 3)).copy()

    def mul(self, a, b):
        return np.asarray(a) @ np.asarray(b)

    def inv(self, a):
        return np.swapaxes(np.asarray(a), -1, -2)

    def uniform(self, rng, shape=()):
        shape = _tup(shape)
        # Uniform rotation from a uniform unit quaternion.
        q = rng.normal(size=shape + (4,))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return quat_to_matrix(q)

    def chi(self, x):
        return np.trace(np.asarray(x), axis1=-2, axis2=-1)

    def rotation_angle(self, x):
        """Angle of rotation in [0, pi]: chi = 1 + 2 cos(angle)."""
        return np.arccos(np.clip((self.chi(x) - 1) / 2, -1, 1))

    @staticmethod
    def about_axis(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
        return quat_to_matrix(q)


def quat_to_matrix(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


# ---------------------------------------------------------------------------
# Face costs (glossary 4.2)
# ---------------------------------------------------------------------------


def standard_face_cost(G: Group):
    """The standard face cost phi(x) = dim - chi(x): zero at the identity, positive elsewhere.

    Z2: phi(+1)=0, phi(-1)=2.  U1/ZN: 1 - cos(angle).  S3/SO3: 3 - trace.
    """

    def phi(x):
        return G.dim - G.chi(x)

    phi.__name__ = f"standard_face_cost[{G.name}]"
    return phi


# Physicists call this the Wilson action density; kept as an alias.
wilson_cost = standard_face_cost


# ---------------------------------------------------------------------------
# SU(2) as unit quaternions
# ---------------------------------------------------------------------------


def qmul(a, b):
    """Quaternion product of arrays (..., 4): (w, x, y, z)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(a):
    a = np.asarray(a, dtype=float)
    return a * np.array([1.0, -1.0, -1.0, -1.0])


class SU2(Group):
    """2x2 unitary matrices with determinant 1, stored as unit quaternions (w, x, y, z).

    The quaternion w + x i + y j + z k is the matrix w I - i (x s1 + y s2 + z s3)
    (s1, s2, s3 the Pauli matrices), so the trace is 2w. Geometrically SU(2) is the 3-sphere |q| = 1 in R^4; each element
    and its negative give the same 3D rotation (SU(2) is a "double cover" of SO(3)).
    """

    name = "SU2"
    elem_shape = (4,)
    is_abelian = False
    dim = 2

    def identity(self, shape=()):
        out = np.zeros(_tup(shape) + (4,))
        out[..., 0] = 1.0
        return out

    def mul(self, a, b):
        return qmul(a, b)

    def inv(self, a):
        return qconj(a)

    def uniform(self, rng, shape=()):
        q = rng.normal(size=_tup(shape) + (4,))
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    def chi(self, x):
        return 2.0 * np.asarray(x, dtype=float)[..., 0]

    def angle(self, x):
        """Angle in [0, 2pi] with chi = 2 cos(angle / 2)."""
        return 2.0 * np.arccos(np.clip(np.asarray(x, dtype=float)[..., 0], -1, 1))

    def exp(self, v):
        """exp of a 'small rotation' v in R^3 (Lie algebra): (cos|v|, sin|v| v/|v|)."""
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        safe = np.where(n > 0, n, 1.0)
        return np.concatenate([np.cos(n), np.sin(n) * v / safe], axis=-1)

    def to_matrix(self, q):
        q = np.asarray(q, dtype=float)
        w, x, y, z = np.moveaxis(q, -1, 0)
        return np.stack([np.stack([w - 1j * z, -y - 1j * x], -1),
                         np.stack([y - 1j * x, w + 1j * z], -1)], -2)
