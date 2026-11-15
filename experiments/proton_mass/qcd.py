"""A minimal quenched lattice QCD: SU(3) labels on a 4D torus, Wilson quarks, hadron correlators.

Conventions
-----------
* Lattice arrays have axes (x, y, z, t) = directions mu = 0, 1, 2, 3.
* Gauge field U has shape (4, Lx, Ly, Lz, Lt, 3, 3): U[mu][x] is the label of the edge
  x -> x + mu (a 3x3 special unitary matrix).
* Quark fields have shape (Lx, Ly, Lz, Lt, 4, 3): (spin, color) at every vertex.
* Gauge cost: the standard face cost with the physicists' normalization,
      S = beta * sum_faces (1 - Re tr U_f / 3),   beta = 6 / g^2.
  (In the course's normalization the stiffness is beta / 3.)
* Wilson-Dirac operator (hopping parameter kappa, antiperiodic in time):
      D psi(x) = psi(x) - kappa * sum_mu [ (1 - g_mu) U_mu(x) psi(x+mu)
                                         + (1 + g_mu) U_mu(x-mu)^dagger psi(x-mu) ].
  Its inverse S = D^{-1} is the quark propagator: the amplitude for a quark to go
  from one vertex to another through the random gauge labels.
"""

from __future__ import annotations

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# Dirac matrices (Euclidean, Dirac basis)
# ---------------------------------------------------------------------------
_s1 = np.array([[0, 1], [1, 0]], dtype=complex)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_s3 = np.array([[1, 0], [0, -1]], dtype=complex)
_I2, _Z2 = np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex)


def _blk(a, b, c, d):
    return np.block([[a, b], [c, d]])


GAMMA = np.array([_blk(_Z2, -1j * s, 1j * s, _Z2) for s in (_s1, _s2, _s3)]
                 + [_blk(_I2, _Z2, _Z2, -_I2)])                    # gamma_1..3, gamma_4
G5 = GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]
CHARGE_C = GAMMA[1] @ GAMMA[3]                                     # C = gamma_2 gamma_4
CG5 = CHARGE_C @ G5
P_PLUS = (np.eye(4) + GAMMA[3]) / 2                                # positive-parity projector
EPS = [(a, b, c, 1 if (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1)
       for a, b, c in itertools.permutations(range(3))]


def check_gammas():
    for m in range(4):
        assert np.allclose(GAMMA[m], GAMMA[m].conj().T)
        for n in range(4):
            assert np.allclose(GAMMA[m] @ GAMMA[n] + GAMMA[n] @ GAMMA[m], 2 * (m == n) * np.eye(4))
        assert np.allclose(CHARGE_C @ GAMMA[m] @ np.linalg.inv(CHARGE_C), -GAMMA[m].T)
    assert np.allclose(G5 @ G5, np.eye(4))
    return True


# ---------------------------------------------------------------------------
# SU(3) helpers
# ---------------------------------------------------------------------------
def dag(U):
    return np.conj(np.swapaxes(U, -1, -2))


def mm(A, B):
    return A @ B


def reunitarize(U):
    """Project matrices back onto SU(3) (Gram-Schmidt on rows, third row = cross product)."""
    a = U[..., 0, :]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = U[..., 1, :]
    b = b - np.sum(np.conj(a) * b, axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c = np.conj(np.cross(a, b))
    return np.stack([a, b, c], axis=-2)


def random_su3_near_identity(rng, n, eps):
    """n random SU(3) matrices exp(i eps H), H random Hermitian traceless; closed under inverse."""
    H = rng.normal(size=(n, 3, 3)) + 1j * rng.normal(size=(n, 3, 3))
    H = (H + dag(H)) / 2
    H = H - np.trace(H, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    w, V = np.linalg.eigh(H)
    R = V @ (np.exp(1j * eps * w)[..., None] * dag(V))
    return np.concatenate([R, dag(R)])                               # include the inverses


# ---------------------------------------------------------------------------
# Gauge field
# ---------------------------------------------------------------------------
class Gauge:
    def __init__(self, dims, beta, rng):
        self.dims, self.beta, self.rng = tuple(dims), float(beta), rng
        self.U = np.broadcast_to(np.eye(3, dtype=complex), (4,) + self.dims + (3, 3)).copy()
        coords = np.indices(self.dims)
        self.masks = [[(sum(coords[n] for n in range(4) if n != mu) % 2) == p for p in (0, 1)]
                      for mu in range(4)]
        self.pool = random_su3_near_identity(rng, 200, 0.25)

    @staticmethod
    def sh(X, mu, k=1):
        return np.roll(X, -k, axis=mu)

    def staples(self, mu):
        U, sh = self.U, self.sh
        A = np.zeros(self.dims + (3, 3), dtype=complex)
        for nu in range(4):
            if nu == mu:
                continue
            A += sh(U[nu], mu) @ dag(sh(U[mu], nu)) @ dag(U[nu])
            Unu_m = sh(U[nu], nu, -1)
            A += dag(sh(Unu_m, mu)) @ dag(sh(U[mu], nu, -1)) @ Unu_m
        return A

    def sweep(self, hits=8):
        """Checkerboard Metropolis with `hits` proposals per edge. Returns acceptance rate."""
        acc = tot = 0
        b3 = self.beta / 3
        for mu in range(4):
            for p in (0, 1):
                mask = self.masks[mu][p]
                A = self.staples(mu)
                Ucur = self.U[mu]
                old = np.real(np.trace(Ucur @ A, axis1=-2, axis2=-1))
                for _ in range(hits):
                    R = self.pool[self.rng.integers(0, len(self.pool), self.dims)]
                    prop = R @ Ucur
                    new = np.real(np.trace(prop @ A, axis1=-2, axis2=-1))
                    dS = -b3 * (new - old)
                    ok = mask & (self.rng.random(self.dims) < np.exp(np.minimum(0, -dS)))
                    Ucur = np.where(ok[..., None, None], prop, Ucur)
                    old = np.where(ok, new, old)
                    acc += int(ok.sum()); tot += int(mask.sum())
                self.U[mu] = Ucur
        self.U = reunitarize(self.U)
        return acc / tot

    def plaquette(self):
        U, sh, vals = self.U, self.sh, []
        for mu, nu in itertools.combinations(range(4), 2):
            P = U[mu] @ sh(U[nu], mu) @ dag(sh(U[mu], nu)) @ dag(U[nu])
            vals.append(np.real(np.trace(P, axis1=-2, axis2=-1)).mean() / 3)
        return float(np.mean(vals))

    def reframe(self, h):
        """Apply a random reframing h (SU(3) at every vertex): U_mu(x) -> h(x) U_mu(x) h(x+mu)^dag."""
        for mu in range(4):
            self.U[mu] = h @ self.U[mu] @ dag(self.sh(h, mu))


# ---------------------------------------------------------------------------
# Wilson-Dirac operator and solver
# ---------------------------------------------------------------------------
class Dirac:
    def __init__(self, U, kappa):
        self.U, self.kappa = U, kappa
        self.dims = U.shape[1:5]
        self.Pm = [np.eye(4) - GAMMA[m] for m in range(4)]
        self.Pp = [np.eye(4) + GAMMA[m] for m in range(4)]
        self.Ud = [dag(np.roll(U[m], 1, axis=m)) for m in range(4)]      # U_mu(x-mu)^dagger

    def _shift(self, psi, mu, k):
        out = np.roll(psi, -k, axis=mu)
        if mu == 3:                                     # antiperiodic boundary in time
            if k == 1:
                out[:, :, :, -1] *= -1
            else:
                out[:, :, :, 0] *= -1
        return out

    def hop(self, psi):
        out = np.zeros_like(psi)
        for m in range(4):
            fwd = np.einsum("...ab,...sb->...sa", self.U[m], self._shift(psi, m, 1))
            out += np.einsum("st,...ta->...sa", self.Pm[m], fwd)
            bwd = np.einsum("...ab,...sb->...sa", self.Ud[m], self._shift(psi, m, -1))
            out += np.einsum("st,...ta->...sa", self.Pp[m], bwd)
        return out

    def apply(self, psi):
        return psi - self.kappa * self.hop(psi)

    def apply_dag(self, psi):
        # gamma_5-hermiticity: D^dagger = g5 D g5
        g = lambda v: np.einsum("st,...ta->...sa", G5, v)
        return g(self.apply(g(psi)))

    def solve(self, b, tol=1e-8, maxiter=5000):
        """Solve D x = b by conjugate gradients on the normal equations D^dag D x = D^dag b."""
        rhs = self.apply_dag(b)
        x = np.zeros_like(b)
        r = rhs.copy(); p = r.copy()
        rr = np.vdot(r, r).real
        stop = tol ** 2 * np.vdot(rhs, rhs).real
        for it in range(maxiter):
            Ap = self.apply_dag(self.apply(p))
            alpha = rr / np.vdot(p, Ap).real
            x += alpha * p
            r -= alpha * Ap
            rr_new = np.vdot(r, r).real
            if rr_new < stop:
                return x, it + 1
            p = r + (rr_new / rr) * p
            rr = rr_new
        raise RuntimeError("CG did not converge")

    def solve_bicgstab(self, b, tol=1e-8, maxiter=5000):
        """Solve D x = b directly with BiCGStab (usually far fewer operator applications)."""
        x = np.zeros_like(b)
        r = b.copy(); rhat = r.copy()
        rho = alpha = omega = 1.0
        v = np.zeros_like(b); p = np.zeros_like(b)
        bnorm = np.sqrt(np.vdot(b, b).real)
        for it in range(maxiter):
            rho_new = np.vdot(rhat, r)
            beta_ = (rho_new / rho) * (alpha / omega)
            p = r + beta_ * (p - omega * v)
            v = self.apply(p)
            alpha = rho_new / np.vdot(rhat, v)
            s = r - alpha * v
            t = self.apply(s)
            omega = np.vdot(t, s) / np.vdot(t, t)
            x += alpha * p + omega * s
            r = s - omega * t
            rho = rho_new
            if np.sqrt(np.vdot(r, r).real) < tol * bnorm:
                return x, it + 1
        raise RuntimeError("BiCGStab did not converge")

    def propagator(self, src=(0, 0, 0, 0), tol=1e-8):
        """S(x; src) as an array (Lx, Ly, Lz, Lt, 4, 3, 4, 3): 12 solves, one per source spin-color."""
        S = np.zeros(self.dims + (4, 3, 4, 3), dtype=complex)
        iters = []
        for s0 in range(4):
            for c0 in range(3):
                b = np.zeros(self.dims + (4, 3), dtype=complex)
                b[src + (s0, c0)] = 1.0
                x, it = self.solve_bicgstab(b, tol)
                S[..., s0, c0] = x
                iters.append(it)
        return S, iters


# ---------------------------------------------------------------------------
# Hadron correlators (zero momentum: summed over space at each time)
# ---------------------------------------------------------------------------
def pion_correlator(S):
    """C(t) = sum_x |S(x;0)|^2   (pseudoscalar, gamma_5-hermiticity)."""
    return np.sum(np.abs(S) ** 2, axis=(0, 1, 2, 4, 5, 6, 7))


def rho_correlator(S):
    """C(t) = -sum_x sum_i tr[g_i S g_i g5 S^dag g5]  (vector meson), averaged over i."""
    Sm = S.reshape(S.shape[:4] + (12, 12))
    tot = 0
    for i in range(3):
        Gi = np.kron(GAMMA[i], np.eye(3))
        G5k = np.kron(G5, np.eye(3))
        M = Gi @ Sm @ Gi @ G5k @ dag(Sm) @ G5k
        tot = tot - np.trace(M, axis1=-2, axis2=-1)
    return np.real(np.sum(tot, axis=(0, 1, 2))) / 3


def nucleon_correlator(S):
    """Proton-like correlator, interpolator N = eps_abc (u_a^T C g5 d_b) u_c, two light quarks degenerate.

    C_{gg'} = sum eps eps' (Cg5)_{ab}(Cg5)_{a'b'} S^d_{bb'} [ S_{aa'} S_{cc'} - S_{ac'} S_{ca'} ]
    (spin indices shown; color via eps), projected with (1 + g4)/2 and summed over space.
    """
    def blk(a, ap):
        return S[..., :, a, :, ap]                     # (..., 4, 4) spin matrix for colors a, a'
    C = np.zeros(S.shape[:4] + (4, 4), dtype=complex)
    for (a, b, c, s1) in EPS:
        for (ap, bp, cp, s2) in EPS:
            Sd = blk(b, bp)
            Saa, Scc, Sac, Sca = blk(a, ap), blk(c, cp), blk(a, cp), blk(c, ap)
            direct = np.sum(CG5 * (Saa @ CG5 @ np.swapaxes(Sd, -1, -2)), axis=(-2, -1))
            exch = Sca @ CG5 @ np.swapaxes(Sd, -1, -2) @ CG5.T @ Sac
            C += s1 * s2 * (direct[..., None, None] * Scc - exch)
    return np.real(np.trace(P_PLUS @ C, axis1=-2, axis2=-1)).sum(axis=(0, 1, 2))
