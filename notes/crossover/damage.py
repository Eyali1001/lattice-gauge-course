"""Damage spreading for 4D SU(2): how far could a coupling certificate reach?

See README.md section 4 for the results and the three lessons (synchronous coupling is useless;
a clamped single label tests pure gauge redundancy; per-edge maximal coupling is the right baseline).


Two copies of an equilibrated 4D SU(2) labeling differ at one edge. Both are updated by
heat-bath sweeps that use *identical* random numbers (a synchronous coupling: the class
angle of each new label is drawn by inverse CDF from one shared uniform, its direction from
one shared normal vector). We track the gauge-invariant damage

    D(t) = sum over faces of |tr X_f - tr Y_f| / 2 ,

which upper-bounds, edge by edge, the Wasserstein distance between the two conditional laws
that a Dobrushin-type certificate would have to control. If D(t) dies out, this coupling
contracts at that stiffness (for typical, not worst-case, configurations). If it survives,
this coupling is not a certificate there.

Usage:  uv run python notes/crossover/damage.py
"""
import time
import numpy as np
from lgt import SU2
from lgt.grid import GridModel
from lgt.groups import qmul, qconj

X0 = np.linspace(-1, 1, 513)


def sync_heatbath_sweep(M, rng, after=None):
    """One heat-bath sweep of M using a fixed amount of randomness per edge (inverse CDF).

    `after` (optional) is called after every sub-step, e.g. to re-impose clamped labels; clamps
    restored only at the end of a sweep would be overwritten mid-sweep (identically in both copies)."""
    for mu in range(M.d):
        for p in (0, 1):
            mask = M._masks[mu][p]
            A = M.remainder_sum(mu)
            u = rng.random(M.dims)
            nvec = rng.normal(size=M.dims + (3,))
            k = np.linalg.norm(A, axis=-1)
            V = A / np.where(k > 0, k, 1.0)[..., None]
            a = 2.0 * M.beta * k                                  # density sqrt(1-x^2) exp(a x)
            logw = a[..., None] * X0 + 0.5 * np.log(np.clip(1 - X0 ** 2, 1e-300, None))
            w = np.exp(logw - logw.max(axis=-1, keepdims=True))
            cdf = np.concatenate([np.zeros(w.shape[:-1] + (1,)),
                                  np.cumsum((w[..., 1:] + w[..., :-1]) / 2, axis=-1)], axis=-1)
            cdf /= cdf[..., -1:]                                  # trapezoid CDF on the grid
            idx = (cdf < u[..., None]).sum(axis=-1).clip(1, len(X0) - 1)
            c0 = np.take_along_axis(cdf, (idx - 1)[..., None], -1)[..., 0]
            c1 = np.take_along_axis(cdf, idx[..., None], -1)[..., 0]
            frac = np.clip((u - c0) / np.maximum(c1 - c0, 1e-300), 0, 1)
            x0 = X0[idx - 1] + frac * (X0[1] - X0[0])
            r = np.sqrt(np.clip(1 - x0 ** 2, 0, None))
            n = nvec / np.linalg.norm(nvec, axis=-1, keepdims=True)
            Xq = np.concatenate([x0[..., None], r[..., None] * n], axis=-1)
            new = qmul(Xq, qconj(V))
            M.U[mu] = np.where(mask[..., None], new, M.U[mu])
            if after is not None:
                after()


def face_traces(M):
    return 2 * M.face_products()[..., 0]                         # tr = 2 w for unit quaternions


def damage_run(beta, L=6, therm=150, T=60, seed=0):
    """Free damage at one edge. NOTE: trivially healed in the first sweep, because the damaged
    edge is resampled from identical surroundings before any neighbor sees it. Kept as a
    cautionary example; use clamped_profile instead."""
    rng = np.random.default_rng(seed)
    X = GridModel((L,) * 4, SU2(), beta).hot_start(rng)
    X.sweep(rng, therm)
    Y = GridModel((L,) * 4, SU2(), beta)
    Y.U = X.U.copy()
    Y.U[0][(0,) * 4] = SU2().uniform(rng)
    rx, ry = np.random.default_rng(seed + 1000), np.random.default_rng(seed + 1000)
    out = []
    for _ in range(T + 1):
        out.append(np.abs(face_traces(X) - face_traces(Y)).sum() / 2)
        sync_heatbath_sweep(X, rx); sync_heatbath_sweep(Y, ry)
    return np.array(out)


def clamped_profile(beta, L=8, therm=100, burn=20, T=60, seed=0, rmax=8):
    """Steady-state damage profile from a permanently clamped disagreement.

    Edge e0 = (origin, direction 0) is clamped to g in X and to -g in Y (maximally different)
    and never resampled. Both copies run synchronous heat-bath sweeps. Returns the mean of
    |tr X_f - tr Y_f|/2 over faces whose base vertex is at periodic L1 distance r from the
    origin, r = 0..rmax, averaged over T sweeps after a burn-in.
    """
    rng = np.random.default_rng(seed)
    X = GridModel((L,) * 4, SU2(), beta).hot_start(rng)
    X.sweep(rng, therm)
    g = X.U[0][(0,) * 4].copy()
    Y = GridModel((L,) * 4, SU2(), beta)
    Y.U = X.U.copy()
    Y.U[0][(0,) * 4] = -g
    coords = np.indices((L,) * 4)
    dist = sum(np.minimum(c, L - c) for c in coords)                # periodic L1 distance
    rx, ry = np.random.default_rng(seed + 1000), np.random.default_rng(seed + 1000)
    acc = np.zeros(rmax + 1)
    for t in range(burn + T):
        cx = lambda: X.U[0].__setitem__((0,) * 4, g)                  # clamp (boundary condition)
        cy = lambda: Y.U[0].__setitem__((0,) * 4, -g)
        sync_heatbath_sweep(X, rx, cx); sync_heatbath_sweep(Y, ry, cy)
        if t >= burn:
            dmg = np.abs(face_traces(X) - face_traces(Y)).mean(axis=0) / 2   # per base vertex
            acc += [dmg[dist == r].mean() for r in range(rmax + 1)]
    return acc / T



def clamped_face_profile(beta, L=8, therm=100, burn=20, T=60, seed=0, rmax=8):
    """Like clamped_profile, but the clamped disagreement is gauge-invariant.

    The four labels of the face f0 = (origin, plane 01) are clamped: in X all to the identity
    (face product I), in Y the same except U_1(origin) = -I (face product -I, a maximal defect).
    Unlike a single clamped label, this changes the law of the invariants: its true influence
    on a face at distance r decays like the connected face-face correlation, ~exp(-m r).
    """
    rng = np.random.default_rng(seed)
    X = GridModel((L,) * 4, SU2(), beta).hot_start(rng)
    X.sweep(rng, therm)
    o = (0,) * 4
    e1 = tuple(int(i == 0) for i in range(4))     # origin + e_0
    e2 = tuple(int(i == 1) for i in range(4))     # origin + e_1
    clamp = [(0, o), (1, e1), (0, e2), (1, o)]    # the face's four edges (direction, base vertex)
    I = np.array([1.0, 0, 0, 0])
    Y = GridModel((L,) * 4, SU2(), beta)
    Y.U = X.U.copy()
    coords = np.indices((L,) * 4)
    dist = sum(np.minimum(c, L - c) for c in coords)
    rx, ry = np.random.default_rng(seed + 1000), np.random.default_rng(seed + 1000)
    acc = np.zeros(rmax + 1); sgn = np.zeros(rmax + 1)

    def clamp_x():
        for mu, v in clamp:
            X.U[mu][v] = I

    def clamp_y():
        for mu, v in clamp:
            Y.U[mu][v] = I
        Y.U[1][o] = -I

    clamp_x(); clamp_y()
    for t in range(burn + T):
        sync_heatbath_sweep(X, rx, clamp_x); sync_heatbath_sweep(Y, ry, clamp_y)
        if t >= burn:
            diff = (face_traces(Y) - face_traces(X)) / 2
            dmg, sd = np.abs(diff).mean(axis=0), diff.mean(axis=0)
            acc += [dmg[dist == r].mean() for r in range(rmax + 1)]
            sgn += [sd[dist == r].mean() for r in range(rmax + 1)]
    return acc / T, sgn / T


def run_face_experiment(betas=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.2), runs=2):
    """Prints, per distance r: the coupling's damage E|tr Y_f - tr X_f|/2 (what a coupling
    certificate can prove) and the signed mean E(tr Y_f - tr X_f)/2 (an unbiased estimate of
    the true influence). Their ratio measures how tight the coupling is."""
    t0 = time.time()
    print("Clamped FACE disagreement; per L1 distance r: coupling damage | true influence (signed)")
    for beta in betas:
        res = [clamped_face_profile(beta, seed=s, burn=40) for s in range(runs)]
        dmg = np.mean([r[0] for r in res], axis=0); sgn = np.mean([r[1] for r in res], axis=0)
        print(f"beta_ours {beta:4.2f} (beta_phys {2 * beta:4.2f})   [{time.time() - t0:.0f}s]", flush=True)
        for r in range(len(dmg)):
            print(f"    r={r}: damage {dmg[r]:9.2e}   true {sgn[r]:+9.2e}", flush=True)


# ----------------------------------------------------------------------------------------------
# Quotient coupling: re-gauge Y toward X after every sweep.
#
# The heat bath commutes in law with reframings, so the sequence of *classes* is itself a Markov
# chain, and replacing Y by any reframing h.Y (h chosen from X and Y) keeps the class process of
# Y exactly right. Choosing h to bring Y's labels close to X's makes the coupling ignore
# gauge-redundant disagreements. h(v) maximizes sum over edges at v of Re tr(X(e)^-1 (h.Y)(e)),
# which for unit quaternions is <h, K> with K built from neighbors: h = K / |K|.
# ----------------------------------------------------------------------------------------------

def relative_gauge_fix(X, Y, fixed_vertices=(), passes=8):
    """Reframe Y (in place) toward X by greedy vertex updates on a checkerboard."""
    d, L = X.d, X.dims
    h = np.zeros(L + (4,)); h[..., 0] = 1.0
    parity = sum(np.indices(L)) % 2
    frozen = np.zeros(L, dtype=bool)
    for v in fixed_vertices:
        frozen[v] = True
    sh = GridModel._shift
    for _ in range(passes):
        for p in (0, 1):
            K = np.zeros(L + (4,))
            for mu in range(d):
                Xo, Yo = X.U[mu], Y.U[mu]                       # out-edge v -> v+mu
                K += qconj(qmul(qmul(Yo, qconj(sh(h, mu))), qconj(Xo)))
                Xi, Yi = sh(X.U[mu], mu, -1), sh(Y.U[mu], mu, -1)  # in-edge v-mu -> v
                K += qmul(qmul(qconj(Xi), sh(h, mu, -1)), Yi)
            n = np.linalg.norm(K, axis=-1, keepdims=True)
            new = K / np.where(n > 0, n, 1.0)
            upd = (parity == p) & ~frozen
            h = np.where(upd[..., None], new, h)
    for mu in range(d):
        Y.U[mu] = qmul(qmul(h, Y.U[mu]), qconj(sh(h, mu)))


def clamped_face_profile_quotient(beta, L=8, therm=100, burn=40, T=60, seed=0, rmax=8):
    """clamped_face_profile, but with Y re-gauged toward X after every sweep."""
    rng = np.random.default_rng(seed)
    X = GridModel((L,) * 4, SU2(), beta).hot_start(rng)
    X.sweep(rng, therm)
    o = (0,) * 4
    e1 = tuple(int(i == 0) for i in range(4)); e2 = tuple(int(i == 1) for i in range(4))
    e12 = tuple(int(i in (0, 1)) for i in range(4))
    clamp = [(0, o), (1, e1), (0, e2), (1, o)]
    I = np.array([1.0, 0, 0, 0])
    Y = GridModel((L,) * 4, SU2(), beta); Y.U = X.U.copy()
    coords = np.indices((L,) * 4)
    dist = sum(np.minimum(c, L - c) for c in coords)
    rx, ry = np.random.default_rng(seed + 1000), np.random.default_rng(seed + 1000)
    acc = np.zeros(rmax + 1); sgn = np.zeros(rmax + 1)

    def clamp_x():
        for mu, v in clamp:
            X.U[mu][v] = I

    def clamp_y():
        for mu, v in clamp:
            Y.U[mu][v] = I
        Y.U[1][o] = -I

    clamp_x(); clamp_y()
    for t in range(burn + T):
        sync_heatbath_sweep(X, rx, clamp_x); sync_heatbath_sweep(Y, ry, clamp_y)
        relative_gauge_fix(X, Y, fixed_vertices=[o, e1, e2, e12])
        if t >= burn:
            diff = (face_traces(Y) - face_traces(X)) / 2
            dmg, sd = np.abs(diff).mean(axis=0), diff.mean(axis=0)
            acc += [dmg[dist == r].mean() for r in range(rmax + 1)]
            sgn += [sd[dist == r].mean() for r in range(rmax + 1)]
    return acc / T, sgn / T


# ----------------------------------------------------------------------------------------------
# Maximal (Dobrushin) coupling per edge. Lesson from the synchronous experiments: drawing both
# copies' labels from one shared random number *in the frame of their staples* is not a good
# coupling. Two nearly identical conditional laws with slightly rotated staples give rotated
# outputs, so damage saturates even at tiny beta. The maximal coupling makes the two labels
# equal with probability 1 - d(p, q), exactly as in chapter 7's Fact 7.3.
# ----------------------------------------------------------------------------------------------
from scipy.special import ive as _ive


def _logdens(x, A, beta):
    """log of the conditional density (w.r.t. the uniform distribution) of label x given staple sum A."""
    a = 2.0 * beta * np.linalg.norm(A, axis=-1)
    a = np.maximum(a, 1e-12)
    logZ = np.log(2.0) + np.log(_ive(1, a)) + a - np.log(a)   # int exp(a x0) dHaar = 2 I_1(a)/a
    return 2.0 * beta * qmul(x, A)[..., 0] - logZ                # density exp(beta Re tr(x A)) = exp(2 beta (xA)_0)


def maximal_coupled_sweep(X, Y, rng, after_x=None, after_y=None, max_tries=200):
    beta = X.beta
    for mu in range(X.d):
        for p in (0, 1):
            mask = X._masks[mu][p]
            AX, AY = X.remainder_sum(mu), Y.remainder_sum(mu)
            x = X._heatbath_su2(AX, rng)
            same = np.log(rng.random(X.dims)) + _logdens(x, AX, beta) <= _logdens(x, AY, beta)
            y = x.copy()
            todo = ~same
            for _ in range(max_tries):                      # residual part of q, by rejection
                if not todo.any():
                    break
                cand = Y._heatbath_su2(AY, rng)
                ok = todo & (np.log(rng.random(X.dims)) + _logdens(cand, AY, beta) > _logdens(cand, AX, beta))
                y = np.where(ok[..., None], cand, y)
                todo &= ~ok
            X.U[mu] = np.where(mask[..., None], x, X.U[mu])
            Y.U[mu] = np.where(mask[..., None], y, Y.U[mu])
            if after_x is not None:
                after_x()
            if after_y is not None:
                after_y()


def face_clamp_maximal(beta, regauge, L=8, therm=100, burn=30, T=50, seed=0, rmax=8):
    """Clamped-face experiment with per-edge maximal coupling, optionally re-gauging Y toward X."""
    rng = np.random.default_rng(seed)
    X = GridModel((L,) * 4, SU2(), beta).hot_start(rng)
    X.sweep(rng, therm)
    o = (0,) * 4
    e1 = tuple(int(i == 0) for i in range(4)); e2 = tuple(int(i == 1) for i in range(4))
    e12 = tuple(int(i in (0, 1)) for i in range(4))
    clamp = [(0, o), (1, e1), (0, e2), (1, o)]
    I = np.array([1.0, 0, 0, 0])
    Y = GridModel((L,) * 4, SU2(), beta); Y.U = X.U.copy()
    dist = sum(np.minimum(c, L - c) for c in np.indices((L,) * 4))

    def clamp_x():
        for mu, v in clamp:
            X.U[mu][v] = I

    def clamp_y():
        for mu, v in clamp:
            Y.U[mu][v] = I
        Y.U[1][o] = -I

    clamp_x(); clamp_y()
    acc = np.zeros(rmax + 1); sgn = np.zeros(rmax + 1)
    for t in range(burn + T):
        maximal_coupled_sweep(X, Y, rng, clamp_x, clamp_y)
        if regauge:
            relative_gauge_fix(X, Y, fixed_vertices=[o, e1, e2, e12])
        if t >= burn:
            diff = (face_traces(Y) - face_traces(X)) / 2
            dmg, sd = np.abs(diff).mean(axis=0), diff.mean(axis=0)
            acc += [dmg[dist == r].mean() for r in range(rmax + 1)]
            sgn += [sd[dist == r].mean() for r in range(rmax + 1)]
    return acc / T, sgn / T


def run_maximal_experiment(betas=(0.01, 0.1, 0.3, 0.6, 1.0, 1.2)):
    t0 = time.time()
    print("Clamped FACE, per-edge maximal coupling. Columns: r, damage without re-gauging, "
          "damage with re-gauging (quotient coupling), true influence (signed)")
    for beta in betas:
        d0, s0 = face_clamp_maximal(beta, regauge=False)
        d1, s1 = face_clamp_maximal(beta, regauge=True)
        print(f"beta_ours {beta:4.2f} (beta_phys {2 * beta:4.2f})   [{time.time() - t0:.0f}s]", flush=True)
        for r in range(len(d0)):
            print(f"    r={r}: plain {d0[r]:9.2e}   quotient {d1[r]:9.2e}   true {(s0[r] + s1[r]) / 2:+9.2e}", flush=True)


if __name__ == "__main__":
    # The synchronous-coupling experiments above (clamped_profile, clamped_face_profile) are kept
    # as cautionary examples: that coupling saturates at every beta. The informative run is:
    run_maximal_experiment()
