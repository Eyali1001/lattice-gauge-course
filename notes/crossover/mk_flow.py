"""Migdal-Kadanoff blocking for 4D lattice gauge models, in the course's terms.

One blocking step (scale factor b = 2) on a d-dimensional grid:

1. "move faces": slide the faces of b^(d-2) parallel planes onto one plane, so each surviving
   face carries the weight f^(b^(d-2)) (an approximation: the moved faces really sat elsewhere);
2. "merge faces": inside each plane, merge b x b faces into one. In 2D this step is exact
   (chapter 5: face elements of a slab are independent and their products random-walk on G):
   the normalized character coefficients multiply, c_r -> c_r^(b^2).

A face weight f is a class function on G, stored on a grid of angles. We track
  * the influence bound of chapter 7, alpha <= 6(d-1) tanh(spread(log f)/2)  (Fact 7.2 with
    beta*Phi replaced by the spread of -log f), and
  * an "effective stiffness": the Wilson beta with the same fundamental coefficient c_1.

Usage:  uv run python notes/crossover/mk_flow.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

TH = np.linspace(1e-6, np.pi - 1e-6, 4001)          # class angle


class SU2:
    name = "SU(2)"
    haar = (2 / np.pi) * np.sin(TH) ** 2               # density of the class angle
    js = np.arange(0, 40)                             # 2j+1 = d_j, store index k = 2j
    @staticmethod
    def chi(k):
        return np.sin((k + 1) * TH) / np.sin(TH)
    @staticmethod
    def dim(k):
        return k + 1
    @staticmethod
    def wilson(beta):                                 # our cost 2 - tr = 2 - 2 cos(theta)
        return np.exp(-beta * (2 - 2 * np.cos(TH)))
    @staticmethod
    def c1_wilson(beta):                              # fundamental coefficient I_2(2b)/I_1(2b)
        return ive(2, 2 * beta) / ive(1, 2 * beta)


class U1:
    name = "U(1)"
    haar = np.ones_like(TH) / np.pi                   # angle in [0, pi] (class functions are even)
    js = np.arange(0, 60)
    @staticmethod
    def chi(k):
        return np.cos(k * TH)                         # real part of e^{ik theta}; even functions
    @staticmethod
    def dim(k):
        return 1
    @staticmethod
    def mult(k):                                      # e^{ik} and e^{-ik} both carry c_k
        return 1 if k == 0 else 2
    @staticmethod
    def wilson(beta):                                 # cost 1 - cos(theta)
        return np.exp(-beta * (1 - np.cos(TH)))
    @staticmethod
    def c1_wilson(beta):
        return ive(1, beta) / ive(0, beta)


def integrate(G, g):
    return np.trapezoid(g * G.haar, TH)


def coeffs(G, f):
    """Normalized character coefficients c_k = <f, chi_k> / (d_k <f, 1>)."""
    z = integrate(G, f)
    return np.array([integrate(G, f * G.chi(k)) / (G.dim(k) * z) for k in G.js])


def from_coeffs(G, c):
    mult = getattr(G, "mult", G.dim)                  # SU(2): d_j; U(1): 1 or 2
    f = sum(mult(k) * c[i] * G.chi(k) for i, k in enumerate(G.js))
    return np.clip(f, 1e-300, None)


def mk_step(G, f, d=4, b=2):
    moved = f ** (b ** (d - 2))
    moved /= moved.max()
    c = coeffs(G, moved) ** (b * b)                   # exact 2D merge of b x b faces
    f_new = from_coeffs(G, c)
    return f_new / f_new.max()


def alpha(f, d=4):
    spread = np.log(f.max()) - np.log(f.min())
    return 6 * (d - 1) * np.tanh(spread / 2)


def beta_eff(G, f):
    c1 = coeffs(G, f)[1]                              # SU(2): k=1 is j=1/2; U(1): charge 1
    try:
        return brentq(lambda b: G.c1_wilson(b) - c1, 1e-8, 200)
    except ValueError:
        return np.nan


def flow(G, beta0, d=4, max_steps=60):
    f = G.wilson(beta0)
    out = [(0, beta_eff(G, f), alpha(f, d))]
    for n in range(1, max_steps + 1):
        f = mk_step(G, f, d)
        out.append((n, beta_eff(G, f), alpha(f, d)))
        if out[-1][2] < 1:
            break
    return out


if __name__ == "__main__":
    # sanity check: in 2D the merge is exact and there is no face moving, so c_1 -> c_1^4 per step
    f = SU2.wilson(1.0)
    c_before = coeffs(SU2, f)[1]
    c_after = coeffs(SU2, mk_step(SU2, f, d=2))[1]
    print(f"2D check: c1^4 = {c_before ** 4:.6f}, after one exact merge = {c_after:.6f}")

    for G, betas in [(SU2, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]), (U1, [0.5, 1.0, 2.0, 4.0, 8.0])]:
        print(f"\n{G.name}, d = 4: blocking steps until the chapter-7 criterion alpha < 1 holds")
        for b0 in betas:
            fl = flow(G, b0)
            n, be, a = fl[-1]
            path = "  ".join(f"{x[1]:.3f}" for x in fl[:8])
            print(f"  beta = {b0:4}: {n:3d} steps (alpha = {a:.2f})   beta_eff along the way: {path}")

    # running per step at large beta, compared with one-loop asymptotic freedom for SU(2)
    print("\nSU(2) drop of beta_eff in one step vs one-loop prediction (11/(6 pi^2)) ln 2 = "
          f"{11 / (6 * np.pi ** 2) * np.log(2):.3f}")
    for b0 in [2.0, 4.0, 8.0]:
        f = SU2.wilson(b0)
        print(f"  beta = {b0}: drop = {b0 - beta_eff(SU2, mk_step(SU2, f)):.3f}")
    print("U(1) drop of beta_eff in one step (true 4D U(1) at large beta: no running, massless photon)")
    for b0 in [2.0, 4.0, 8.0]:
        f = U1.wilson(b0)
        print(f"  beta = {b0}: drop = {b0 - beta_eff(U1, mk_step(U1, f)):.3f}")

    # Exact check of the curvature derivation (README section 2): start from an SU(2) heat kernel,
    # whose characters are c_j = exp(-s j(j+1)). With unit-sphere time t = s/4 the weight is
    # ~ exp(-B theta^2) with B = 1/(4t) - 1/6 = 1/s - 1/6. Prediction: B drops by exactly 1/8 per step.
    print("\nHeat-kernel start: drop of B = 1/s - 1/6 per step (prediction 0.125)")
    for B0 in [3.0, 6.0]:
        s = 1 / (B0 + 1 / 6)
        f = from_coeffs(SU2, np.exp(-s * (SU2.js / 2) * (SU2.js / 2 + 1)))
        Bs = [B0]
        for _ in range(4):
            f = mk_step(SU2, f / f.max())
            s = -np.log(coeffs(SU2, f)[1]) / 0.75           # j = 1/2: c = exp(-3s/4)
            Bs.append(1 / s - 1 / 6)
        print(f"  B0 = {B0}: drops {np.round(-np.diff(Bs), 4)}")
