"""Checks run before trusting the production numbers."""

import time

import numpy as np

from qcd import (Gauge, Dirac, check_gammas, pion_correlator, rho_correlator, nucleon_correlator,
                 random_su3_near_identity, reunitarize, dag)

rng = np.random.default_rng(0)
print("1. gamma matrices and charge conjugation:", check_gammas())

# 2. Free quarks (all labels = identity): masses known exactly.
dims = (4, 4, 4, 16)
g = Gauge(dims, 5.7, rng)
kappa = 0.11
m_q = np.log(1 + 1 / (2 * kappa) - 4)             # free Wilson quark mass at zero momentum
S, _ = Dirac(g.U, kappa).propagator()
for name, C, k in [("pion", pion_correlator(S), 2), ("nucleon", nucleon_correlator(S), 3)]:
    meff = np.log(np.abs(C[2:6] / C[3:7]))
    print(f"2. free {name}: effective mass {np.round(meff, 3)}   expected ~{k} m_q = {k * m_q:.3f}")

# 3. Reframing invariance on a random (hot) labeling: correlators must not change.
g.U = reunitarize(np.exp(1j * rng.normal(size=g.U.shape)) * rng.normal(size=g.U.shape))
S1, _ = Dirac(g.U, 0.12).propagator()
c1 = [f(S1) for f in (pion_correlator, rho_correlator, nucleon_correlator)]
h = reunitarize(rng.normal(size=dims + (3, 3)) + 1j * rng.normal(size=dims + (3, 3)))
h[0, 0, 0, 0] = np.eye(3)                         # keep the source vertex's frame fixed
g.reframe(h)
S2, _ = Dirac(g.U, 0.12).propagator()
c2 = [f(S2) for f in (pion_correlator, rho_correlator, nucleon_correlator)]
print("3. invariant under reframing (pion, rho, nucleon):", [bool(np.allclose(a, b, rtol=1e-6)) for a, b in zip(c1, c2)])

# 4. Plaquette at beta = 5.7 (literature: about 0.549 on large lattices).
g = Gauge((6, 6, 6, 6), 5.7, rng)
t0 = time.time()
plaq = []
for i in range(120):
    acc = g.sweep()
    if i >= 60:
        plaq.append(g.plaquette())
print(f"4. plaquette at beta=5.7: {np.mean(plaq):.4f} +- {np.std(plaq) / np.sqrt(len(plaq) / 5):.4f}  "
      f"(acceptance {acc:.2f}, {(time.time() - t0) / 120:.2f} s/sweep on 6^4)")
