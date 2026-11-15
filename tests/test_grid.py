import numpy as np
import pytest
from scipy.special import i0, i1

from lgt import *
from lgt.grid import GridModel


@pytest.mark.parametrize("G", [Z2(), U1(), SU2()], ids=str)
@pytest.mark.parametrize("dims", [(4, 4), (2, 2), (4, 2, 2), (2, 2, 2, 2)])
def test_faces_match_board(G, dims):
    rng = np.random.default_rng(0)
    M = GridModel(dims, G, 1.0).hot_start(rng)
    B, U = M.to_labeling()
    assert np.isclose(M.mean_face(), np.mean(G.chi(face_products(G, B, U))) / G.dim)
    M2 = GridModel(dims, G, 1.0).from_labeling(U, B)
    assert np.allclose(M2.U, M.U)


@pytest.mark.parametrize("G", [Z2(), U1(), SU2()], ids=str)
def test_rectangle_matches_loop_product(G):
    rng = np.random.default_rng(1)
    M = GridModel((6, 4, 4), G, 1.0).hot_start(rng)
    B, U = M.to_labeling()
    R, T, mu, nu = 3, 2, 0, 2
    x = np.array([1, 2, 3])
    def vid(c):
        c = np.mod(c, M.dims)
        return int(np.ravel_multi_index(tuple(c), M.dims, order="F"))
    e_mu, e_nu = np.eye(3, dtype=int)[mu], np.eye(3, dtype=int)[nu]
    path = [x + k * e_mu for k in range(R + 1)]
    path += [x + R * e_mu + k * e_nu for k in range(1, T + 1)]
    path += [x + T * e_nu + (R - k) * e_mu for k in range(1, R + 1)]
    path += [x + (T - k) * e_nu for k in range(1, T + 1)]
    eidx = B.meta["edge_index"]
    steps = []
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        mu_ = int(np.flatnonzero(d)[0])
        if d[mu_] > 0:
            steps.append((eidx[(vid(a), mu_)], 1))
        else:
            steps.append((eidx[(vid(b), mu_)], -1))
    w = loop_product(G, B, U, steps)
    assert np.isclose(M.rectangle(R, T, mu, nu)[tuple(x)], G.chi(w) / G.dim)


def test_z2_heatbath_exact_2x2_torus():
    rng = np.random.default_rng(2)
    beta = 0.6
    ex = Model(Board.grid(2, 2, periodic=True), Z2(), beta).exact()
    exact = ex.average(lambda U: Z2().chi(face_products(Z2(), Board.grid(2, 2, periodic=True), U)).mean(axis=1))
    M = GridModel((2, 2), Z2(), beta).hot_start(rng)
    M.sweep(rng, 200)
    vals = []
    for _ in range(6000):
        M.sweep(rng)
        vals.append(M.mean_face())
    assert abs(np.mean(vals) - exact) < 0.02


def test_2d_large_torus_matches_independent_faces():
    rng = np.random.default_rng(3)
    beta = 0.8
    M = GridModel((16, 16), Z2(), beta).hot_start(rng)
    M.sweep(rng, 100)
    z2 = np.mean([(M.sweep(rng), M.mean_face())[1] for _ in range(300)])
    assert abs(z2 - np.tanh(beta)) < 0.01
    M = GridModel((16, 16), U1(), 2.0).hot_start(rng)
    M.sweep(rng, 100)
    u1 = np.mean([(M.sweep(rng), M.mean_face())[1] for _ in range(300)])
    assert abs(u1 - i1(2.0) / i0(2.0)) < 0.01


def test_su2_heatbath_matches_metropolis_2d():
    # 2D, planar-like large torus: faces nearly independent, <tr U_f / 2> = I_2(2b)/I_1(2b)
    from scipy.special import iv
    rng = np.random.default_rng(5)
    beta = 1.5
    M = GridModel((16, 16), SU2(), beta).hot_start(rng)
    M.sweep(rng, 100)
    v = np.mean([(M.sweep(rng), M.mean_face())[1] for _ in range(300)])
    assert abs(v - iv(2, 2 * beta) / iv(1, 2 * beta)) < 0.01
