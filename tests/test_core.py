import numpy as np
import pytest

from lgt import *

RNG = np.random.default_rng(0)
GROUPS = [Z2(), ZN(3), U1(), S3(), SO3()]


def boards():
    return [Board.triangle(), Board.grid(3, 3), Board.grid(3, 3, periodic=True),
            Board.grid(2, 2, periodic=True), Board.complete(4), Board.grid(2, 2, 2)]


def test_counts():
    B = Board.grid(3, 3)
    assert (B.n_vertices, B.n_edges, B.n_faces) == (9, 12, 4)
    T = Board.grid(3, 3, periodic=True)
    assert (T.n_vertices, T.n_edges, T.n_faces) == (9, 18, 9)
    T2 = Board.grid(2, 2, periodic=True)
    assert (T2.n_vertices, T2.n_edges, T2.n_faces) == (4, 8, 4)
    C = Board.grid(2, 2, 2)
    assert (C.n_vertices, C.n_edges, C.n_faces) == (8, 12, 6)
    assert len(Board.grid(4, 3).fundamental_cycles()) == Board.grid(4, 3).cycle_rank


@pytest.mark.parametrize("G", GROUPS, ids=str)
@pytest.mark.parametrize("B", boards(), ids=repr)
def test_reframing_conjugates_loops(G, B):
    U = random_labeling(G, B, RNG)
    h = random_reframing(G, B, RNG)
    U2 = reframe(G, B, U, h)
    fp, fp2 = face_products(G, B, U), face_products(G, B, U2)
    for f, face in enumerate(B.faces):
        v = B.walk_vertices(face)[0]
        hv = h[v]
        assert G.equal(fp2[f], G.conj(hv, fp[f]))
    assert np.allclose(G.chi(fp), G.chi(fp2))


@pytest.mark.parametrize("G", GROUPS, ids=str)
@pytest.mark.parametrize("B", boards(), ids=repr)
def test_tree_normal_form(G, B):
    U = random_labeling(G, B, RNG)
    Un, h, tree = tree_normal_form(G, B, U)
    assert np.all(G.is_identity(Un[tree]))
    assert np.all(G.equal(Un, reframe(G, B, U, h)))
    for e, cyc in B.fundamental_cycles().items():
        assert G.equal(Un[e], loop_product(G, B, U, cyc))


def test_batch_matches_single():
    G, B = U1(), Board.grid(3, 3, periodic=True)
    Us = random_labeling(G, B, RNG, batch=5)
    fp = face_products(G, B, Us)
    for i in range(5):
        assert np.allclose(fp[i], face_products(G, B, Us[i]))
    Un, _, _ = tree_normal_form(G, B, Us)
    for i in range(5):
        assert np.allclose(Un[i], tree_normal_form(G, B, Us[i])[0])


def test_consistent():
    G, B = SO3(), Board.grid(3, 3)
    h = random_reframing(G, B, RNG)
    U = reframe(G, B, identity_labeling(G, B), h)
    assert is_consistent(G, B, U)
    assert not is_consistent(G, B, random_labeling(G, B, RNG))


@pytest.mark.parametrize("G", [Z2(), ZN(3)], ids=str)
def test_abelian_class_count(G):
    B = Board.grid(2, 3)
    Us = all_labelings(G, B)
    ids = class_ids(G, B, Us)
    assert ids.max() + 1 == G.order ** B.cycle_rank


def test_nonabelian_classes_are_orbits():
    G, B = S3(), Board.triangle()
    Us = all_labelings(G, B)
    ids = class_ids(G, B, Us)
    # brute force orbits
    hs = all_reframings(G, B)
    code = lambda U: tuple(int(x) for x in U)
    orbit_of = {}
    for i, U in enumerate(Us):
        orbit_of[code(U)] = frozenset(code(reframe(G, B, U, h)) for h in hs)
    for i, U in enumerate(Us):
        for j, V in enumerate(Us[:60]):
            same = code(V) in orbit_of[code(U)]
            assert same == (ids[i] == ids[j])
    # triangle with S3: classes <-> conjugacy classes of the loop product: 3
    assert ids.max() + 1 == 3


def test_model_exact_and_invariance():
    G, B = Z2(), Board.grid(2, 2, periodic=True)
    M = Model(B, G, beta=0.7)
    ex = M.exact()
    assert np.isclose(ex.p.sum(), 1)
    # Planar board: faces independent, <face> = tanh(beta)
    P = Model(Board.grid(3, 3), G, beta=0.7)
    fp = lambda U: face_products(G, P.board, U)[:, 0]
    assert np.isclose(P.average(fp), np.tanh(0.7))
    # a single edge label averages to zero
    assert np.isclose(M.average(lambda U: U[:, 0]), 0)


def test_metropolis_matches_exact():
    G, B = Z2(), Board.grid(3, 2)
    M = Model(B, G, beta=0.8)
    s = M.metropolis(np.random.default_rng(1), 4000, burn_in=200)
    est = face_products(G, B, s).astype(float).mean()
    assert abs(est - np.tanh(0.8)) < 0.03


def test_noncompact_model_refused():
    with pytest.raises(ValueError):
        Model(Board.triangle(), PositiveReals())


def test_metropolis_negative_beta():
    # beta < 0 favors defects; on a planar grid faces are independent with <U_f> = tanh(beta) < 0
    G, B = Z2(), Board.grid(3, 2)
    M = Model(B, G, beta=-0.8)
    s = M.metropolis(np.random.default_rng(2), 4000, burn_in=200)
    est = face_products(G, B, s).astype(float).mean()
    assert abs(est - np.tanh(-0.8)) < 0.03
