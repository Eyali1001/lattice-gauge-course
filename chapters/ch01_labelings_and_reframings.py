# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (lgt course)
#     language: python
#     name: lgt
# ---

# %% [markdown]
# # Chapter 1 — Labelings, reframings and classes
#
# In Chapter 0 every edge of a currency market carried an exchange rate. When a
# country redenominated its currency, many rates changed at once, but no arbitrage
# opportunity appeared or disappeared. The rates were a *description*; the arbitrage
# was the *content*.
#
# This chapter takes that idea and strips it down to its skeleton, so that it works
# for **any group** of labels, not just exchange rates. The labels might be coin flips
# ($\pm1$), clock hands (angles), shuffles of three cards (permutations), or the
# relative orientations of cameras (3D rotations). In every case the same three
# ingredients appear:
#
# 1. labels on the edges of a graph,
# 2. products of labels around loops,
# 3. a freedom to change conventions at each vertex, which changes the labels but
#    not the loop products (up to a twist we'll meet for non-commuting groups).
#
# Nothing in this chapter is random. Probability arrives in Chapter 2.
#
# **What you'll learn**
# - the precise definitions of *board*, *labeling*, *walk product*, *reframing* and *class*;
# - which functions of a labeling survive every reframing, and why traces appear for non-commuting groups;
# - how to count classes, and how to pick one canonical representative of each (*tree normal form*).
#
# **Prerequisites:** Chapter 0 (helpful, not required); basic group theory
# (groups, conjugation, conjugacy classes).

# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import HTML

from lgt import *
from lgt import viz

viz.use_style()
rng = np.random.default_rng(1)

# %% [markdown]
# ---
# ## 1. Boards
#
# We need a place to put labels. A graph alone is almost enough, but we will also want
# to single out some short loops as "elementary" (the little squares of a grid,
# for instance). So we bundle a graph with a chosen list of closed walks.
#
# **Definition (board, glossary 1.1–1.3).** A **board** $X$ consists of
# - a finite connected graph with vertex set $V$ and edge set $E$;
# - a list $F$ of closed walks in that graph, called **faces**.
#
# Every edge $\lbrace s,t\rbrace$ can be walked in two **directions**, $s\to t$ and $t\to s$. We
# write $\vec E$ for the set of all directions ($|\vec E| = 2|E|$), and for a
# direction $d$ we write $\bar d$ for the same edge walked the other way.
#
# Which closed walks count as faces is a choice we make when we build the board, much
# like choosing which pixels count as "neighbours" in an image model. On a grid the
# natural choice is the unit squares.
#
# In code (and in the pictures) each edge is stored with one **reference direction**
# $s\to t$, drawn as a small arrow. This is pure bookkeeping: the other direction is
# always available too. A walk is stored as a list of *steps* `(e, sign)`, where
# `sign = +1` walks edge `e` along its arrow and `sign = -1` walks it against the arrow.

# %%
tri = Board.triangle()
grid = Board.grid(4, 3)
torus = Board.grid(3, 3, periodic=True)
for X in (tri, grid, torus):
    print(X)

# %% [markdown]
# Three boards we'll use throughout. The right-hand picture is a window cut through the
# middle of the wrap-around edges. Look at its borders: an edge that leaves through the
# right side comes back in through the left side, and one that leaves through the top
# comes back in at the bottom. The partial squares along the borders are pieces of the
# same wrapped faces. Gluing left to right and top to bottom turns the grid into the
# surface of a doughnut, so we call it a **torus**.

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
viz.draw_board(axs[0], tri, vertex_labels=["0", "1", "2"], title="triangle: 3 edges, 1 face")
viz.draw_board(axs[1], grid, title="4×3 grid: 17 edges, 6 faces")
viz.draw_board(axs[2], torus, title="3×3 periodic grid (torus): 18 edges, 9 faces")
plt.show()

# %% [markdown]
# A face is a list of steps. Notice the two `-1` signs in the grid's first face: going
# counter-clockwise around a unit square, the top and left edges are walked against
# their arrows.

# %%
print("triangle face (steps):  ", tri.faces[0], "  vertices:", tri.walk_vertices(tri.faces[0]))
print("first grid face (steps):", grid.faces[0], "  vertices:", grid.walk_vertices(grid.faces[0]))

# %% [markdown]
# > **Recap.** A board is a connected graph plus a list of designated closed walks
# > (faces). Each edge has two directions; code stores one reference direction per edge
# > and records walks as `(edge, ±1)` steps.
#
# **Check yourself**
#
# 1. How many vertices, edges and faces does an $L\times L$ grid (no wrap-around) have?
#
# <details><summary>Answer</summary>
#
# $L^2$ vertices; $2L(L-1)$ edges ($L-1$ horizontal edges in each of $L$ rows, same for columns); $(L-1)^2$ faces.
#
# </details>
#
# 2. And an $L\times L$ torus (with $L\ge 3$)?
#
# <details><summary>Answer</summary>
#
# $L^2$ vertices; $2L^2$ edges (every vertex has one edge going right and one going up); $L^2$ faces.
#
# </details>
#
# 3. Why do we store a face as a list of steps rather than as a set of vertices?
#
# <details><summary>Answer</summary>
#
# A face is a *closed walk*: it has an order, a direction, and a starting point, and all three will matter once labels are multiplied around it. Also, on a small torus two different edges can join the same pair of vertices, so a list of vertices would not even say which edges are used.
#
# </details>

# %% [markdown]
# ---
# ## 2. Labelings and their products
#
# ### 2.1 Label groups
#
# **Definition (label group, glossary 2.1).** A **label group** $G$ is any group.
# Its elements are the labels we will put on edges. Our running examples:
#
# | group | elements | multiplication | how we draw a label |
# |---|---|---|---|
# | $\mathbb Z_2$ ("coins") | $\lbrace +1,-1\rbrace$ | ordinary product | $-1$ edges are thick and gold |
# | $\mathbb Z_3$ | $\lbrace 0,1,2\rbrace$ | addition mod 3 | clock hand at $0^\circ, 120^\circ, 240^\circ$ |
# | $U(1)$ ("clocks") | angles $\theta\in[0,2\pi)$ | addition mod $2\pi$ | clock hand at angle $\theta$ |
# | $S_3$ ("shuffles") | permutations of $\lbrace 1,2,3\rbrace$ | composition | cycle notation, e.g. $(12)$ |
# | $SO(3)$ ("cameras") | rotations of 3D space | composition | rotation angle |
#
# The first three are **abelian** (the order of multiplication doesn't matter); the
# last two are not. For permutations we compose like functions: $xy$ means "apply $y$
# first, then $x$".

# %%
S = S3()
fmt = lambda x: viz.format_label(S, x)          # S3 element -> cycle notation
x, y = 2, 1                                      # x = (12), y = (23)
print(f"x = {fmt(x)},  y = {fmt(y)}")
print(f"xy = {fmt(S.mul(x, y))},  yx = {fmt(S.mul(y, x))}   <- different: S3 is not abelian")

# %% [markdown]
# ### 2.2 Labelings
#
# **Definition (labeling, glossary 2.2).** A **labeling** is a function
# $U:\vec E\to G$ with
#
# $$
# U(\bar d) = U(d)^{-1}\quad\text{for every direction } d.
# $$
#
# Read $U(s\to t)$ as *a converter from the conventions at $s$ to the conventions at
# $t$*. Walking the edge backwards must undo the conversion, which is what the rule
# says. (In Chapter 0: if a dollar buys 3.7 shekels, a shekel buys $1/3.7$ dollars.)
#
# Because the reverse label is forced, a labeling is the same thing as one label per
# edge, so the set of all labelings is $G^E$. In code, a labeling is an array with one
# entry per edge (for its reference direction).
#
# ### 2.3 Walk products, loop products, face products
#
# **Definition (walk product, glossary 2.3).** For a walk $\gamma$ that takes the
# directions $d_1, d_2, \dots, d_k$ in order, its **walk product** is
#
# $$
# U(\gamma) = U(d_1)\,U(d_2)\cdots U(d_k)\ \in G .
# $$
#
# **Definition (loop product, face product, defect; glossary 2.4–2.5).** If $\gamma$ is
# closed and starts and ends at $v$, $U(\gamma)$ is a **loop product at $v$**. The loop
# product of a face $f$ is its **face product** $U_f\in G$. A face with $U_f \neq 1$
# (not the identity) is a **defect**.
#
# Here is a random $U(1)$ labeling of the 4×3 grid. Each clock hand is a label. Each
# face is coloured by its face product: pale means identity (angle $0$), and as the angle
# goes around the circle the colour runs through blues and purples to dark, then through
# reds and tans back to pale. The teal loop goes around a $2\times 2$ block of faces.

# %%
G = U1()
U = random_labeling(G, grid, rng)      # U: one angle per edge, an array of shape (|E|,)
block = grid.walk([0, 1, 2, 6, 10, 9, 8, 4, 0])   # vertex list -> list of steps

fig, ax = plt.subplots(figsize=(6.5, 4.8))
viz.draw_board(ax, grid, U, G, highlight_walk=block, title="a U(1) labeling and a loop")
plt.show()

# %% [markdown]
# For an abelian group, the product around the block equals the product of the four
# face products inside it: every interior edge is walked once in each direction and
# cancels. Let's check.

# %%
inside = [i for i, (c, mu, nu) in enumerate(grid.meta["face_anchor"]) if c[0] < 2 and c[1] < 2]
fp = face_products(G, grid, U)                       # one angle per face
print("loop product around the block:   ", round(float(walk_product(G, grid, U, block)), 4))
print("product of the 4 faces inside it:", round(float(G.prod(fp[inside])), 4))

# %% [markdown]
# For a non-abelian group, order matters, and two more facts are worth seeing at work.
#
# > **Fact 1.1 (reversal).** Walking a walk backwards inverts its product: if $\bar\gamma$
# > is $\gamma$ walked backwards, then $U(\bar\gamma) = U(\gamma)^{-1}$.
#
# *Why:* $U(\bar d_k)\cdots U(\bar d_1) = U(d_k)^{-1}\cdots U(d_1)^{-1} = (U(d_1)\cdots U(d_k))^{-1}$.
#
# Moving the **starting point** of a loop changes its product by a conjugation: if the
# loop is $\gamma = \gamma_1\gamma_2$ (first a walk $\gamma_1$ from $v$ to $w$, then a walk
# $\gamma_2$ back to $v$), then starting at $w$ gives
# $U(\gamma_2)U(\gamma_1) = U(\gamma_1)^{-1}\,\big(U(\gamma_1)U(\gamma_2)\big)\,U(\gamma_1)$.

# %%
US = random_labeling(S, grid, rng)                 # an S3 labeling of the same grid
while S.chi(walk_product(S, grid, US, block)) != 0:  # redraw until the product is a 3-cycle
    US = random_labeling(S, grid, rng)              # (a swap is its own inverse: less instructive)
fwd = walk_product(S, grid, US, block)
bwd = walk_product(S, grid, US, grid.reverse_walk(block))
shifted = walk_product(S, grid, US, block[1:] + block[:1])      # same loop, starting at vertex 1
print(f"around the block from 0:    {fmt(fwd)}")
print(f"backwards:                  {fmt(bwd)}   (inverse of {fmt(fwd)} is {fmt(S.inv(fwd))})")
print(f"same loop, starting at 1:   {fmt(shifted)}   (a conjugate of {fmt(fwd)})")

# %% [markdown]
# ### 2.4 Consistent labelings
#
# **Definition (consistent, glossary 2.6).** A labeling is **consistent** if *every*
# loop product is the identity. In Chapter 0 this was "no arbitrage".
#
# Checking every loop sounds expensive, but on a plain grid it's enough to check the faces:
#
# > **Fact 1.2 (faces suffice on a grid).** On a grid without wrap-around, a labeling is
# > consistent if and only if it has no defects.
#
# *Why, roughly:* every loop on a flat grid can be shrunk across faces one at a time,
# and each step changes the loop product only by a face product (and conjugations).
# Section 7 shows that this fails on a torus.
#
# Let's check it by brute force: all $2^{12}=4096$ coin labelings of the 3×3 grid.

# %%
Z = Z2()
small = Board.grid(3, 3)
Us = all_labelings(Z, small)                          # shape (4096, 12)
no_defects = ~defects(Z, small, Us).any(axis=1)
consistent = is_consistent(Z, small, Us)
print(f"{len(Us)} labelings, {no_defects.sum()} with no defects, "
      f"{consistent.sum()} consistent; the two lists agree: {(no_defects == consistent).all()}")

# %% [markdown]
# **An application: cameras.** Suppose each vertex is a camera with an unknown
# orientation $O_v\in SO(3)$, and for each pair of cameras we measure their relative
# rotation $U(s\to t) = O_s O_t^{-1}$. Exact measurements give a consistent labeling:
# around any loop the $O$'s cancel. Real measurements are noisy, so the loop products
# become small non-identity rotations. Recovering the $O_v$ from noisy $U$'s is called
# **synchronization**, and it is used in 3D reconstruction from photos and in cryo-electron
# microscopy.

# %%
R = SO3()
cams = Board.complete(4)                                   # 4 cameras, all pairs compared
O = R.uniform(rng, cams.n_vertices)                        # true orientations, unknown in practice
s, t = cams.edges[:, 0], cams.edges[:, 1]
U_exact = R.mul(O[s], R.inv(O[t]))                         # U(s->t) = O_s O_t^{-1}
noise = np.array([R.about_axis(rng.normal(size=3), rng.normal(0, 0.03)) for _ in s])
U_noisy = R.mul(U_exact, noise)
for name, X in [("exact", U_exact), ("noisy", U_noisy)]:
    ang = np.degrees(R.rotation_angle(face_products(R, cams, X)))
    print(f"{name:5}  rotation angle of each triangle's face product (degrees):", np.round(ang, 2))

# %% [markdown]
# > **Recap.** A labeling puts a group element on every edge direction, with reverse
# > = inverse. Multiplying labels along a walk gives its walk product; around a closed
# > walk it's a loop product, around a face a face product. Reversing a walk inverts the
# > product; moving a loop's start conjugates it. A labeling is consistent when every
# > loop product is the identity; on a plain grid, that's the same as having no defects.
#
# **Check yourself**
#
# 1. For $G=\mathbb Z_2$, what is a face product in words?
#
# <details><summary>Answer</summary>
#
# It is $(-1)^k$ where $k$ is the number of $-1$ edges around the face. So a face is a defect exactly when an odd number of its edges carry $-1$.
#
# </details>
#
# 2. For an abelian group, does the starting point of a loop affect its product?
#
# <details><summary>Answer</summary>
#
# No. Moving the start conjugates the product, and in an abelian group $a^{-1}xa = x$.
#
# </details>
#
# 3. In the camera example, why is the exact labeling consistent?
#
# <details><summary>Answer</summary>
#
# Around a closed walk $v_0\to v_1\to\cdots\to v_k=v_0$ the product is $O_{v_0}O_{v_1}^{-1}O_{v_1}O_{v_2}^{-1}\cdots O_{v_{k-1}}O_{v_0}^{-1} = 1$: every inner pair cancels.
#
# </details>

# %% [markdown]
# ---
# ## 3. Reframings
#
# The camera example already contains the key idea. The labels $O_sO_t^{-1}$ were built
# from one choice per **vertex**. If you rotated all your notes about camera $v$ (you
# decided "up" means something else for that camera), every label touching $v$ would
# change, yet nothing physical about the cameras would.
#
# **Definition (reframing, glossary 3.1–3.2).** A **reframing** is a function
# $h:V\to G$. It acts on a labeling by
#
# $$
# U(s\to t)\ \longmapsto\ h(s)\,U(s\to t)\,h(t)^{-1}.
# $$
#
# The set of all reframings, $G^V$, is a group under vertex-by-vertex multiplication; we
# call it the **reframing group**.
#
# Each label is a converter between the conventions at its two endpoints, so a change of
# convention at either endpoint enters it: $h(s)$ on the left, $h(t)^{-1}$ on the right.
# (Check that the reverse direction stays the inverse: $h(t)U(s\to t)^{-1}h(s)^{-1}$ is
# indeed the inverse of $h(s)U(s\to t)h(t)^{-1}$.)
#
# The payoff of this exact shape is that along a walk the inner factors **telescope**:
#
# $$
# \big(h(a)\,U_1\,h(v)^{-1}\big)\big(h(v)\,U_2\,h(b)^{-1}\big) = h(a)\,U_1U_2\,h(b)^{-1}.
# $$
#
# Here $U_1 = U(a\to v)$ and $U_2 = U(v\to b)$ are the labels of two consecutive steps.
#
# > **Fact 1.3 (how reframing moves products; glossary 3.5).** If $\gamma$ is a walk
# > from $a$ to $b$, a reframing changes its product to $h(a)\,U(\gamma)\,h(b)^{-1}$.
# > In particular a loop product at $v$ becomes $h(v)\,U(\gamma)\,h(v)^{-1}$:
# > it is **conjugated**.
#
# For an abelian group, conjugation does nothing, so *loop products are untouched*. For
# a non-abelian group the loop product can change, but only within its conjugacy class.
# Here is an $S_3$ labeling of the triangle, before and after a reframing:

# %%
U = np.array([2, 1, 1])                 # labels (12), (23), (23) on 0->1, 1->2, 2->0
h = np.array([1, 3, 0])                 # reframing: h(0)=(23), h(1)=(123), h(2)=e
U2 = reframe(S, tri, U, h)
for name, X in [("before", U), ("after ", U2)]:
    w = face_products(S, tri, X)[0]
    print(f"{name}: labels {[fmt(x) for x in X]}   loop product {fmt(w):5}   "
          f"fixed points of loop product: {S.chi(w):.0f}")
print(f"check: h(0) w h(0)^-1 = {fmt(S.conj(h[0], face_products(S, tri, U)[0]))}")

# %% [markdown]
# Every label changed and the loop product changed from $(12)$ to $(13)$. But both are
# swaps of two cards: same conjugacy class, same number of fixed points.
#
# ### Frozen faces, spinning edges
#
# For an abelian group the effect is striking. Below, a $U(1)$ labeling of a 3×3 grid is
# hit by a smoothly varying reframing: every vertex slowly turns its own "zero
# direction". **Watch the clock hands spin while the face colours do not move at all.**
# The faces are what the labeling *means*; the hands are how it is *written down*.

# %%
G = U1()
U = random_labeling(G, small, np.random.default_rng(7))
anim = viz.animate_reframing(small, G, U, n_frames=36, figsize=(4.2, 4.2), interval=110)
HTML(anim.to_jshtml())

# %% [markdown]
# Reframings compose the way you'd hope: doing $h_1$ and then $h_2$ is the same as doing
# the single reframing $h_2h_1$ (multiply vertex by vertex). This is what it means for
# the reframing group to *act* on labelings.

# %%
h1, h2 = random_reframing(S, grid, rng), random_reframing(S, grid, rng)
one_by_one = reframe(S, grid, reframe(S, grid, US, h1), h2)
combined = reframe(S, grid, US, S.mul(h2, h1))
print("h1 then h2  ==  h2·h1 :", np.array_equal(one_by_one, combined))

# %% [markdown]
# > **Recap.** A reframing $h:V\to G$ changes conventions vertex by vertex and acts by
# > $U(s\to t)\mapsto h(s)U(s\to t)h(t)^{-1}$. Along a walk the inner factors cancel, so
# > only the endpoints feel it; a loop product at $v$ gets conjugated by $h(v)$. For
# > abelian groups, loop products don't change at all.
#
# **Check yourself**
#
# 1. Let $h$ be constant, $h(v)=g$ for every $v$. What does it do to a labeling when $G$ is abelian? When it isn't?
#
# <details><summary>Answer</summary>
#
# Abelian: $gUg^{-1}=U$, so nothing at all. Non-abelian: it conjugates *every* label by the same $g$.
#
# </details>
#
# 2. Write out the telescoping for a three-step walk $a\to v\to w\to b$.
#
# <details><summary>Answer</summary>
#
# $h(a)U_1h(v)^{-1}\cdot h(v)U_2h(w)^{-1}\cdot h(w)U_3h(b)^{-1} = h(a)\,U_1U_2U_3\,h(b)^{-1}$.
#
# </details>
#
# 3. In the camera example, the exact labeling was $U(s\to t) = O_sO_t^{-1}$. Express it as a reframing of something.
#
# <details><summary>Answer</summary>
#
# It is the reframing $h=O$ applied to the identity labeling (every label $=1$): $O_s\cdot 1\cdot O_t^{-1}$.
#
# </details>

# %% [markdown]
# ---
# ## 4. Invariants and loop observables
#
# Since reframings only change the *description*, the interesting functions of a
# labeling are the ones that ignore them.
#
# **Definition (invariant, glossary 3.4).** An **invariant** is a function
# $f: G^E\to\mathbb C$ such that $f(U') = f(U)$ whenever $U'$ is a reframing of $U$.
#
# For abelian $G$, Fact 1.3 hands us invariants for free: every loop product. For
# non-abelian $G$ a loop product only survives up to conjugation, so we need a way to
# read off a number that doesn't notice conjugation.
#
# **Definition (class function, glossary 3.6).** A **class function** is a function
# $\chi: G\to\mathbb C$ with $\chi(hxh^{-1}) = \chi(x)$ for all $h, x\in G$.
#
# The most useful class function is the **trace** of a matrix, since
# $\operatorname{tr}(hxh^{-1}) = \operatorname{tr}(x)$. Each of our groups comes with
# one, available as `G.chi`:
#
# | $G$ | $\chi(x)$ |
# |---|---|
# | $\mathbb Z_2$ | $x$ itself |
# | $U(1)$, $\mathbb Z_N$ | $\cos(\text{angle})$ |
# | $S_3$ | number of fixed points (the trace of the $3\times3$ permutation matrix) |
# | $SO(3)$ | trace $= 1 + 2\cos(\text{rotation angle})$ |
#
# **Definition (loop observable, glossary 3.7).** For a closed walk $\gamma$ and a
# class function $\chi$, the **loop observable** is
# $W_\gamma(U) = \chi\big(U(\gamma)\big)\in\mathbb C$.
#
# > **Fact 1.4.** Every loop observable is an invariant.
#
# *Why:* the loop product gets conjugated (Fact 1.3) and $\chi$ doesn't notice
# conjugation. Let's test it with random rotations on the 3×3 grid: the face products
# change, their traces don't. A walk that is *not* closed gives no invariant.

# %%
U = random_labeling(R, small, rng)
h = random_reframing(R, small, rng)
U2 = reframe(R, small, U, h)
fp, fp2 = face_products(R, small, U), face_products(R, small, U2)
boundary = small.walk([0, 1, 2, 5, 8, 7, 6, 3, 0])
open_walk = small.walk([0, 1, 2])
print("face products unchanged?        ", np.allclose(fp, fp2))
print("traces of face products unchanged?", np.allclose(R.chi(fp), R.chi(fp2)))
print("trace around the boundary:  before %.4f  after %.4f" % (
    R.chi(loop_product(R, small, U, boundary)), R.chi(loop_product(R, small, U2, boundary))))
print("trace along an open walk:   before %.4f  after %.4f" % (
    R.chi(walk_product(R, small, U, open_walk)), R.chi(walk_product(R, small, U2, open_walk))))

# %% [markdown]
# > **Recap.** An invariant is a function of a labeling that every reframing leaves
# > alone. Loop products are invariants for abelian groups; for non-abelian groups we
# > apply a class function such as the trace, giving loop observables
# > $W_\gamma = \chi(U(\gamma))$.
#
# **Check yourself**
#
# 1. Is "the label on edge $e$" an invariant?
#
# <details><summary>Answer</summary>
#
# No. Reframing at either endpoint of $e$ changes it (unless $G$ is trivial).
#
# </details>
#
# 2. Is "the number of defects" an invariant?
#
# <details><summary>Answer</summary>
#
# Yes. Each face product is conjugated, and $x = 1$ if and only if $hxh^{-1}=1$, so a face is a defect before the reframing exactly when it is one after.
#
# </details>
#
# 3. Is the rotation angle of an $SO(3)$ loop product an invariant?
#
# <details><summary>Answer</summary>
#
# Yes: it is a function of the trace ($\text{angle} = \arccos\frac{\chi-1}{2}$), hence a class function of the loop product.
#
# </details>

# %% [markdown]
# ---
# ## 5. Classes, and how many there are
#
# **Definition (class, glossary 3.3).** The **class** of a labeling $U$ is
#
# $$
# [U] = \lbrace \text{all labelings obtained from } U \text{ by some reframing}\rbrace \subseteq G^E.
# $$
#
# Two labelings are in the same class exactly when they describe the same thing in
# different conventions. Invariants are precisely the functions of the class.
#
# How many classes are there? For a finite group, there are $|G|^{|E|}$ labelings and
# $|G|^{|V|}$ reframings. If the only reframing that fixed a labeling were the trivial one
# ($h(v)=1$ for all $v$), each class would have $|G|^{|V|}$ members. But for an abelian
# group the constant reframings ($h(v) = g$ for all $v$) do nothing, and on a connected
# board they are the only reframings that fix a labeling. That gives:
#
# > **Fact 1.5 (counting classes, abelian).** For a finite abelian $G$ on a connected
# > board, every class has exactly $|G|^{|V|-1}$ members, so there are
# > $|G|^{|E|}/|G|^{|V|-1} = |G|^{|E|-|V|+1}$ classes. The exponent $|E|-|V|+1$ is the
# > number of independent cycles of the graph.
#
# We check this by brute force: list every labeling, sort them into classes, count.

# %%
def count_classes(G, X):
    Us = all_labelings(G, X)
    return len(Us), class_ids(G, X, Us).max() + 1

for G in (Z2(), ZN(3)):
    for X in (Board.triangle(), Board.grid(2, 2), Board.grid(3, 2), Board.grid(2, 2, periodic=True)):
        n, k = count_classes(G, X)
        print(f"{G.name:3} {X.name:18}  labelings {n:5}   classes {k:4}   "
              f"|G|^(E-V+1) = {G.order ** X.cycle_rank}")

# %% [markdown]
# For a non-abelian group, some labelings are fixed by more reframings than others (for
# instance the identity labeling is fixed by every constant reframing), so classes have
# different sizes and the simple division fails. A standard counting argument (Burnside's
# lemma) gives the answer:
#
# > **Fact 1.6 (counting classes, any finite $G$).** With $b = |E|-|V|+1$, the number of
# > classes is $\frac{1}{|G|}\sum_{g\in G} |C(g)|^{b}$,
# > where $C(g) = \lbrace x\in G : xg = gx\rbrace$ is the set of elements commuting with $g$.
#
# For abelian $G$ every $C(g)$ is all of $G$ and this collapses to $|G|^b$.

# %%
def burnside(G, b):
    els = G.elements()
    commuting = [sum(bool(G.equal(G.mul(g, x), G.mul(x, g))) for x in els) for g in els]
    return sum(c ** b for c in commuting) // len(els)

for X in (Board.triangle(), Board.grid(2, 2), Board.grid(3, 2)):
    n, k = count_classes(S, X)
    print(f"S3 {X.name:10}  labelings {n:6}   classes {k:3}   Burnside formula {burnside(S, X.cycle_rank)}")

# %% [markdown]
# On the triangle, $S_3$ has 3 classes: one per conjugacy class of the loop product
# (identity, swaps, 3-cycles). Here are the class sizes on the 3×2 grid, which are
# indeed unequal:

# %%
ids = class_ids(S, Board.grid(3, 2), all_labelings(S, Board.grid(3, 2)))
sizes, how_many = np.unique(np.bincount(ids), return_counts=True)
for sz, m in zip(sizes, how_many):
    print(f"{m:3} classes of size {sz}")

# %% [markdown]
# ### Seeing classes: the triangle with clock labels
#
# For $U(1)$ there are infinitely many labelings, but on the triangle we can *see* all
# of them. A labeling is three angles $(\theta_{01},\theta_{12},\theta_{20})$, one per
# edge, so the set of labelings is a cube $[0,2\pi)^3$ whose opposite sides are glued
# together (an angle of $2\pi$ is an angle of $0$). Every invariant is a function of the
# loop product $\theta_{01}+\theta_{12}+\theta_{20} \pmod{2\pi}$, so each class is a level set
# of that sum: a flat sheet that gets cut into pieces where it hits the cube's walls.
#
# The figure shows three classes, coloured by their loop product, together with 600
# random reframings of one labeling (black dots). **Look at where the dots land: all on
# one colour, spread over all of its pieces.** One labeling's reframings fill out its
# entire sheet, and never leave it. Collapsing each sheet to a single point leaves one
# number, the loop product: the space of classes is a circle.

# %%
def plane_in_cube(s):
    """Corners of the polygon where the plane x+y+z = s cuts the cube [0, 2π]^3."""
    corners = [np.array(c, float) for c in itertools.product([0, TAU], repeat=3)]
    pts = []
    for a, b in itertools.combinations(corners, 2):
        sa, sb = a.sum(), b.sum()
        if np.sum(a != b) == 1 and sa != sb and (sa - s) * (sb - s) <= 0:
            pts.append(a + (s - sa) / (sb - sa) * (b - a))
    pts = np.unique(np.round(pts, 9), axis=0)
    u, v = np.array([1, -1, 0]) / 2**0.5, np.array([1, 1, -2]) / 6**0.5   # axes in the plane
    d = pts - pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(d @ v, d @ u))]

def sheet_traces(c, color, name):
    """All pieces of the class 'loop product = c' as Plotly meshes."""
    out = []
    for k in range(3):                                   # the sum can be c, c+2π or c+4π
        P = plane_in_cube(c + k * TAU)
        if len(P) < 3:
            continue
        n = len(P)
        out.append(go.Mesh3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], i=[0] * (n - 2),
                             j=list(range(1, n - 1)), k=list(range(2, n)), color=color,
                             opacity=0.45, name=name, showlegend=(k == 0), flatshading=True))
    return out

# %%
cmap = plt.get_cmap("twilight")
classes = [0.6, 2.4, 4.4]                                  # three loop products (radians)
traces = []
for c in classes:
    r, g, b, _ = cmap(c / TAU)
    traces += sheet_traces(c, f"rgb({255*r:.0f},{255*g:.0f},{255*b:.0f})",
                           f"class with loop product {c / np.pi:.2f}π")

theta = np.array([1.0, 2.0, 2.4 - 3.0 + TAU])               # a labeling with loop product 2.4
hs = rng.uniform(0, TAU, size=(600, 3))                     # 600 random reframings
orbit = reframe(U1(), tri, theta, hs)                       # shape (600, 3)
traces.append(go.Scatter3d(x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], mode="markers",
                           marker=dict(size=2, color="#2b2d42"), name="reframings of one labeling"))

# %%
edges = [(a, b) for a, b in itertools.combinations(itertools.product([0, TAU], repeat=3), 2)
         if sum(x != y for x, y in zip(a, b)) == 1]
for a, b in edges:                                          # wireframe of the cube
    traces.append(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                               line=dict(color="#8d99ae", width=2), showlegend=False, hoverinfo="skip"))
ax = dict(range=[0, TAU], tickvals=[0, np.pi, TAU], ticktext=["0", "π", "2π"])
fig = go.Figure(traces)
fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0, y=1),
                  scene=dict(xaxis=dict(title="θ(0→1)", **ax), yaxis=dict(title="θ(1→2)", **ax),
                             zaxis=dict(title="θ(2→0)", **ax), aspectmode="cube"))
fig.show()

# %% [markdown]
# > **Recap.** The class of $U$ is everything reframings can turn it into; invariants are
# > functions of the class. For a finite abelian group there are $|G|^{|E|-|V|+1}$
# > classes, all the same size. For non-abelian groups the sizes differ and Burnside's
# > formula counts them. For $U(1)$ on the triangle, classes are sheets in a glued cube,
# > labelled by one angle.
#
# **Check yourself**
#
# 1. How many classes does $\mathbb Z_2$ have on the 3×3 grid (no wrap-around)? What determines a class?
#
# <details><summary>Answer</summary>
#
# $|E|-|V|+1 = 12-9+1 = 4$, so $2^4 = 16$ classes. There are exactly 4 faces, and by Facts 1.2 and 1.5 a class is pinned down by which of the 4 faces are defects ($2^4$ possibilities, all of which occur).
#
# </details>
#
# 2. For $S_3$ on the triangle, name a representative labeling of each of the 3 classes.
#
# <details><summary>Answer</summary>
#
# Put labels $(x, e, e)$ on the three edges with $x = e$, $x = (12)$, or $x = (123)$. Their loop products are $x$, which lie in the three different conjugacy classes.
#
# </details>
#
# 3. In the cube picture, how many dimensions does a class have, and why?
#
# <details><summary>Answer</summary>
#
# Two. A reframing has 3 angles, but adding the same angle at all three vertices does nothing (abelian group), leaving $3-1=2$ directions of motion. That is the same count, $|V|-1$, as in Fact 1.5.
#
# </details>

# %% [markdown]
# ---
# ## 6. Tree normal form: one representative per class
#
# A class is a large set. It would be convenient to pick one canonical member, the way
# you'd reduce a fraction to lowest terms. Chapter 0 did this for currencies by quoting
# every rate in dollars along a tree of exchanges.
#
# **Construction.** Fix a spanning tree $T$ and a root vertex $r$. For each vertex $v$,
# let $h(v)$ be the walk product from $r$ to $v$ along the tree. Reframing by $h$
# makes every tree edge carry the identity: for a tree edge $p\to v$ we have
# $h(v) = h(p)U(p\to v)$, so the new label is
# $h(p)\,U(p\to v)\,h(v)^{-1} = 1$.
#
# > **Fact 1.7 (tree normal form; glossary 3.8).** Every class contains a labeling that is
# > the identity on every edge of $T$. On each of the other $|E|-|V|+1$ edges, its label
# > is the loop product of that edge's **fundamental cycle**: go from $r$ to $s$ in the
# > tree, cross the edge $s\to t$, return from $t$ to $r$ in the tree. For abelian $G$ this
# > representative is unique; for non-abelian $G$ it is unique up to conjugating all labels
# > by one common element.
#
# The free labels are exactly the $|E|-|V|+1$ numbers that Fact 1.5 counted.

# %%
G = U1()
U = random_labeling(G, grid, rng)
Un, h, tree = tree_normal_form(G, grid, U)        # Un = reframe(G, grid, U, h)
cycles = grid.fundamental_cycles()                # {non-tree edge: its fundamental cycle}
e0 = sorted(cycles, key=lambda e: len(cycles[e]))[len(cycles) // 2]
ok = all(G.equal(Un[e], loop_product(G, grid, U, cycles[e])) for e in cycles)
print(f"{len(tree)} tree edges, {len(cycles)} free edges; "
      f"free labels = fundamental loop products: {ok}")

# %% [markdown]
# Left: the random labeling. Right: its tree normal form. The grey tree edges all point
# their hands at angle $0$ (the identity). **The face colours are identical in the two
# pictures**, because it's the same class. The teal walk is the fundamental cycle of one
# free edge, and that edge's clock shows its loop product.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
viz.draw_board(axs[0], grid, U, G, title="a labeling U")
viz.draw_board(axs[1], grid, Un, G, tree_edges=tree, highlight_walk=cycles[e0],
               title="its tree normal form (tree in grey)")
plt.show()

# %% [markdown]
# > **Recap.** Choosing a spanning tree and pushing all labels off it gives a canonical
# > representative of each class: identity on the tree, fundamental loop products on the
# > remaining $|E|-|V|+1$ edges.
#
# **Check yourself**
#
# 1. Two labelings have the same tree normal form (same tree and root, abelian $G$). What can you conclude?
#
# <details><summary>Answer</summary>
#
# They are in the same class: each is a reframing of the common normal form, and reframings can be undone and composed.
#
# </details>
#
# 2. Why is the non-abelian normal form only unique up to a common conjugation?
#
# <details><summary>Answer</summary>
#
# The reframings that keep the tree at the identity are exactly the constant ones, $h(v)=g$. They conjugate every free label by the same $g$.
#
# </details>

# %% [markdown]
# ---
# ## 7. The torus: when faces don't tell the whole story
#
# Fact 1.2 said that on a plain grid, "no defects" means "consistent". On a torus it
# doesn't. Take the coin group on the 3×3 torus and flip the three horizontal edges that
# cross one vertical line (a "seam"). Each face either misses the seam or crosses it
# twice, so no face is a defect. But a loop that goes once around the torus horizontally
# crosses the seam exactly once.

# %%
T, Z = Board.grid(3, 3, periodic=True), Z2()
U = identity_labeling(Z, T)
seam = [T.meta["edge_index"][(v, 0)] for v in (0, 3, 6)]   # horizontal edges from column 0 to 1
U[seam] = -1
row = T.walk([0, 1, 2, 0])                                  # once around, horizontally
col = T.walk([0, 3, 6, 0])                                  # once around, vertically
print("defects:", int(defects(Z, T, U).sum()),
      "  | around a row:", int(loop_product(Z, T, U, row)),
      "  | around a column:", int(loop_product(Z, T, U, col)),
      "  | consistent?", bool(is_consistent(Z, T, U)))

# %% [markdown]
# Left: the seam (gold edges are $-1$); every face is pale, so there are no defects.
# Right: the same labeling with the bottom row's loop (once around the torus) in teal. It
# uses exactly one seam edge, so its loop product is $-1$.

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
viz.draw_board(axs[0], T, U, Z, title="a seam: no defects")
viz.draw_board(axs[1], T, U, Z, highlight_walk=row, title="...but this loop has product −1")
plt.show()

# %% [markdown]
# Counting confirms there's room for exactly this kind of thing. The 3×3 torus has
# $|E|-|V|+1 = 18-9+1 = 10$ independent cycles but only 9 faces, and the faces aren't
# even independent:
#
# > **Fact 1.8 (faces on a torus).** For abelian $G$ on a periodic grid, the product of
# > all face products is the identity. So the faces carry only $|F|-1$ independent pieces
# > of information, and the remaining $(|E|-|V|+1) - (|F|-1) = 2$ are the loop products
# > around the two directions of the torus. In particular there are $|G|^2$ classes with
# > no defects at all.
#
# *Why the first part:* every edge borders exactly two faces and is walked once in each
# direction, so in an abelian group every label cancels against its inverse.

# %%
T4 = Board.grid(4, 4, periodic=True)
U = random_labeling(U1(), T4, rng)
everything = U1().prod(face_products(U1(), T4, U))
print("4×4 torus, random U(1) labeling: product of all 16 face products is the identity?",
      bool(U1().is_identity(everything)))
Us = all_labelings(Z, T)                                   # all 2^18 coin labelings
ids = class_ids(Z, T, Us)
clean = ~defects(Z, T, Us).any(axis=1)
print(f"{len(Us)} labelings, {ids.max() + 1} classes, "
      f"{len(np.unique(ids[clean]))} classes with no defects (Fact 1.8 predicts 2^2 = 4)")

# %% [markdown]
# > **Recap.** On a torus, a labeling can be defect-free and still have non-identity loop
# > products around the holes. For abelian groups the faces carry $|F|-1$ independent
# > pieces of information and the two winding loops carry the rest.
#
# **Check yourself**
#
# 1. On an $L\times L$ torus with $G=\mathbb Z_N$, how many classes have no defects?
#
# <details><summary>Answer</summary>
#
# $N^2$: one for each choice of the two winding loop products.
#
# </details>
#
# 2. Where does the argument "product of all face products $=1$" use that $G$ is abelian?
#
# <details><summary>Answer</summary>
#
# To cancel a label against its inverse, the two must be brought next to each other in the product. That reordering is only allowed when $G$ is abelian; for non-abelian $G$ the faces' products start at different vertices and come in different orders.
#
# </details>
#
# 3. On the 3×3 torus, describe a coin labeling whose loop product is $-1$ around *both* a row and a column but which has no defects.
#
# <details><summary>Answer</summary>
#
# Combine two seams: flip the three horizontal edges crossing one vertical line *and* the three vertical edges crossing one horizontal line. Each face still crosses each seam zero or two times.
#
# </details>

# %% [markdown]
# ---
# ## Chapter recap
#
# - A **board** is a connected graph with a chosen list of closed walks called **faces**.
# - A **labeling** $U:\vec E\to G$ puts a group element on every direction, with reverse = inverse.
#   Multiplying along a walk gives its **walk product**; around closed walks, **loop products**; around faces, **face products**.
# - A **reframing** $h:V\to G$ acts by $U(s\to t)\mapsto h(s)U(s\to t)h(t)^{-1}$. Walk products
#   only feel it at their endpoints; loop products are conjugated.
# - **Invariants** are the functions that every reframing leaves alone. Loop products for abelian $G$,
#   and **loop observables** $\chi(U(\gamma))$ for any class function $\chi$, are invariants.
# - The **class** $[U]$ is the set of all reframings of $U$. For finite abelian $G$ there are
#   $|G|^{|E|-|V|+1}$ classes.
# - **Tree normal form** picks one representative per class: identity on a spanning tree,
#   fundamental loop products elsewhere.
# - On a torus, faces don't see everything: the loops around the holes carry two extra invariants.
#
# ## Dictionary
#
# | Our term | Other names you'll meet |
# |---|---|
# | board | lattice, 2-complex; "spacetime" in physics |
# | face | plaquette (physics), 2-cell (topology) |
# | label group | gauge group (physics), gain group (gain graphs), structure group (geometry) |
# | labeling | gauge field, configuration, link variables (physics); gain function (gain graphs); connection (geometry, synchronization) |
# | walk product | parallel transport, Wilson line (physics); gain of a walk |
# | loop product | holonomy, monodromy |
# | face product | plaquette variable (physics); curvature, field strength (geometry) |
# | defect | vortex, frustrated plaquette (spin glasses) |
# | consistent | flat, pure gauge (physics); balanced (gain graphs); cycle-consistent (vision) |
# | reframing | gauge transformation (physics); switching (gain graphs) |
# | reframing group | gauge group (in mathematics) |
# | class | gauge orbit, physical configuration (physics); switching class (gain graphs) |
# | invariant | gauge-invariant observable |
# | loop observable | Wilson loop |
# | tree normal form | gauge fixing, maximal-tree or axial gauge |
#
# ## What's next
#
# **Interlude (optional, recommended).** Why should edges carry labels at all? The interlude *Why the labels are forced* puts values on vertices and shows that comparing neighbors requires a label on the edge between them. For the circle group, that is electromagnetism.
#
# So far a labeling was just *given* to us. In Chapter 2 we choose labelings **at random**,
# with a distribution that prefers few and mild defects. That raises an immediate
# question: if the labels are random, but only their class has meaning, does the
# redundancy we found here distort the probabilities? (It doesn't. The class counts of
# Section 5 and the tree normal form of Section 6 are both part of the reason.)
