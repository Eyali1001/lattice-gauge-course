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
# # Chapter 2 — Models: putting a probability distribution on labelings
#
# Take the $2\times2$ grid wrapped around in both directions, so that it becomes a torus as
# in chapter 1. It has 4 vertices, 8 edges and 4 faces. Flip a
# fair coin for every edge and write $+1$ or $-1$ on it. How many faces do you expect to be
# **defects**, meaning faces whose face product is $-1$? Now suppose someone tells you that
# defects are *expensive*, and that a labeling with many of them should be less likely.
# How would you tilt the coins?
#
# This chapter answers that question, and the answer is the central object of the whole
# course. We stop looking at **one** labeling and start looking at a **probability
# distribution over all labelings**. We call that distribution a *model*. Everything that
# comes later (averages, correlations, and in chapter 5 even "dynamics") will be read off
# from the model alone.
#
# **What you'll learn**
# - What it means to pick a group element "uniformly at random", and why that needs a
#   *compact* group.
# - How a cost on faces and a single number $\beta$ (the *stiffness*) turn the uniform
#   distribution into a model.
# - Why the redundancy from chapter 1 (reframings) does no harm, and why, on a planar grid with
#   the coin group, the faces are independent coins.
#
# **Prerequisites:** chapter 1 (boards, labelings, face products, reframings, classes, tree
# normal form). Chapter 0 helps for intuition but is not required.

# %%
import numpy as np
import matplotlib.pyplot as plt
from lgt import *
from lgt import viz

viz.use_style()
rng = np.random.default_rng(2)

# %% [markdown]
# ---
# ## 1. Choosing a label "uniformly at random"
#
# Before we can make labelings *unequally* likely, we need to know what *equally likely*
# means. For a fair coin this is obvious: $+1$ and $-1$ each get probability $\tfrac12$. For
# a general group the right notion is this: picking an element at random and then multiplying
# it by any fixed element should not change the odds.
#
# **Definition (uniform distribution, glossary 4.1).** Let $G$ be a group. A *uniform
# distribution* on $G$ is a probability distribution $\sigma$ on $G$ such that, for every
# fixed $g\in G$ and every set $A\subseteq G$,
#
# $$
# \sigma(gA)=\sigma(A),\qquad gA=\lbrace ga : a\in A\rbrace .
# $$
#
# In words: if $x$ is drawn from $\sigma$, then $gx$ is drawn from $\sigma$ too.
#
# Our examples:
# - **Coin group** $\mathbb Z_2=\lbrace +1,-1\rbrace$: each element gets probability $\tfrac12$.
# - **$\mathbb Z_N$** (integers mod $N$): each element gets probability $\tfrac1N$.
# - **Circle** $U(1)$ (angles mod $2\pi$): the angle is uniform on $[0,2\pi)$.
# - **Rotations** $SO(3)$: a "uniformly random rotation". As we'll see in a moment, this is
#   not what you might first guess.
#
# > **Fact 2.1 (when a uniform distribution exists).** A uniform distribution on $G$ exists
# > exactly when $G$ is *compact*, and then it is unique. For us, "compact" means one of
# > two things: finite (like $\mathbb Z_2$, $\mathbb Z_N$, the permutations $S_3$), or a
# > closed and bounded set of matrices (like the circle or the rotations $SO(3)$).
#
# The exchange rates of chapter 0 live in $\mathbb R_{>0}$, which is **not** compact.
# Picking a "uniformly random positive number" is as impossible as picking a uniformly random
# integer. That is why chapter 0 had no randomness in it.

# %%
for G in [Z2(), ZN(5), U1()]:
    print(f"{str(G):>4}: eight uniform draws ->", np.round(G.uniform(rng, 8), 2))

# %% [markdown]
# ### A surprise: a uniform rotation does *not* have a uniform angle
#
# Every rotation of 3D space turns by some angle $\omega\in[0,\pi]$ about some axis. You
# might expect a uniformly random rotation to have a uniformly random angle. It doesn't.
#
# > **Fact 2.2 (angle of a uniform rotation).** If $R$ is drawn from the uniform distribution
# > on $SO(3)$, its rotation angle $\omega\in[0,\pi]$ has density
# > $p(\omega)=(1-\cos\omega)/\pi$.
#
# *Why, roughly:* the rotations by a fixed angle $\omega$ form a whole sphere's worth of
# axes, and that sphere gets bigger as $\omega$ grows. Large angles have "more room", so tiny
# rotations are rare.
#
# The figure checks two things. The filled histogram is $200{,}000$ uniform rotations; look
# at how it follows the dark curve and not the flat dashed line. The outlined histogram is
# the same rotations each multiplied by one fixed rotation $g$. It is indistinguishable from
# the first, which is exactly the defining property of "uniform".

# %%
R = SO3()
w = R.rotation_angle(R.uniform(rng, 200_000))
g = R.about_axis([1, 2, 0], 2.0)                       # one fixed rotation
w_g = R.rotation_angle(R.mul(g, R.uniform(rng, 200_000)))
om = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.hist(w, bins=60, density=True, alpha=0.5, label="uniform rotations $R$")
ax.hist(w_g, bins=60, density=True, histtype="step", lw=2, label="the same, each turned into $gR$")
ax.plot(om, (1 - np.cos(om)) / np.pi, color=viz.INK, lw=2, label=r"$(1-\cos\omega)/\pi$")
ax.axhline(1 / np.pi, ls="--", color=viz.MUTED, label="flat: what you might have guessed")
ax.set(xlabel=r"rotation angle $\omega$", ylabel="density", xlim=(0, np.pi))
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.show()

# %% [markdown]
# > **Recap.** A uniform distribution $\sigma$ on a group $G$ is one that multiplying by a fixed
# > element leaves unchanged. It exists (and is unique) exactly for compact groups: finite
# > groups, the circle, rotations. It does not exist for $\mathbb R_{>0}$. "Uniform" is a
# > statement about the group, not about any coordinate, so a uniform rotation does not have
# > a uniform angle.
#
# **Check yourself**
#
# 1. Show from the definition that the uniform distribution on $\mathbb Z_N$ must give every
#    element probability $1/N$.
#
# <details><summary>Answer</summary>
#
# Take $g=1$ (adding 1). Invariance says $\sigma(\lbrace k+1\rbrace)=\sigma(\lbrace k\rbrace)$
# for every $k$, so all $N$ elements have the same probability. The probabilities sum to 1,
# so each is $1/N$.
#
# </details>
#
# 2. Why can't there be a uniform distribution on $\mathbb R_{>0}$ (with multiplication)?
#
# <details><summary>Answer</summary>
#
# Invariance under multiplying by 2 would force the intervals $[1,2),[2,4),[4,8),\dots$ (and
# $[\tfrac12,1),\dots$) to all have the same probability. There are infinitely many of these
# disjoint intervals, so that common probability must be 0. But they cover all of
# $\mathbb R_{>0}$, so the total would be 0, not 1.
#
# </details>
#
# 3. Here is a recipe for a random rotation: pick a uniformly random axis and an angle
#    uniform in $[0,\pi]$. Is the result uniform on $SO(3)$?
#
# <details><summary>Answer</summary>
#
# No. Its angle histogram would be the flat dashed line, not $(1-\cos\omega)/\pi$, so it gives
# far too many small rotations.
#
# </details>

# %% [markdown]
# ---
# ## 2. Putting a price on defects: face cost and total cost
#
# Now the tilt. We want labelings with defects to be *less likely*, and big defects even less
# likely than small ones. So we first attach a price to every face.
#
# **Definition (face cost, glossary 4.2).** A *face cost* is a function
# $\varphi:G\to\mathbb R$ that is a class function, so $\varphi(hxh^{-1})=\varphi(x)$, and
# that is smallest at the identity. We use the standard choice, written in `lgt` as
# `standard_face_cost(G)`:
#
# | group | face cost $\varphi(x)$ |
# |---|---|
# | coin $\mathbb Z_2$ | $\varphi(+1)=0,\ \varphi(-1)=2$ |
# | circle $U(1)$, and $\mathbb Z_N$ drawn as angles | $1-\cos\theta$ |
# | matrices ($S_3$ as permutation matrices, $SO(3)$) | $\operatorname{Re}\operatorname{tr}(I-x)$ |
#
# **Definition (total cost, glossary 4.3).** For a board $X$ with faces $F$, the *total cost*
# of a labeling $U$ is
#
# $$
# S(U)=\sum_{f\in F}\varphi(U_f)\ \in\mathbb R_{\ge0},\qquad S:G^E\to\mathbb R_{\ge 0}.
# $$
#
# For the coin group, $S(U)=2\times(\text{number of defects})$.
#
# The plot shows the face cost on the circle, with the elements of $\mathbb Z_2$ and
# $\mathbb Z_5$ sitting on the same curve. $\mathbb Z_N$ is a finite "pixelation" of the
# circle.

# %%
th = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(th, standard_face_cost(U1())(th), color=viz.INK, lw=2, label=r"circle: $\varphi(\theta)=1-\cos\theta$")
for G, mk, ms in [(ZN(5), "s", 8), (Z2(), "o", 15)]:
    x = G.elements()
    ax.plot(G.angle(x), standard_face_cost(G)(x), mk, ms=ms, mfc="none", mew=2.5,
            label=f"elements of {G}")
ax.set(xlabel=r"face product, as an angle $\theta$", ylabel=r"$\varphi$",
       xticks=[0, np.pi, 2 * np.pi], xticklabels=["0", r"$\pi$", r"$2\pi$"])
ax.legend(frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# Why do we insist that $\varphi$ is a class function? A face product depends on where you
# start walking around the face, and on the reframing. Both change it only by a conjugation
# (glossary 3.5). A class function doesn't notice either.
#
# > **Fact 2.3 (total cost is an invariant).** For every labeling $U$ and every reframing
# > $h:V\to G$, $S(\text{reframed }U)=S(U)$.
#
# Let's check it on a $4\times4$ grid with four groups: finite and continuous, abelian and
# non-abelian.

# %%
B = Board.grid(4, 4)
for G in [Z2(), U1(), S3(), SO3()]:
    M = Model(B, G)
    U, h = random_labeling(G, B, rng), random_reframing(G, B, rng)
    print(f"{str(G):>4}:  S(U) = {M.cost(U):8.4f}   S(reframed U) = {M.cost(reframe(G, B, U, h)):8.4f}")

# %% [markdown]
# Here is one random coin labeling on that grid. Gold edges carry $-1$, coral faces are
# defects. Count the coral faces and double the count: that's $S(U)$.

# %%
G, B = Z2(), Board.grid(4, 4)
U = random_labeling(G, B, rng)
fig, ax = plt.subplots(figsize=(4, 4))
viz.draw_board(ax, B, U, G, title=f"S(U) = {Model(B, G).cost(U):.0f}  "
               f"({defects(G, B, U).sum()} defects)")
plt.show()

# %% [markdown]
# > **Recap.** A face cost $\varphi:G\to\mathbb R$ prices a single face. It is a class
# > function, minimal at the identity. The total cost $S:G^E\to\mathbb R_{\ge0}$ adds up the
# > prices of all faces. Because $\varphi$ ignores conjugation, $S$ is an invariant: reframings
# > don't change it.
#
# **Check yourself**
#
# 1. A coin-group labeling on the $4\times4$ grid has 3 defects. What is $S(U)$?
#
# <details><summary>Answer</summary>
#
# $S(U)=3\cdot\varphi(-1)=6$.
#
# </details>
#
# 2. A face is a closed walk, but which vertex does it start at? Explain why this choice
#    cannot affect $\varphi(U_f)$.
#
# <details><summary>Answer</summary>
#
# Starting the same loop at a different vertex turns the product $abcd$ into, for example,
# $bcda=a^{-1}(abcd)a$. That is a conjugate, and a class function gives conjugates the same
# value.
#
# </details>
#
# 3. On the circle, for which face products is the face cost zero?
#
# <details><summary>Answer</summary>
#
# Only for $\theta=0$, the identity. $1-\cos\theta>0$ for every other angle.
#
# </details>

# %% [markdown]
# ---
# ## 3. Stiffness and the model
#
# One more ingredient: a single number that says how *seriously* to take the cost.
#
# **Definition (stiffness, glossary 4.4).** The stiffness $\beta\in\mathbb R_{\ge0}$ is a
# number. Large $\beta$ means defects are heavily punished; $\beta=0$ means cost is ignored.
#
# **Definition (model, glossary 4.5).** Let $\sigma^E$ be "every edge label uniform and
# independent", a probability distribution on $G^E$. The *model* with board $X$, group $G$,
# face cost $\varphi$ and stiffness $\beta$ is the probability distribution $\mu_\beta$ on
# $G^E$ given by
#
# $$
# \mu_\beta(dU)=\frac{1}{Z(\beta)} e^{-\beta S(U)} \sigma^E(dU),\qquad
# Z(\beta)=\int e^{-\beta S(U)} \sigma^E(dU)\ \in\mathbb R_{>0}.
# $$
#
# For a finite group, the integral is just an average over the $|G|^{|E|}$ labelings, and
# this reads
#
# $$
# \mu_\beta(U)=\frac{e^{-\beta S(U)}}{\sum_{U'} e^{-\beta S(U')}}.
# $$
#
# $Z(\beta)$ is just the normalizing number that makes the probabilities add up to 1.
#
# **The stance of this course.** The model $\mu_\beta$ *is* the object we study. The cost is
# not extra information on top of it. Up to the constant $\log Z$, the scaled cost $\beta S$ is
# just minus the logarithm of the density of $\mu_\beta$ relative to uniform:
#
# $$
# -\log\frac{\mu_\beta(U)}{\sigma^E(U)}=\beta S(U)+\log Z(\beta).
# $$
#
# So a model is completely specified by four things: (board, $G$, $\varphi$, $\beta$).
#
# For finite groups, `lgt` computes $\mu_\beta$ *exactly*, by listing every labeling. Let's
# do that for the coin model on the $2\times2$ torus from the opening puzzle: $2^8=256$
# labelings.

# %%
T = Board.grid(2, 2, periodic=True)
beta = 1.0
ex = Model(T, Z2(), beta=beta).exact()          # every labeling with its probability
k = (ex.cost / 2).astype(int)                   # number of defects of each labeling
for kk in np.unique(k):
    sel = k == kk
    print(f"{kk} defects: {sel.sum():3d} labelings, each with probability "
          f"{ex.p[sel][0]:.5f}  (together {ex.p[sel].sum():.3f})")
print(f"\nZ(beta={beta}) = {np.exp(ex.log_Z):.5f}")
print("cost = log-density?", np.allclose(-np.log(ex.p * 256), beta * ex.cost + ex.log_Z))

# %% [markdown]
# Three things to notice:
# - A torus never has exactly 1 or 3 defects. Every edge lies on exactly two faces, so the
#   product of all face products is always $+1$. We'll come back to this in section 5.
# - Each individual 4-defect labeling is $e^{-8}\approx 3\times10^{-4}$ times as likely as a
#   0-defect labeling.
# - The last line confirms the stance: $-\log$ of the density *is* $\beta S+\log Z$.
#
# And the opening puzzle: with fair coins ($\beta=0$), each face is a defect with probability
# $\tfrac12$, so you expect $2$ defects out of 4.
#
# > **Recap.** A model $\mu_\beta$ is the uniform distribution on labelings, reweighted by
# > $e^{-\beta S}$ and renormalized by $Z(\beta)$. It is determined by (board, $G$,
# > $\varphi$, $\beta$). Up to sign, scale and a constant, the cost is the log-density relative
# > to uniform, so the model is the single object that holds everything.
#
# **Check yourself**
#
# 1. In the torus model at $\beta=1$, how much less likely is a particular labeling with 2
#    defects than a particular labeling with none?
#
# <details><summary>Answer</summary>
#
# The costs are $4$ and $0$, so the ratio is $e^{-4}\approx0.018$. The normalizer $Z$ cancels
# in ratios.
#
# </details>
#
# 2. Someone hands you $\mu_\beta$ as a table of probabilities but forgets to tell you $S$.
#    Can you recover $\beta S$?
#
# <details><summary>Answer</summary>
#
# Yes, up to an additive constant: $\beta S(U)=-\log\big(\mu_\beta(U)/\sigma^E(U)\big)-\log Z$.
# The constant is pinned down by the identity labeling, whose cost is 0.
#
# </details>
#
# 3. What happens to the model if you double $\beta$ and halve $\varphi$?
#
# <details><summary>Answer</summary>
#
# Nothing. Only the product $\beta\varphi$ enters $\mu_\beta$. Stiffness is the convention for
# "how sharp" the cost is, measured against a fixed $\varphi$.
#
# </details>

# %% [markdown]
# ---
# ## 4. Redundancy does no harm
#
# In chapter 1 we saw that a labeling carries both real information (its class) and
# convention (a reframing). Does that convention distort the model? No, for three reasons.
#
# > **Fact 2.4 (the model doesn't see reframings).** For every reframing $h:V\to G$ and
# > every labeling $U$: $\mu_\beta(\text{reframed }U)=\mu_\beta(U)$. Equivalently, reframing a
# > random labeling drawn from $\mu_\beta$ gives a random labeling with the *same*
# > distribution $\mu_\beta$.
#
# *Why:* a reframing permutes the labelings, it leaves $\sigma^E$ unchanged, and it leaves $S$
# unchanged (Fact 2.3). Below we reframe all 256 labelings at once, see where each one lands, and
# compare probabilities.

# %%
def index_of(U):
    """Position of a coin labeling in the enumeration order of `Model.exact()`."""
    bits = (np.asarray(U) == -1).astype(np.int64)
    return bits @ (2 ** np.arange(bits.shape[-1])[::-1])

assert np.all(index_of(ex.U) == np.arange(len(ex.U)))
h = random_reframing(Z2(), T, rng)
moved = index_of(reframe(Z2(), T, ex.U, h))           # where each labeling goes
print("reframing permutes the labelings:", len(np.unique(moved)) == len(moved))
print("and preserves every probability: ", np.allclose(ex.p[moved], ex.p))

# %% [markdown]
# Next, how big are the classes? For an abelian group they all have the same size.
#
# > **Fact 2.5 (class sizes).** If $G$ is finite and abelian (recall that boards are
# > connected), every class contains exactly $|G|^{|V|-1}$ labelings, and there are
# > $|G|^{|E|-|V|+1}$ classes. Hence the probability of a class is proportional to
# > $e^{-\beta S}$ of any of its members.
# >
# > For non-abelian $G$, classes can have different sizes.
#
# *Why the exponent is $|V|-1$:* there are $|G|^{|V|}$ reframings. When $G$ is abelian, a
# reframing leaves a labeling unchanged exactly when $h(s)=h(t)$ on every edge, that is, when
# $h\equiv g$ is constant. So each labeling in a class is reached by exactly $|G|$ reframings,
# and the class has $|G|^{|V|}/|G|$ members.

# %%
sizes = np.bincount(class_ids(Z2(), T, ex.U))
print(f"coin on the torus: {len(sizes)} classes, sizes {set(sizes.tolist())}"
      f"  (formula: 2^{T.cycle_rank} classes of size 2^{T.n_vertices - 1})")

S, tri = S3(), Board.triangle()
Us = all_labelings(S, tri)
ids = class_ids(S, tri, Us)
loop = loop_product(S, tri, Us, [0, 1, 2, 0])
for c in range(ids.max() + 1):
    print(f"S3 on the triangle, class {c}: {np.sum(ids == c):3d} labelings, "
          f"loop product like {viz.format_label(S, loop[ids == c][0])}")

# %% [markdown]
# On the triangle with $S_3$, a class is determined by the conjugacy class of its loop product
# (chapter 1). The 216 labelings split $36/108/72$, because identity, transpositions and
# 3-cycles make up $1/6$, $3/6$ and $2/6$ of $S_3$. So, unlike the abelian case, the class
# sizes differ. That's fine: the size is simply how much uniform probability the class
# carries.
#
# The most useful form of "redundancy does no harm" is the next fact. It lets us throw the
# redundancy away *before* computing. It is about averages: for a function $f:G^E\to\mathbb R$,
# we write $\langle f\rangle_\beta\in\mathbb R$ for its expectation when $U$ is drawn from
# $\mu_\beta$ (glossary 4.6). Chapter 3 is all about averages; here we only need the notation.
#
# > **Fact 2.6 (fixing the tree is free).** Pick a spanning tree. For any invariant
# > $f:G^E\to\mathbb R$, the average $\langle f\rangle_\beta$ equals the average of $f$ in
# > the smaller model where every tree edge is fixed to the identity and only the
# > $|E|-|V|+1$ other labels are random (with the same weight $e^{-\beta S}$).
#
# *Why:* putting a uniformly random labeling into tree normal form leaves the non-tree labels
# uniform and independent. Each of them is a walk product that contains its own edge's
# uniform label exactly once. Invariants and $S$ don't change.
#
# Let's verify this with two invariants on the torus (the number of defects, and the face
# product of one face) and one on the $S_3$ triangle ($\chi$ of the loop product, where $\chi$
# is the trace of the permutation matrix).

# %%
def tree_fixed_average(model, f):
    """Average of f when tree labels are pinned to the identity (Fact 2.6)."""
    tree = model.board.spanning_tree()[0]
    ex = model.exact()
    keep = np.all(model.G.is_identity(ex.U[:, tree]), axis=1)
    p = ex.p[keep] / ex.p[keep].sum()
    return float(np.sum(p * f(ex.U[keep]))), int(keep.sum())

M = Model(T, Z2(), beta=0.7)
for name, f in [("number of defects", lambda U: defects(Z2(), T, U).sum(-1)),
                ("product of face 0", lambda U: face_products(Z2(), T, U)[:, 0])]:
    small, n = tree_fixed_average(M, f)
    print(f"{name:>25}: full model {M.average(f):+.5f} | tree fixed {small:+.5f} ({n} labelings)")

Mt = Model(tri, S, beta=0.7)
chi = lambda U: S.chi(loop_product(S, tri, U, [0, 1, 2, 0]))
print(f"{'S3 triangle, chi(loop)':>25}: full model {Mt.average(chi):+.5f} | "
      f"tree fixed {tree_fixed_average(Mt, chi)[0]:+.5f}")

# %% [markdown]
# Same numbers, from $2^5=32$ instead of $256$ labelings on the torus, and from 6 instead
# of 216 on the triangle. On big boards this saving is enormous.
#
# > **Recap.** Reframings permute labelings without changing their probability (Fact 2.4).
# > For abelian groups all classes have the same size, so the redundancy is a constant
# > overcount that cancels (Fact 2.5). For any group, you may pin a spanning tree to the
# > identity before computing averages of invariants (Fact 2.6).
#
# **Check yourself**
#
# 1. Coin model on the $3\times3$ torus (9 vertices, 18 edges). How many classes, and how
#    big is each?
#
# <details><summary>Answer</summary>
#
# $2^{18-9+1}=2^{10}=1024$ classes, each with $2^{9-1}=256$ labelings. Check:
# $1024\cdot256=2^{18}$.
#
# </details>
#
# 2. With $S_3$ on the triangle, the class of the identity labeling has only 36 members.
#    Which reframings leave the identity labeling unchanged?
#
# <details><summary>Answer</summary>
#
# The six constant reframings $h\equiv g$: they give $g e g^{-1}=e$ on every edge. Each
# labeling in the class is therefore produced by 6 different reframings, and the $6^3$
# reframings give only $6^3/6=36$ different labelings. A labeling whose loop product is a
# 3-cycle is fixed by only 3 reframings, so its class has $6^3/3=72$ members.
#
# </details>
#
# 3. Fact 2.6 needs $f$ to be an invariant. Find a non-invariant $f$ for which it fails.
#
# <details><summary>Answer</summary>
#
# In the coin model, take $f(U)=U(e)$ for a *tree* edge $e$. In the tree-fixed model it is
# always $+1$, so its average there is $1$. In the full model its average is $0$, as chapter 3
# explains.
#
# </details>

# %% [markdown]
# ---
# ## 5. The stiffness dial
#
# What does turning $\beta$ do? Look at the two ends first.
#
# > **Fact 2.7 (the two ends of the dial).**
# > - At $\beta=0$, $\mu_0=\sigma^E$: all labels are independent and uniform.
# > - As $\beta\to\infty$, $\mu_\beta$ piles up on the labelings with **no defects** (cost 0),
# >   spread uniformly over them.
#
# On a torus, "no defects" is a *weaker* condition than "consistent" (glossary 2.6). A loop
# that goes once around the torus is not a product of faces, so its loop product can still
# be $-1$ when every face product is $+1$. On the $2\times2$ torus there are 32 defect-free
# labelings, but only 8 consistent ones.

# %%
for beta in [0.0, 1.0, 3.0, 6.0]:
    ex_b = Model(T, Z2(), beta=beta).exact()
    k = (ex_b.cost / 2).astype(int)
    ok = is_consistent(Z2(), T, ex_b.U)
    print(f"beta = {beta:3.1f}:  P(no defects) = {ex_b.p[k == 0].sum():.4f}   "
          f"P(consistent) = {ex_b.p[ok].sum():.4f}")
print("defect-free labelings:", int(np.sum(k == 0)), "| consistent:", int(ok.sum()))

# %% [markdown]
# At large $\beta$, $P(\text{consistent})$ settles at $8/32=\tfrac14$: the model spreads evenly
# over the 32 defect-free labelings, and only 8 of them are consistent.
#
# Now the whole dial, on the $3\times3$ torus ($2^{18}=262{,}144$ labelings, still exact).
# The weight of a labeling depends only on its number of defects $k$. So we count the
# labelings with each $k$ once, at $\beta=0$, and reweight:
# $P_\beta(k)\propto N_k e^{-2\beta k}$, where $N_k$ is that count.
#
# What to look at:
# - **Left:** at $\beta=0$ the defect count is a binomial-like hump around half the faces
#   (only even counts are possible, as before). As $\beta$ grows the hump slides to 0.
# - **Right:** the average fraction of defect faces drops smoothly. The dashed curve is a
#   formula for *planar* grids (not wrapped around), coming next.

# %%
T3 = Board.grid(3, 3, periodic=True)
ex3 = Model(T3, Z2(), beta=0.0).exact()
N_k = np.bincount((ex3.cost / 2).astype(int))           # labelings with k defects
kk = np.arange(len(N_k))
def P_defects(beta):
    w = N_k * np.exp(-2 * beta * kk)
    return w / w.sum()

betas = np.linspace(0, 2.5, 101)
mean_frac = [P_defects(b) @ kk / T3.n_faces for b in betas]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
for i, b in enumerate([0, 0.3, 0.6, 1.0, 2.0]):
    a1.bar(kk + (i - 2) * 0.16, P_defects(b), width=0.16, label=fr"$\beta={b}$")
a1.set(xlabel="number of defects $k$ (of 9 faces)", ylabel="probability", xticks=kk)
a1.legend(frameon=False, fontsize=9)
a2.plot(betas, mean_frac, lw=2.5, label=r"$3\times3$ torus (exact)")
a2.plot(betas, np.exp(-2 * betas) / (1 + np.exp(-2 * betas)), "--", color=viz.INK,
        label=r"planar grid: $e^{-2\beta}/(1+e^{-2\beta})$")
a2.set(xlabel=r"stiffness $\beta$", ylabel="average fraction of defect faces")
a2.legend(frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# ### On a planar grid, the faces are independent coins
#
# The dashed curve comes from a fact that makes the coin model on a planar grid almost
# embarrassingly simple. Chapter 3 builds on it.
#
# > **Fact 2.8 (independent faces).** In the coin model ($G=\mathbb Z_2$, standard face cost)
# > on a planar grid (not wrapped around), the face products $U_f\in\lbrace\pm1\rbrace$, one
# > per face $f$, are **independent**. Each is a defect with probability
# > $P(U_f=-1)=e^{-2\beta}/(1+e^{-2\beta})$; equivalently, $\langle U_f\rangle_\beta=\tanh\beta$.
#
# *Why, lightly:* put the labeling in tree normal form (Fact 2.6). On a planar grid the number
# of non-tree edges equals the number of faces, since $|E|-|V|+1=(L-1)^2$ for an $L\times L$
# grid. The non-tree labels and the face products determine each other one-to-one. So the
# face products are independent fair coins under $\sigma^E$. The weight
# $e^{-\beta S}=\prod_f e^{-\beta\varphi(U_f)}$ then tilts each coin separately.
#
# Let's check it exactly on the $3\times3$ grid (12 edges, 4 faces). We compare all 16 joint
# probabilities of the four face products with the product of the four marginals.

# %%
P3 = Board.grid(3, 3)
beta = 0.6
exP = Model(P3, Z2(), beta=beta).exact()
FP = face_products(Z2(), P3, exP.U)                      # (4096, 4): one row per labeling
patterns = np.array(np.meshgrid(*[[1, -1]] * 4, indexing="ij")).reshape(4, -1).T
marg = (1 + np.tanh(beta) * patterns) / 2                # P(U_f = s) = (1 + s tanh beta)/2
joint = np.array([exP.p[np.all(FP == s, axis=1)].sum() for s in patterns])
print("largest |joint - product of marginals| over 16 patterns:",
      f"{np.abs(joint - marg.prod(axis=1)).max():.1e}")
print(f"<U_f> = {exP.average(lambda U: face_products(Z2(), P3, U)[:, 0]):.5f}"
      f"   tanh(beta) = {np.tanh(beta):.5f}")

# %% [markdown]
# The independence fails on a torus, and we already know why: there the product of *all*
# face products is forced to be $+1$.

# %%
FP3 = face_products(Z2(), T3, ex3.U)
print("3x3 torus: product of all face products is +1 for every labeling:",
      bool(np.all(FP3.prod(axis=1) == 1)))

# %% [markdown]
# ### Seeing the dial
#
# Fact 2.8 hands us an *exact* sampler for large planar grids, far too large to enumerate. Flip
# one tilted coin per face. Then build any labeling with those face products: keep vertical
# edges at $+1$ and fill in the horizontal edges row by row. Finally apply a random reframing
# so that the labeling doesn't look special. (By Fact 2.4 the result is an exact draw from
# $\mu_\beta$.)
#
# What to look at:
# - **The faces:** the coral defects thin out as $\beta$ grows.
# - **The edges:** the gold $-1$ edges look like the same random noise at *every* $\beta$.
#
# Edge labels are mostly convention. The information lives on the faces.

# %%
def sample_coin_planar(L, beta, rng):
    """Exact draw from the coin model on an L x L planar grid (Fact 2.8 + Fact 2.4)."""
    B, G = Board.grid(L, L), Z2()
    idx = B.meta["edge_index"]                           # (vertex, direction) -> edge
    p = np.exp(-2 * beta) / (1 + np.exp(-2 * beta))
    face = np.where(rng.random((L - 1, L - 1)) < p, -1, 1)
    U = G.identity(B.n_edges)                            # vertical edges stay +1
    for y in range(L - 1):
        for x in range(L - 1):                           # face (x,y) = bottom edge * top edge
            U[idx[(x + L * (y + 1), 0)]] = U[idx[(x + L * y, 0)]] * face[x, y]
    return B, reframe(G, B, U, random_reframing(G, B, rng))

# %%
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, beta in zip(axs, [0.0, 0.5, 1.0, 2.0]):
    B, U = sample_coin_planar(10, beta, rng)
    viz.draw_board(ax, B, U, Z2(), show_arrows=False, vertex_size=8, pad=0.2,
                   title=fr"$\beta={beta}$:  {defects(Z2(), B, U).sum()} defects of {B.n_faces}")
plt.show()

# %% [markdown]
# The same dial works for any compact group. Here is the circle on a $6\times6$ grid. The
# samples come from a simple random-update sampler, which we'll explain in chapter 4. Clock
# hands show the edge labels, and face colour shows the face product as an angle: pale means
# near the identity, dark means far from it. Again the clock hands look random at every
# stiffness, while the faces calm down.

# %%
G, B = U1(), Board.grid(6, 6)
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, beta in zip(axs, [0.0, 1.0, 3.0, 10.0]):
    step = min(1.0, 1.5 / np.sqrt(beta + 1e-9))          # smaller moves when stiff
    U = Model(B, G, beta=beta).metropolis(rng, n_samples=1, burn_in=400, step=step)[0]
    viz.draw_board(ax, B, U, G, show_arrows=False, vertex_size=8, clock_r=0.13,
                   title=fr"$\beta={beta}$")
plt.show()

# %% [markdown]
# > **Recap.** At $\beta=0$ the model is uniform, and as $\beta\to\infty$ it piles up on the
# > defect-free labelings. On a torus those include non-consistent ones, whose loop product
# > once around the torus is not the identity. On a planar grid with the coin group, the
# > faces are independent coins with $\langle U_f\rangle_\beta=\tanh\beta$ (Fact 2.8). Pictures of samples show the lesson of
# > this chapter: edges look like noise, and faces carry the information.
#
# **Check yourself**
#
# 1. In the planar-grid coin model, what is the probability that a given face is a defect at
#    $\beta=0$? At $\beta=\tfrac12\ln 3$?
#
# <details><summary>Answer</summary>
#
# At $\beta=0$ it is $\tfrac12$. At $\beta=\tfrac12\ln3$, $e^{-2\beta}=\tfrac13$, so the
# probability is $\frac{1/3}{1+1/3}=\tfrac14$.
#
# </details>
#
# 2. Why can a torus never have exactly one defect?
#
# <details><summary>Answer</summary>
#
# Every edge lies on exactly two faces. In the product of all face products, each label
# appears twice, and $(\pm1)^2=1$. So the product of all face products is $+1$, which means
# the number of defects is even.
#
# </details>
#
# 3. The $10\times10$ grid in the dial picture has $180$ edges. How many labelings is that,
#    and why could we still sample it exactly?
#
# <details><summary>Answer</summary>
#
# $2^{180}\approx1.5\times10^{54}$ labelings, far beyond enumeration. Fact 2.8 reduces the
# problem to 81 independent tilted coins, one per face. Fact 2.4 lets us reframe the result
# freely.
#
# </details>

# %% [markdown]
# ---
# ## Chapter recap
#
# - A **uniform distribution** on $G$ is unchanged by multiplying by a fixed element. It
#   exists exactly for compact $G$ (finite groups, circle, rotations), not for
#   $\mathbb R_{>0}$. A uniform rotation's angle has density $(1-\cos\omega)/\pi$.
# - A **face cost** $\varphi:G\to\mathbb R$ is a class function, minimal at the identity. The
#   **total cost** $S(U)=\sum_f\varphi(U_f)$ is an invariant.
# - The **model** $\mu_\beta\propto e^{-\beta S} \sigma^E$ is determined by (board, $G$,
#   $\varphi$, $\beta$). The cost is its log-density relative to uniform, so the model is the
#   one object that holds everything.
# - **Redundancy does no harm:** reframings preserve $\mu_\beta$. Abelian classes all have size
#   $|G|^{|V|-1}$. You may pin a spanning tree to the identity before averaging invariants.
# - **The stiffness dial** runs from uniform labels ($\beta=0$) to defect-free labelings
#   ($\beta\to\infty$).
# - On a **planar grid with the coin group**, faces are independent coins with
#   $\langle U_f\rangle_\beta=\tanh\beta$.
#
# ## Dictionary
#
# | Our term | Other names |
# |---|---|
# | uniform distribution on $G$ | Haar measure |
# | face cost $\varphi$ | Wilson action (per plaquette) |
# | total cost $S$ | action, Euclidean action; energy (statistical mechanics) |
# | stiffness $\beta$ | inverse coupling $1/g^2$ (up to a constant); inverse temperature |
# | model $\mu_\beta$ | lattice gauge theory; "the theory"; Euclidean path integral; Gibbs measure |
# | normalizer $Z(\beta)$ | partition function |
# | average $\langle f\rangle_\beta$ | expectation value; vacuum expectation value |
# | coin model | $\mathbb Z_2$ lattice gauge theory (Wegner's Ising gauge model) |
# | fixing the tree (Fact 2.6) | gauge fixing (maximal-tree / axial gauge) |
#
# ## What's next
#
# A model is a distribution, but we can only ever *ask* it questions: "on average, what is
# this number?" Chapter 3 is about which questions make sense. Averages of non-invariant
# quantities, like a single edge label, turn out to be useless, and you'll see exactly why.
# Averages of loop products are where the structure shows up. On a planar grid, Fact 2.8 already
# tells you how the average loop product of a big rectangle must behave. Try to guess before
# you turn the page.
