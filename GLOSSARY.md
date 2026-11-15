# Glossary (draft v2, not yet agreed)

The course is standalone. We assume only:
- sets and functions;
- linear algebra (matrices, eigenvalues);
- graphs (vertices, edges, walks, cycles, spanning trees);
- probability (distributions, expectation, Markov chains);
- groups (homomorphisms, conjugation $x\mapsto hxh^{-1}$).

Everything else is defined below before it is used. Each term lists its "also called" names, so the reader can find it in the literature later.

**Running examples**
- **Currency market:** $G=(\mathbb R_{>0},\times)$.
- **Cameras:** $G=SO(3)$, the rotations of 3D space. This group is non-abelian.
- **Coin model:** $G=\lbrace +1,-1 \rbrace$ under multiplication.

---

## 1. The board

**1.1 Board** $X$. A finite connected graph together with a chosen list of closed walks called *faces*.
- Example: an $L\times L$ grid whose faces are the unit squares.
- *Also called:* lattice, 2-complex, "spacetime" (in physics).

**1.2 Direction.** Each edge $\lbrace s,t \rbrace$ can be walked two ways, $s\to t$ and $t\to s$.
- $\vec E$ is the set of all directions. It has $2|E|$ elements.
- For a direction $d$, $\bar d$ is the same edge walked the other way.

**1.3 Face.** A closed walk that we designate as *elementary*. Costs (§4) are measured only on faces.
- Choosing the faces is a modeling decision, like choosing the neighborhood in an image model.
- On a grid, a face is the 4-step walk around a unit square.
- *Also called:* plaquette (physics), 2-cell (topology).

## 2. Labelings

**2.1 Label group** $G$. The group the labels come from.
- *Also called:* gauge group (physics; ⚠ mathematicians use "gauge group" for 3.2 instead), gain group (gain graphs), structure group (geometry).

**2.2 Labeling** $U:\vec E\to G$, subject to $U(\bar d)=U(d)^{-1}$.
- **How to read it:** $U(s\to t)$ converts the conventions at $s$ into the conventions at $t$. Walking the edge backwards must undo the conversion.
- **Currency:** $U(\text{USD}\to\text{ILS})=3.7$ means one dollar buys 3.7 shekels. Then $U(\text{ILS}\to\text{USD})=1/3.7$ (assuming no bid/ask spread).
- **Counting:** you choose one label per edge, and the reverse is then forced. So the set of labelings is $G^E$.
- *Also called:* configuration, gauge field, link variables (physics); gain function (gain graphs); connection (geometry, synchronization).

**2.3 Walk product.** For a walk $d_1d_2\cdots d_k$, the walk product is $U(d_1)U(d_2)\cdots U(d_k)\in G$.
- **Currency:** it is what one unit at the start of the walk becomes at the end.
- For non-abelian $G$, the order of the factors matters.
- *Also called:* parallel transport, Wilson line (physics); gain of a walk (gain graphs); net voltage (voltage graphs).

**2.4 Loop product at $v$.** The walk product of a closed walk that starts and ends at $v$.
- **Currency:** a loop product of $1.02$ is a 2% arbitrage profit.
- *Also called:* holonomy, monodromy.

**2.5 Face product** $U_f$. The loop product of a face.
- A face with $U_f\neq 1$ is a **defect**.
- *Also called:* plaquette variable (physics), curvature or field strength (geometry), frustrated plaquette or vortex (spin glasses).

**2.6 Consistent labeling.** A labeling in which every loop product equals the identity.
- **Currency:** no arbitrage exists.
- **Cameras:** all the pairwise relative rotations agree with one global orientation.
- *Also called:* flat, pure gauge (physics); balanced (gain graphs); unfrustrated, Mattis (spin glasses); cycle-consistent (vision).

## 3. Redundancy

**3.1 Reframing** $h:V\to G$. It acts on a labeling by

$$
U(s\to t)\quad\longmapsto\quad h(s) U(s\to t) h(t)^{-1}.
$$

- **Meaning:** each vertex changes its own convention, and every edge label is a conversion between its two endpoints, so both endpoint changes enter.
- **Currency:** take $h(v)=\lambda_v$, where the new unit at $v$ equals $\lambda_v$ old units. One new $s$-unit is $\lambda_s$ old $s$-units. That buys $\lambda_s U$ old $t$-units, which is $\lambda_sU/\lambda_t$ new $t$-units.
  - Real instance: Israel's 1985 new shekel replaced 1000 old shekels.
- **Why the formula has this shape:** along a walk the inner factors cancel,
  $h(s)U_1h(v)^{-1}\cdot h(v)U_2h(t)^{-1}=h(s) U_1U_2 h(t)^{-1}$.
  So only the two endpoints of a walk feel a reframing.
- *Also called:* gauge transformation (physics); switching (gain graphs); change of units or numeraire (finance); inserting $XX^{-1}$ on a bond (tensor networks); neuron rescaling (neural nets); Nishimori/Mattis gauge (spin glasses).

**3.2 Reframing group** $G^V$. The set of all reframings, multiplied vertex by vertex.
- *Also called:* gauge group (mathematics).

**3.3 Class** $[U]$. All the labelings that can be reached from $U$ by some reframing.
- *Also called:* gauge orbit, physical configuration (physics); switching class (gain graphs).

**3.4 Invariant.** A function $f:G^E\to\mathbb C$ such that $f(\text{reframed }U)=f(U)$ for every reframing.
- Equivalently, $f$ is a function of the class.
- *Also called:* (gauge-invariant) observable.

**3.5 Fact.** Under a reframing, the loop product at $v$ becomes $h(v) w h(v)^{-1}$, where $w$ is the old loop product.
- For abelian $G$, loop products are therefore invariants.
- For non-abelian $G$, we need 3.6 to turn them into invariants.

**3.6 Class function.** A function $\chi:G\to\mathbb C$ with $\chi(hxh^{-1})=\chi(x)$ for all $h,x$.
- Example: the trace of a matrix.

**3.7 Loop observable.** $W_\gamma=\chi(\text{loop product of }\gamma)$.
- *Also called:* Wilson loop.

**3.8 Tree normal form.** Fix a spanning tree. Every class contains a labeling that is the identity on every tree edge.
- The labels on the remaining $|E|-|V|+1$ edges are then free, one per independent cycle.
- *Also called:* gauge fixing, maximal-tree or axial gauge (physics); spanning-forest normalization (gain graphs); canonical form (tensor networks).

**3.9 Charged field** $\psi:V\to\mathbb C$ (for $G=U(1)$; in general, a vector at each vertex that $G$ acts on). A value at every *vertex*, expressed in that vertex's convention.
- A reframing changes it as $\psi(v)\mapsto h(v)^{-1}\psi(v)$.
- **Currency:** an amount of money held in each currency, measured in that currency's units.
- *Also called:* matter field, wavefunction of a charged particle (physics); section of a bundle (geometry).

**3.10 Covariant difference.** $D_U\psi(s\to t)=\psi(s) U(s\to t)-\psi(t)$: convert the value at $s$ into $t$'s convention, then compare.
- Under a joint reframing of $U$ and $\psi$, it is multiplied by $h(t)^{-1}$, so its size is an invariant.
- The naive difference $\psi(t)-\psi(s)$ is not invariant: comparing values at two vertices *requires* a label on the edge between them.
- *Also called:* covariant derivative; minimal coupling (physics). The idea that local reframings force labels is the *gauge principle*.

**3.11 Covariant Laplacian** $L_U$. The $|V|\times|V|$ Hermitian matrix with $\psi^{*}L_U\psi=\sum_e|D_U\psi(e)|^2$, the *smoothness cost* of a charged field.
- Its spectrum depends only on the class $[U]$.
- Its smallest eigenvalue is 0 exactly when $U$ is consistent (connected board).
- *Also called:* magnetic Laplacian; connection Laplacian (synchronization); tight-binding Hamiltonian with Peierls phases (physics).

## 4. Randomness: the model

**4.1 Uniform distribution on $G$**, written $\sigma$. The unique probability distribution that doesn't change when you multiply by any fixed element of $G$.
- **It exists exactly when $G$ is compact.**
- Examples: for finite $G$, each element gets probability $1/|G|$. For the circle, the angle is uniform.
- *Also called:* Haar measure.

**4.2 Face cost.** A class function $\varphi:G\to\mathbb R$ that is minimized at the identity.
- Coin model: $\varphi(+1)=0$ and $\varphi(-1)=2$.
- Circle: $\varphi=1-\cos\theta$.
- Matrix groups: $\varphi(x)=\operatorname{Re}\operatorname{tr}(I-x)$.

**4.3 Total cost.** $S(U)=\sum_{f}\varphi(U_f)$, summed over all faces. It is an invariant.
- *Also called:* action (physics); energy or Hamiltonian (statistical mechanics). We avoid "Hamiltonian" here, because 5.8 reserves it.

**4.4 Stiffness** $\beta\ge 0$. Large $\beta$ makes defects rare.
- *Also called:* inverse coupling, since $\beta\propto1/g^2$ (for $SU(2)$ with the standard face cost, $\beta=2/g^2$); inverse temperature.

**4.5 Model.** The probability distribution

$$
\mu(dU)=\frac{1}{Z} e^{-\beta S(U)} \sigma^{E}(dU),
$$

where $\sigma^E$ means "each edge label uniform and independent."
- A model is fully specified by (board, $G$, $\varphi$, $\beta$).
- $Z$ is the normalizing constant. *Also called:* partition function.
- *Also called:* lattice gauge theory, "the theory", Euclidean path integral, Gibbs measure.

**4.6 Average** $\langle f\rangle$. The expectation of $f$ under the model.
- *Also called:* expectation value, vacuum expectation value.

**4.7 Correlation and correlation length.** The correlation of $f$ and $g$ is $\langle fg\rangle-\langle f\rangle\langle g\rangle$.
- The **correlation length** $\xi$: correlations decay like $e^{-\text{distance}/\xi}$.

**4.8 Face remainder** $r_f(e)\in G$. For a face $f$ that uses edge $e$: walk $f$ starting with $e$ in its stored direction. The face product is then $U(e) r_f(e)$, and $r_f(e)$ is the product of the face's other labels.
- The cost of a single edge, with everything else held fixed, depends only on the remainders of the faces through it.
- This locality is what makes sampling cheap (chapter 4).
- *Also called:* staple (physics).

## 5. Layering, and the dynamics derived from it (chapter 5 onward)

**5.1 Symmetry of a board.** A bijection of the vertices that maps edges to edges and faces to faces.
- Because the total cost depends only on faces, any such symmetry is automatically a symmetry of every model on that board.

**5.2 Layered board.** The vertices are split into layers $V_0,\dots,V_{N-1}$, with indices taken mod $N$.
- Every edge is either a *layer edge* (inside one layer) or a *step edge* (between two consecutive layers).
- Example: a $W\times N$ grid wrapped top to bottom, where the rows are the layers.
- The faces between layers $k$ and $k+1$ form **slab** $k$. Every face lies in exactly one slab.
- *Also called:* time slices; spatial and temporal links.

**5.3 Shift** $\tau$. The symmetry that moves layer $k$ to layer $k+1$.
- **Why we need it:** every step then looks the same, so a single matrix describes every step, as in a time-homogeneous Markov chain.
- *Also called:* time translation.

**5.4 Flip** $\theta$. The symmetry that reverses the order of the layers around a cut ($k\mapsto -k$ or $k\mapsto 1-k$).
- **Why we need it:** it pairs "past" with "future," which produces an inner product.
- *Also called:* time reflection.

**5.5 Layer labeling.** The restriction of $U$ to the layer edges of a single layer.

**5.6 Transfer matrix** $T$. Its rows and columns are indexed by layer labelings.
- $T(a,b)$ is the total weight of going from layer labeling $a$ to layer labeling $b$, summed over the labels of the step edges between them.
- Then $Z=\operatorname{tr}T^N$.

**5.7 Flip-positivity.** For every $F$ that depends only on the "upper half" of the layers, $\langle\overline{F\circ\theta} F\rangle\ge 0$.
- Consequence: $T$ is symmetric with nonnegative eigenvalues.
- It holds for all the face costs in 4.2.
- Analogy: a reversible Markov chain.
- *Also called:* reflection positivity.

**5.8 Levels and gap.** Let $\lambda_0\ge\lambda_1\ge\cdots$ be the eigenvalues of $T$, restricted as in 5.10.
- The levels are $E_n=-\log(\lambda_n/\lambda_0)$.
- The **gap** is $E_1$.
- Fact: gap $=1/\xi$.
- *Also called:* energy levels; Hamiltonian $H=-\log T$; mass gap.

**5.9 Derived system.** The pair (vector space of functions of a layer labeling, $H$).
- *Also called:* Hilbert space, quantum system.
- **The word "quantum" first appears here,** as a name for this derived object.

**5.10 Layer constraint.** $T$ commutes with reframings of a single layer, so we restrict $T$ to the reframing-invariant vectors.
- *Also called:* Gauss's law.

## 6. Forgetting (chapter 7)

**6.1 Distance between distributions** $d(p,q)\in[0,1]$. The largest difference in probability that $p$ and $q$ assign to one event; equal to $\tfrac12\sum_x|p(x)-q(x)|$.
- *Also called:* total variation distance.

**6.2 Coupling.** A pair of random variables $(X,Y)$ drawn jointly with $X\sim p$ and $Y\sim q$. Every coupling has $P(X\neq Y)\ge d(p,q)$; a **best coupling** achieves equality.
- For the coin-model heat bath, "use the same random number" is the best coupling.
- *Also called:* maximal or optimal coupling; synchronous coupling; path coupling (when built edge by edge).

**6.3 Influence** $C(e,e')\in[0,1]$. The most the heat-bath distribution at $e$ can move, in the distance 6.1, when only the label at $e'$ changes. It is zero unless $e$ and $e'$ share a face.
- *Also called:* Dobrushin interdependence matrix.

**6.4 Influence number** $\alpha=\max_e\sum_{e'}C(e,e')$.
- On a $d$-dimensional torus grid (sides $\ge3$), $\alpha\le6(d-1)\tanh(\beta\Phi/2)$, where $\Phi=\max\varphi-\min\varphi$ is the **spread** of the face cost.
- $\alpha<1$ implies forgetting (sampler), decorrelation (averages) and a gap $E_1\ge\ln(1/\alpha)$ (chapter 7).
- *Also called:* Dobrushin's constant; $\alpha<1$ is Dobrushin's uniqueness condition, a "high-temperature" or "strong-coupling" condition in physics.

**6.5 Influence distance.** The fewest steps between two sets of edges, where one step moves between two edges that share a face.

---

## Prior-art crosswalk

| Ours | Physics (lattice gauge) | Gain graphs (Zaslavsky) | Synchronization / sheaves | Elsewhere |
|---|---|---|---|---|
| labeling | gauge field | gain function | connection on a graph | exchange rates (Ilinski); bond matrices (tensor nets); weights (NNs) |
| loop product | holonomy, Wilson loop | gain of a closed walk | holonomy, cycle inconsistency | arbitrage return |
| reframing | gauge transformation | switching | the unknown frames $g_i$ | insert $XX^{-1}$ (TN); rescaling (NN); Nishimori gauge (spin glass) |
| consistent | pure gauge, flat | balanced | synchronizable, zero frustration | no-arbitrage; Mattis model |
| defect | plaquette $\ne 1$, vortex | – | – | frustrated plaquette (Toulouse) |
| tree normal form | maximal-tree gauge | switching a forest to identity | – | canonical form (TN) |
| model | lattice gauge theory | – | (inference: posterior over frames) | gauge freedom in loop calculus (BP) |

## Sources
- Chatterjee, *Yang–Mills for probabilists*, arXiv:1803.01950.
- Rybnikov & Zaslavsky, gain-graph balance and switching: https://arxiv.org/pdf/math/0209316
- Bandeira, Singer & Spielman, *A Cheeger inequality for the graph connection Laplacian*: https://arxiv.org/abs/1204.3873
- Kenyon, *Spanning forests and the vector bundle Laplacian*: https://arxiv.org/pdf/1001.4028
- Hansen & Ghrist, *Toward a spectral theory of cellular sheaves*: https://arxiv.org/abs/1808.01513
- Lévy, *Two-dimensional Markovian holonomy fields*: https://arxiv.org/pdf/0804.2230
- Jiang, Lim, Yao & Ye, *Statistical ranking and combinatorial Hodge theory*: https://arxiv.org/pdf/0811.1067
- Chertkov & Chernyak, *Loop series for discrete statistical models on graphs*: https://arxiv.org/abs/cond-mat/0603189
- Evenbly, *Gauge fixing, canonical forms and optimal truncations in tensor networks*: https://arxiv.org/pdf/1801.05390
- Ilinski, *Physics of Finance* (arbitrage as a gauge theory): https://arxiv.org/abs/hep-th/9710148
- Nguyen, *A Response to Economics as Gauge Theory* (critique of Malaney–Weinstein): https://arxiv.org/abs/2112.03460
- *Symmetry in Neural Network Parameter Spaces* (survey): https://arxiv.org/pdf/2506.13018
- Morita, Ozeki & Nishimori, *Gauge theory for quantum spin glasses*: https://arxiv.org/abs/cond-mat/0508511
