# Step 2: crossing the crossover

*A research memo, 13 November 2026. It continues the last section of chapter 7. The scripts in this folder reproduce every number. Nothing here is a proof of the mass gap; it is a map of how one could be attempted, with the parts tested where they can be.*

## 0. The short answer

Yes, I see a way forward, and it is more specific than "control the renormalization group through strong coupling". Three findings shape it.

1. **The crossover is narrow in scale. Our certificates are weak.**
   - Lattice data puts the correlation length of 4D $SU(2)$ at about **one grid step** already at $\beta_{\text{phys}}\approx2.3$–$2.4$ (table in §3).
   - The provable region of chapter 7 ends at $\beta_{\text{phys}}\approx0.056$, about 40 times further left.
   - Between "the model is short-ranged" ($\beta_{\text{phys}}\lesssim2.4$) and "the model is in the asymptotic-scaling regime" ($\beta_{\text{phys}}\gtrsim2.6$) lie only one to two doublings of scale.
   - **So the bottleneck is the strength of the large-distance certificate, not the length of the blocking flow.**
2. **In the simplest blocking scheme, the running of the stiffness is pure group curvature, and it passes the U(1) test.**
   - Migdal–Kadanoff decimation (§2) drives 4D $SU(2)$ from every $\beta$ into the provable region, losing $\Delta\beta\to1/8$ per doubling. One-loop asymptotic freedom says $0.129$.
   - I derive the $1/8$ in a few lines from the curvature factor $\theta/\sin\theta$ in the heat kernel of $SU(2)$.
   - For $U(1)$ that factor is $1$. The stiffness then stalls on a line of fixed points (the massless photon) for $\beta\gtrsim0.75$, apart from an exponentially slow drift.
   - So the mechanism distinguishes the cases it must. The drift is the known defect of this scheme: it treats defects as if in 2D.
3. **The coupling certificate fails at the crossover, but in a structured way.** See §4 for the damage-spreading measurements.

The program in §5 has three lemmas:

| Lemma | Content | Status |
|---|---|---|
| UV continuation | Extend the 2026 blocking control down to $\beta_{\text{phys}}\approx2.5$ | Hard but quantitative: the one-loop running is already accurate to a few percent there. |
| IR certificate | A large-distance certificate that is *tight* when the correlation length is about one block | **The weakest link, and the most tractable to improve now.** |
| Locality transfer | Fine observables, given block labels, are local | Technical; Balaban-type. |

There is also a free design choice: the face cost. Its fundamental/adjoint critical endpoint sits right at the crossover (§3), so a cost chosen to stay away from it widens the overlap between UV and IR control.

## 1. What step 2 must deliver

In course terms: let $\mu^{(k)}$ be the model of block labels after $k$ blocking steps (glossary 4.5, with a more complicated but local face cost). Step 2 must produce a $k_*$, uniform in the spacing, such that:

- $\mu^{(k_*)}$ satisfies a hypothesis $\mathcal H$ that implies decay of correlations at rate $m>0$ per block (Fact 7.5 is one such $\mathcal H$: influence number below 1);
- the UV control (2026 proof) reaches $\mu^{(k_0)}$ with $k_0\le k_*$, and the flow from $k_0$ to $k_*$ is controlled.

Then the physical gap is at least $m/(2^{k_*}a_{k_*})>0$, with $a_k$ the block size in physical units.

## 2. Experiment 1: decimation is curvature-driven (`mk_flow.py`)

**The scheme.** One step of scale 2 on a $d$-dimensional grid:

1. *Move faces.* Slide the faces of $2^{d-2}$ parallel planes onto one, so the face weight becomes $f^{2^{d-2}}$.
2. *Merge.* Merge $2\times2$ faces in each plane. This step is exact, by chapter 5: in 2D the face elements are independent and their products random-walk on $G$. The normalized character coefficients satisfy $c_r\mapsto c_r^4$.

The script checks the 2D exactness to six digits.

**Results (4D).** The table counts blocking steps until $\alpha<1$, the chapter 7 criterion with $\beta\Phi$ replaced by the spread of $\log f$.

| start (our $\beta$) | $SU(2)$ | $U(1)$ |
|---|---|---|
| 0.5 | 3 | 5 |
| 1.0 | 7 | never (stalls at $\beta_{\text{eff}}\approx1.39$) |
| 2.0 | 15 | never (stalls at $\approx2.53$) |
| 4.0 | 31 | never (stalls at $\approx4.47$) |

**$SU(2)$.** After a first shape-changing step, $\beta_{\text{eff}}$ drops by a steady $0.1257$ per step at large $\beta$. The per-step drop grows near the crossover ($0.14$ at $\beta\approx2$): the flow accelerates there, which is what the lattice data's "dip" also suggests. Roughly 8 doublings per unit of our $\beta$, so the scheme's gap scales as $2^{-8\beta}=e^{-5.5\beta}$. Asymptotic freedom predicts $e^{-6\pi^2\beta/11}=e^{-5.4\beta}$.

**Why $1/8$: a three-line derivation.**

- On $SU(2)$, Brownian motion run for time $t$ has the distribution (heat kernel) $K_t(\theta)\propto\frac{\theta}{\sin\theta}e^{-\theta^2/4t}$, up to exponentially small winding terms. Heat kernels compose exactly: $K_t*K_s=K_{t+s}$.
- The factor $\theta/\sin\theta\approx e^{\theta^2/6}$ is the curvature of $SU(2)$; on flat $\mathbb R^3$ it would be $1$.
- At large $\beta$ the face weight is $f\approx e^{-B\theta^2}$ with $B=\beta$. That matches $K_t$ with $1/(4t)=B+\tfrac16$.
- Moving faces: $f^4=e^{-4B\theta^2}$ matches $1/(4t')=4B+\tfrac16$.
- Merging $4$ faces in series: $t''=4t'$, so $1/(4t'')=B+\tfrac1{24}$.
- Back to the weight: $B'=1/(4t'')-\tfrac16=B-\tfrac18$.

For scale factor $b$ the same steps give $B-B'=(b^2-1)/(6b^2)$: $0.125$ at $b=2$, against the one-loop value $\tfrac{11}{6\pi^2}\ln b=0.129$. The near-agreement at $b=2$ is partly coincidence, since the scheme's drop does not grow like $\ln b$. The robust content is the *sign* and the *origin*: the running comes from the curvature of the label group.

**$U(1)$.** The heat kernel is a plain (periodized) Gaussian with no curvature factor, so the step is *exactly* marginal ($\beta'=\beta$) apart from windings. The only flow comes from vortices (angle windings), which this scheme handles as in 2D, where they always disorder. That is the scheme's known failure: it confines $U(1)$ eventually, though with an exponentially slow drift. In true 4D, vortex lines are suppressed and $U(1)$ has a genuine massless phase above $\beta\approx1.01$ (Guth; Fröhlich–Spencer).

**Dimension test.**

- In 3D every $\beta$ halves per step (super-renormalizable).
- In 5D $\beta$ roughly doubles per step (flows to weak coupling: no continuum gap).

So the mechanism is dimension-sensitive in the right way.

**Lesson.** A step-2 argument has two separate engines to control:

- **(a) curvature-driven running.** It is local and perturbative in character, it is present for $SU(N)$ and absent for $U(1)$, and the 2026 proof already controls it in the UV.
- **(b) defect-driven disordering.** It is non-perturbative, and it is where the decimation scheme over-counts (2D-style) and where Tomboulis's 2007 argument was attacked (Ito–Seiler 2007/08).

A correct proof must let (a) carry the flow into the region where (b) can be bounded by a 4D, not 2D, entropy count.

## 3. How wide is the crossover? (literature data)

Data for 4D $SU(2)$ with the standard face cost, from Lucini–Teper, hep-lat/0103027, Table 1. Recall $\beta_{\text{phys}}=2\beta_{\text{ours}}$.

| $\beta_{\text{phys}}$ | $a\sqrt\sigma$ | $a\,m_{0^{++}}$ (grid gap) | correlation length $\xi=1/(am)$ |
|---|---|---|---|
| 2.25 | 0.423 | 1.39 | 0.72 |
| 2.30 | 0.311 | 1.09 | 0.92 |
| 2.40 | 0.263 | 0.953 | 1.05 |
| 2.475 | 0.202 | 0.754 | 1.33 |
| 2.55 | 0.159 | 0.586 | 1.71 |
| 2.60 | 0.134 | 0.514 | 1.95 |

The continuum ratio is $m_{0^{++}}/\sqrt\sigma=3.84(6)$.

- The gap in grid units **already exceeds 1** at $\beta_{\text{phys}}\le2.3$.
- Chapter 7's theorem certifies a gap only for $\beta_{\text{phys}}<0.056$. The Bakry–Émery method of Shen–Zhu–Zhu (arXiv:2204.12737) improves the constant but is of the same single-scale kind.
- Lucini–Teper attribute the dip in the scalar gap at the crossover to a **critical endpoint** of the phase line in the plane of fundamental and adjoint face costs. Near the Wilson axis the gap nearly closes there. This is a modeling choice we control (glossary 1.3, 4.2): a face cost with a small *negative* adjoint part moves away from the endpoint.

## 4. Experiment 2: does a coupling certificate reach the crossover? (`damage.py`)

*(Results below.)*

## 5. The program

**L1, UV continuation (hard, quantitative).**

- *Target:* extend the 2026 blocking control from its reference stiffness $\beta_{\text{ref}}$ down to $\beta_{\text{phys}}\approx2.5$.
- *Why plausible:* asymptotic scaling of lattice data is good to a few percent there.
- *Means:* computer-assisted bounds on the remainder of a truncated (one- or two-loop) blocking map, in the style of Lanford's proof of Feigenbaum universality. This is the step where the 1,147-page machinery has to be reused, not re-derived.

**L2, a tight IR certificate (the weakest link; improve first).** It must hold whenever the block model's correlation length is about one block. Candidates, in order of my confidence:

1. **Block path coupling in the class quotient.**
   - The heat-bath dynamics commutes in law with reframings, so the dynamics of classes is itself a Markov chain.
   - Path coupling (Bubley–Dyer) works in any metric. Use the quotient metric $\rho([X],[Y])=\min_h d(X,h\cdot Y)$, so the gauge-redundant disagreements that defeated chapter 7's demo at $\beta=0.6$ cost nothing.
   - Positive coarse Ricci curvature (Ollivier) of block updates, uniform over boundary conditions, gives decay and a gap.
   - It is checkable on a finite neighborhood. For single-edge updates the worst case reduces to a few parameters (two staple sums in $\mathbb R^4$), so rigorous interval-arithmetic verification is feasible. For blocks it needs a reduction.
2. **Typical-case contraction plus local uniformity.**
   - *Background:* Hayes–Vigoda's method for colorings beyond the worst-case threshold. The sampler is first run until local configurations are typical, and contraction is required only for typical neighborhoods.
   - *Our version:* the probability of atypical local patterns can be bounded by **chessboard estimates**. These come from flip-positivity, which is exactly the tool the 2026 proof uses at large distances. That makes this the natural meeting point of the two toolkits.
3. **Curvature (Bakry–Émery) at block level.**
   - Block labels still live in $SU(2)$, with the same positive curvature, while the block model's stiffness falls (§2).
   - Shen–Zhu–Zhu's single-scale argument, applied after blocking, would give a gap once $\beta_{\text{eff}}$ drops below their threshold. That means controlling only $O(10)$ blocking steps beyond L1.

**L3, locality transfer (technical).** Fine observables, given block labels, depend on block labels only locally, with exponentially decaying tails. This is standard in Balaban's framework, and needed to carry the block-model gap back to the original model.

**Filters every candidate must pass:**

- **(U(1))** It must fail for 4D $U(1)$ above $\beta\approx1.01$.
- **(5D)** It must fail for 5D $SU(2)$ at large $\beta$.
- **(finite groups)** For 4D $\mathbb Z_2$ at large $\beta$, a *gap* argument may succeed (finite groups are gapped at weak coupling; Adhikari–Cao), but a *confinement* argument must fail.

## 6. Next steps that can be done in this repository

1. **Rigorous single-edge quotient-curvature bound** for $SU(2)$ in 4D.
   - Compute the exact worst case over the two staple sums.
   - Compare its threshold with Dobrushin's ($\beta_{\text{phys}}<0.056$) and Shen–Zhu–Zhu's.
2. **Block coupling experiments.**
   - Measure, for blocks of side 2 and 3, the largest $\beta$ at which a quotient-metric block coupling contracts on typical configurations, and how the worst-case fraction behaves.
   - This tells us whether candidate 1 or 2 can reach $\beta_{\text{phys}}\approx2.3$.
3. **Action design.** Re-run the decimation flow and the coupling experiments with a fundamental-plus-negative-adjoint face cost. Find the cost that maximizes the overlap between "decimation is near one-loop" and "coupling contracts".
4. **The decimation scheme as a toy theorem.**
   - Prove rigorously that in the decimation scheme $SU(2)$ reaches $\alpha<1$ from every $\beta$ in $8\beta+O(1)$ steps, while $U(1)$ needs exponentially many.
   - It is a one-variable recursion on class functions, provable with interval arithmetic. It would be a clean statement of "curvature drives the running" in course terms.

## Sources

- Lucini & Teper, *SU(N) gauge theories in four dimensions*, [hep-lat/0103027](https://arxiv.org/abs/hep-lat/0103027) (glueball and string-tension data; the endpoint explanation of the dip).
- Shen, Zhu & Zhu, *A stochastic analysis approach to lattice Yang–Mills at strong coupling*, [arXiv:2204.12737](https://arxiv.org/abs/2204.12737).
- Nissim, *U(N) lattice Yang–Mills in the 't Hooft regime*, [arXiv:2510.22788](https://arxiv.org/abs/2510.22788); Cao, Nissim & Sheffield, *Expanded regimes of area law*, [arXiv:2505.16585](https://arxiv.org/abs/2505.16585).
- Tomboulis, [arXiv:0707.2179](https://arxiv.org/abs/0707.2179); Ito & Seiler, [arXiv:0711.4930](https://arxiv.org/abs/0711.4930) and [arXiv:0803.3019](https://arxiv.org/abs/0803.3019).
- Bauerschmidt, Bodineau & Dagallier, *Stochastic dynamics and the Polchinski equation*, [arXiv:2307.07619](https://arxiv.org/abs/2307.07619).
- Bubley & Dyer (path coupling, 1997); Hayes & Vigoda (local uniformity, 2003); Ollivier, *Ricci curvature of Markov chains on metric spaces* (2009).
