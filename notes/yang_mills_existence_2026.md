# Notes: the 2026 Yang–Mills existence proof, in this course's terms

These are reading notes on OpenAI's Lean proof that a quantum Yang–Mills theory exists, together with the Yang–Mills group's validation. The notes use the course vocabulary (`GLOSSARY.md`) wherever possible. They were written on 13 November 2026 from these primary sources:

- **Committee statements:**
  - [first statement (October 2026)](https://www.kitp.ucsb.edu/sites/default/files/YM_working_group.pdf);
  - [third statement (9 November 2026)](https://www.kitp.ucsb.edu/sites/default/files/3rd-Statement-About-Quantum-YangMills-Theory_Nov-9.pdf).
- **Lean code as submitted on 28 October:** [yang-mills-group/OpenAI-YM-20261028](https://github.com/yang-mills-group/OpenAI-YM-20261028).
- **Validation:** [yang-mills-group/YM](https://github.com/yang-mills-group/YM) and its 35-page [assessment-1.pdf](https://github.com/yang-mills-group/YM/blob/main/docs/assessment-1.pdf), dated 10 November 2026.

The 1,147-page informal manuscript has not been released, so these notes rest on the Lean statements and on the assessment.

## 1. What happened

| Date (2026) | Event |
|---|---|
| 9 Sep | OpenAI's models produce a natural-language proof (per OpenAI). |
| 16 Oct | The Lean formalization is finished. OpenAI asks David Gross to take custody. Gross then forms a "shepherding committee": Douglas, Gross (chair), Jaffe, Klebanov, Nekrasov. |
| 28 Oct | The Lean code is submitted to the committee. |
| 30 Oct – 5 Nov | The committee runs three Comparator checks (public theorem; its own stronger combined theorem; the OPE statement). |
| 4 Nov | OpenAI makes a second submission. Its review is "work in progress". |
| 10 Nov | Both repositories and the assessment are made public. |

**Verdict, in the assessment's words:** the submission "proves that a relativistic quantum field theory satisfying the Wightman axioms can be constructed as a scaling limit of four-dimensional Wilson lattice gauge theory for any simple compact gauge group." The same sentence goes on: it "does not prove that the infinite-volume limit is unique or that there is a mass gap."

## 2. The Lean statement, decoded

The public target is `YangMills.JointExistence`, in `ChallengeSpec.lean`:

```lean
def JointExistence : Prop := ∀ g : GaugeGroup, g.IsSimple → ∃ C : WilsonFamily g,
  WilsonRegularity C ∧ ∀ p : ℕ → ℕ, StrictMono p → ∃ q : ℕ → ℕ, StrictMono q ∧
    ∃ T : Hierarchy C, T.Properties (p ∘ q)
```

Here it is line by line, in our words.

| Lean | Our terms |
|---|---|
| `g : GaugeGroup`, `g.IsSimple` | A label group: a closed subgroup of $U(m)$ whose Lie algebra is simple and non-abelian. Examples are $SU(2)$ and $SU(3)$. |
| `C : WilsonFamily g` | A family of models (glossary 4.5) on 4D torus grids. They have the standard face cost, with $\varphi_{\text{theirs}}=\beta(1-\tfrac1m\operatorname{Re}\operatorname{tr})$, so $\beta_{\text{theirs}}=m\beta_{\text{ours}}$. Each model is indexed by a spacing $a_N\to0$ and a side length $L_p\to\infty$, both chosen by the proof. `C.insertion` attaches a local observable to each "label" of a continuum field. |
| `WilsonRegularity C` | `running`: $\beta_{N,p}/\log(1/a_N)$ tends to a positive slope, uniformly in volume. This is the logarithmic growth of stiffness from section 6.3, asymptotic freedom. `invariant`: every observable is an invariant (glossary 3.4). `localSupport`: every observable reads only a fixed finite patch of edges, at every spacing. |
| `∀ p, StrictMono p → ∃ q …` | For any increasing sequence of volumes there is a sub-subsequence along which everything converges. This is a compactness statement; the limit is not shown to be unique. |
| `T : Hierarchy C` | The limit object: a Hilbert space and Wightman fields, a compact space of "characters" (phases) with a probability measure, and correlation functions. It also includes an analytic continuation to real time, OPE coefficients, and the centered trace of the stress tensor. |
| `T.Properties` | Four conclusions about this one hierarchy: `wilson`, `reconstruction`, `ope` and `nonfree`. The table below spells them out. |

| Property | What it says |
|---|---|
| `wilson` | The continuum correlations are limits of the lattice averages (glossary 4.6). The spacing goes to zero first, at fixed volume; then the volume goes to infinity along the subsequence. |
| `reconstruction` | The Wightman axioms hold: positive energy, a unique vacuum within each phase, Poincaré covariance and locality. The fields are faithful and the phase measure has full support. The limit is a mixture of phases, and analytic continuation recovers the Euclidean correlations. |
| `ope` | An all-orders operator product expansion with explicit error bounds, uniform over phases. |
| `nonfree` | Non-triviality. The centered trace $\Theta$ of the stress tensor has a two-point function with $r^8\log^2(1/r)\,S_{\Theta\Theta}(r)\to 3\dim_{\mathbb R}\mathfrak g/\pi^4$. No finite combination of free-field profiles reproduces it. |

The detailed theorem `YangMills.constructed` also keeps the "construction certificates". They tie the continuum fields back to the specific lattice observables. The committee's own target, `YMReview.Acceptance.combined_acceptance`, requires these certificates. The reason: OpenAI later told the committee that the public target alone "omitted some of the connections to the lattice construction".

## 3. The architecture of the proof

This is a reading of the assessment's Part I. The section numbers refer to the assessment.

1. **Start from our model (assessment §4).** The input is the Wilson model on a 4D torus, with uniform labels and $e^{-\beta S}$. The committee proved in Lean that it equals an independently written lattice-gauge formalization (`mrdouglasny/lgt`).
2. **Blocking ("renormalization group", §4.3).** Following Balaban (1980s), the grid is coarsened step by step by integrating out fine-scale labels. The flow of the leading ("marginal") terms is computed explicitly and agrees with one-loop asymptotic freedom. The other ("irrelevant") terms are bounded and shown to stay small.
   - **One new ingredient:** the gauge fixing (in our words, a choice of normal form for each class, like glossary 3.8) is *averaged over all symmetry-breaking choices*. This keeps the grid's quarter-turn and reflection symmetries. A tree normal form would break them.
   - **Where the flow stops:** it runs from spacing $a_N$ up to a fixed reference spacing $1$, where the stiffness has a fixed value $\beta_{\text{ref}}>0$ (assessment §11). **It does not continue beyond that scale.**
3. **Continuum limit at fixed volume (Theorem 4.1).** Correlations of renormalized observables form a Cauchy sequence as $a_N\to0$, along the chosen sequence of spacings. Only test functions that avoid coincident points are allowed.
4. **Infinite volume, by compactness only (Theorem 5.1).** The bounds are uniform in volume: local boundedness and equicontinuity on compact sets. They come from general tools:
   - flip-positivity (glossary 5.7) and its "chessboard" consequence;
   - the transfer-matrix representation (glossary 5.6);
   - Cauchy estimates.

   Arzelà–Ascoli then gives a convergent subsequence. **No decay of correlations is proved or used.**
5. **Phases (§6.1).** The limit is not shown to cluster (correlations factorizing at large distance). So it is decomposed, by the Gelfand and Riesz–Markov theorems, into components that each do cluster. Each component then satisfies the OS axioms.
6. **Reconstruction (§6).** OS reconstruction, also proved in Lean, gives a Wightman theory in each phase. This is the continuum version of chapter 5's "derived system".
7. **Rotation symmetry (§6.3).** This is the hardest part conceptually, because the grid has only quarter-turns. The proof has three steps:
   - make the gauge fixing symmetric;
   - build a conserved stress tensor and its Ward identities;
   - show that the symmetry is not spontaneously broken in the limit. For this, the stress-energy flux must decay at infinity. The proof gets this from flip-positivity in several orthogonal planes.
8. **Evidence that it is Yang–Mills (§7):**
   - the trace anomaly coefficient above;
   - an effective coupling with $\log(1/r)\,g(r)^2\to 1/(2b_0)$, where $b_0=11C_{\mathfrak g}/(48\pi^2)$;
   - free Maxwell behavior at short distances;
   - an OPE.

## 4. How it was checked

- **Clean build:** 9,470 modules on Lean 4.32.0 in about 3 hours, with zero errors.
- **Size:** 250,319 declarations, of which 100,168 are reachable from the 17 headline theorems. The rest is unused code.
- **Axioms:** only the standard three (`propext`, `Classical.choice`, `Quot.sound`). There is no `sorry` except the placeholder in the reference statement.
- **Comparator:** it passed against frozen reference statements three times. Control runs (a changed statement, a forbidden axiom, a changed definition) were rejected as expected. On macOS the runs were done without process isolation.
- **Independent check of the OS axioms:** for the centered trace, it was made against a formalization written before the proof was known (`xiyin137/OSreconstruction`).
- **Caveats the committee itself states:**
  - "We do not claim to fully understand the proof yet."
  - OpenAI's Astra model was used to help digest the code.
  - Code quality is poor: whimsical names such as `Farewell.dead` and `anxiety` in `Parade.lean`, and 60% of declarations unused.

## 5. What is not proved (assessment §8)

1. Uniqueness of the infinite-volume limit, and convergence without passing to subsequences.
2. A probability distribution on continuum "labelings" (only correlation functions are constructed).
3. **A mass gap:** exponential decay of correlations, and a spectral gap of the reconstructed Hamiltonian.
4. Loop observables (Wilson loops) in the continuum, and the area law.
5. Particle states and their spectrum.

The October committee statement says "including coupling to fermions". The released Lean target is pure Yang–Mills only.

## 6. Notes for the mass-gap attempt

These are the lessons drawn for chapter 7.

- **The infrared side is soft.** Per the assessment, the large-volume step rests on compactness (Arzelà–Ascoli), flip-positivity, chessboard estimates and the transfer-operator representation. Clustering of correlations is not proved, which is why a phase decomposition is needed. (Rotation symmetry does use a fall-off of stress-energy flux at infinity, obtained from flip-positivity, but that is not a rate of decay of correlations.) A mass gap needs exactly the estimate the proof avoids: exponential clustering.
- **A gap implies clustering, so it would remove the phase decomposition.** Within one phase the vacuum is unique. A uniform gap would make the mixture trivial, at least for gap-compatible limits.
- **The blocking flow stops at $\beta_{\text{ref}}$, at the reference scale.** The flow is controlled while the effective stiffness is *large* (labels nearly consistent). The provable-gap regime of chapter 7 needs stiffness *small*, below about $0.03$ for $SU(2)$ in 4D. Between them lies the crossover visible in chapter 6's Creutz plot. No known method controls it. In our terms, this is precisely where the Millennium problem's gap lives.
- **Locality certificates are reusable.** Every observable reads a fixed finite patch (`localSupport`), and the effective costs after blocking are local. Influence-type arguments (chapter 7) need exactly this input. If a blocked model at some scale had influence number below 1, Fact 7.5 would give a positive gap at that scale.
- **The scale problem is the real obstruction.** A gap bounded below *in grid units* along the continuum path would make every particle infinitely heavy, so the limit would be trivial. A proof must show that the gap in grid units *shrinks* at exactly the asymptotic-freedom rate. That is a multi-scale statement. No single-scale argument (Dobrushin, path coupling, spectral independence, expanders) can give it on its own.
