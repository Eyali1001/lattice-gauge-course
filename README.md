# Labelings, models, and gauge theory

A computational course for CS and math undergraduates.

- **It starts from an abstract framework:** random labelings of a graph by a group.
- **It derives "quantum dynamics" as a consequence,** rather than assuming it.
- **It only then turns to physics,** showing that lattice Yang–Mills is one model in the framework.

Every idea is checked numerically on an example small enough to enumerate exactly.

- [GLOSSARY.md](GLOSSARY.md): the agreed terminology. It is the contract for every chapter.
- [STYLE.md](STYLE.md): how chapters are written and built.
- [notes/yang_mills_existence_2026.md](notes/yang_mills_existence_2026.md): reading notes on the 2026 Lean proof of Yang–Mills existence, in the course's terms.

## Course map

| Part | Chapter | Status |
|---|---|---|
| **I. The world of labelings** | 0. A market of currencies | draft |
| | 1. Labelings, reframings and classes | draft |
| | Interlude: why the labels are forced (charged fields, electromagnetism) | draft |
| | 2. Models: a probability distribution on labelings | draft |
| | 3. What a model can tell you: invariants and averages | draft |
| | 4. Sampling: averages without enumeration | draft |
| **II. Dynamics from the distribution** | 5. Layers and the transfer matrix | draft |
| **Bridge** | 6. Yang–Mills and the Millennium problem, in our terms | draft |
| **III. What can be proved** | 7. A gap you can prove (influence, coupling, and why it stops short of the continuum) | draft |

Possible later chapters:
- the 2D model solved exactly, with the Witten/Mednykh counting formula;
- exact duality and the derivation of 0.4407;
- a proof that the coin model has a phase transition in 3D and above;
- Coulomb's law from the 4D circle model;
- asymptotic scaling;
- analogues beyond physics.

## Running

```bash
uv sync
```

```bash
uv run jupyter lab chapters/
```

Rebuild and execute every chapter from its `.py` source:

```bash
./build.sh
```

Run the library tests:

```bash
uv run pytest -q
```
