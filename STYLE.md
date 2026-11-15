# Chapter style guide

## Source format and build

- **Source of truth:** each chapter is a jupytext "percent" file, `chapters/chNN_slug.py`.
  - Markdown cells start with `# %% [markdown]`; code cells start with `# %%`.
- **Build and execute** a chapter with:

  ```bash
  uv run jupytext --to ipynb --set-kernel lgt --execute chapters/chNN_slug.py
  ```

  This writes the executed `chapters/chNN_slug.ipynb`.
- **Runtime:** each chapter must execute in under 2 minutes.
- **Code location:**
  - Library code lives in `src/lgt`. Import it with `from lgt import *` and `from lgt import viz`.
  - Helpers specific to one chapter go inside the chapter itself, in a clearly marked "helpers" cell near the top or right where they are needed.
  - Do not edit `src/lgt` from a chapter.

## Terminology

- `GLOSSARY.md` is the contract. Use its words, with its numbering where helpful (e.g. "(glossary 2.4)").
- **Never use a term before defining it.**
- Baseline knowledge you may assume, and nothing else:
  - sets and functions;
  - linear algebra;
  - graphs (vertices, edges, walks, cycles, spanning trees);
  - probability (distributions, expectation, Markov chains);
  - group theory (groups, homomorphisms, conjugation, conjugacy classes).
- **Physics names** (plaquette, holonomy, gauge, Wilson loop, action, Haar, coupling, …) appear only:
  - in the end-of-chapter **Dictionary** box;
  - or in an occasional "*(physicists call this …)*" aside, after our own term is defined.
- **State the type of every object** when it is introduced. For example: $U:\vec E\to G$; $h:V\to G$; $\langle f\rangle_\beta\in\mathbb R$; $\varphi:G\to\mathbb R$.

## Notation

| Object | Notation |
|---|---|
| board | $X$, with vertex set $V$, edge set $E$, set of directions $\vec E$, list of faces $F$ |
| label group | $G$ |
| labeling | $U$, with labels written $U(s\to t)$ |
| reframing | $h$ |
| walk | $\gamma$ |
| walk product | $U(\gamma)$ |
| face | $f$ |
| face product | $U_f$ |
| face cost | $\varphi$ |
| total cost | $S(U)$ |
| stiffness | $\beta$ |
| model | $\mu_\beta$ |
| average | $\langle f\rangle_\beta$ |
| class | $[U]$ |

Products along a walk are written in walking order: $U(\gamma)=U(d_1)U(d_2)\cdots U(d_k)$.

## Shape of a chapter

1. **Title and a one-paragraph hook.** Start with a concrete puzzle or story, not a definition.
2. **What you'll learn:** three bullets.
3. **Prerequisites:** the earlier chapters the reader needs.
4. **Sections,** each following this pattern:
   - intuition in plain words;
   - the definition, with types;
   - a tiny worked example computed in code;
   - a visualization, if one genuinely helps;
   - a **Recap**: a blockquote starting `> **Recap.**`;
   - **Check yourself**: 2–4 questions, each with its answer hidden in a `<details>` block:

     ```
     <details><summary>Answer</summary>

     ... markdown and math ...

     </details>
     ```

5. **Facts** (theorems) are stated in a clearly marked blockquote:

   ```
   > **Fact N.M (name).** ...
   ```

   - Immediately verify each fact numerically on an example.
   - Give a proof only when it is short and illuminating. Otherwise give a one-line "why".
6. **End matter:**
   - a chapter **Recap** (5–7 bullets);
   - a **Dictionary** table (our term, other names);
   - a "**What's next**" teaser.

## Markdown that renders everywhere

The reader's renderer is stricter than Jupyter's. Follow these rules:

- **Never use `$` or `\$` for money.** Write "dollars" or "USD" instead.
- **Put every display equation on its own lines:** `$$` alone on a line, with a blank line before and after. Never put one inside a sentence or a list item.
- **Open and close inline math on the same line.**
- **In `<details>` blocks,** leave a blank line after `<summary>…</summary>` and another before `</details>`.

## Approachability

- Write in short paragraphs, plain English, and second person ("you").
- Keep code cells short (about 15 lines at most). Move heavy lifting into `lgt` or a helper cell.
- Print small, readable outputs. Round numbers and label them.
- Every figure gets one sentence telling the reader *what to look at*.

## Visual language

The shared helpers are in `lgt.viz`. Call `viz.use_style()` once per chapter.

- Draw boards with `viz.draw_board`. Its conventions:
  - clock hands for circle-valued labels;
  - gold edges for $-1$ in $\mathbb Z_2$;
  - coral faces for defects;
  - pale faces for the identity;
  - grey edges for a spanning tree;
  - teal for highlighted walks.
- Show animations with `IPython.display.HTML(anim.to_jshtml())`, so they survive in the saved notebook.
- Use Plotly only for genuinely 3D content.
- Visualizations must earn their place:
  - each one shows a *mechanism* or an *invariance*;
  - no decorative charts;
  - prefer a few large, clear figures over many small ones.
