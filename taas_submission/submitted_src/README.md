# SADDPM — TAAS submission (Overleaf-ready)

**Main file:** `main.tex` (set as the main document in Overleaf).
**Compiler:** pdfLaTeX + BibTeX (Overleaf default for the `acmart` class). Sequence:
`pdflatex → bibtex → pdflatex → pdflatex` (Overleaf runs this automatically).

## Contents
- `main.tex` — preamble, front matter, `\input`s the sections, bibliography, appendix.
- `sections/` — one file per section: `introduction`, `related_work`, `method`,
  `fig_architecture` (Figure 1), `experiments`, `conclusion`, `appendix` (proofs).
- `references.bib` — bibliography (ACM-Reference-Format).
- `figures/architecture.pdf` — Figure 1.
- `acmart.cls`, `ACM-Reference-Format.bst`, `acm-jdslogo.png` — ACM class files,
  bundled for reproducibility.

## If Overleaf complains about the bundled `acmart.cls`
Either set **Menu → TeX Live version → latest**, or delete `acmart.cls` and
`ACM-Reference-Format.bst` from the project so Overleaf uses its built-in `acmart`.

## Placeholders still to fill (NOT yet done — pending author input)
- Author block, ORCIDs, corresponding author.
- Copyright / DOI / volume / number / article / month / received dates.
- CCS concepts — regenerate the exact XML from https://dl.acm.org/ccs.
- Items raised in the latest review (see the chat): Table 2 `s6` mean,
  the Table 3 `(s7,s3)` value, the downstream-classifier name, and the
  subject-correlation metric definition. The numbers are left exactly as-is
  pending your audit.
