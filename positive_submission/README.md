# Subject-Calibrated EOG-Guided Diffusion Manuscript

This directory contains an anonymous ACM-style manuscript organized as:

1. Abstract
2. Introduction
3. Related Work
4. Methodology
5. Experiments
6. Results
7. Conclusion

The paper describes one method: a population conditional diffusion denoiser guided by an EOG-to-EEG propagation matrix estimated from a separate participant-session calibration segment. The TeX source contains detailed placeholders for all planned figures. These placeholders specify the required real evaluator outputs and prevent illustrative waveforms or participant data from being fabricated.

`figures/drafts/method_schematic_draft.png` is a non-data-bearing concept sketch for the method figure. It is not included in the manuscript and should be redrawn as editable vector artwork before submission.

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

If `latexmk` is unavailable, run `pdflatex`, `bibtex`, and two further `pdflatex` passes. The current PDF is a review draft; authorship, affiliation, the anonymous code archive, and final data-derived figure assets remain to be completed before submission. The public source dataset and the ethics and consent statement from the original acquisition are identified in the Experiments section.
