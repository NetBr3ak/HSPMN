HSPMN v5: unpublished manuscript
================================

Title: Nothing to Route On: A Measured Information Ceiling Explains
       Routing Absorption in Hybrid Mamba-Attention Language Models
Author: Szymon Jędryczko

Main source: HSPMN_v5_draft.tex
Included source: appendix_tables.tex
Figure PDFs: figures/
Recorded evidence: ../results/
Code and result limitations: ../TROUBLESHOOTING.md
Reproduction commands: ../REPRODUCE.md

Compile from this directory with TeX Live:
    pdflatex HSPMN_v5_draft.tex
    pdflatex HSPMN_v5_draft.tex

The bibliography is included in the source; no BibTeX run is needed.

No generated preview is shipped. build_v5_pdf.py writes HSPMN_v5_draft.pdf
using its own text and tables; it does not compile or parse the TeX source. The preview
can differ from a TeX build. Its figure helper uses PNG versions and omits
a figure when its PNG is absent.

make_figures.py generates the project figure PDF/PNG pairs. The router
heatmap has a source CSV in ../results/, but its original rendering script
is not included. Fonts retain the licence in fonts/LICENSE.

The manuscript in one paragraph:
  The manuscript reports 140M and 350M runs on one RTX 5090. Block-level
  fusion with a learned router has 19.47% higher PPL than head fusion at
  140M. A learned per-token gate ties a frozen random gate at both scales
  (3 seeds each). A gate-channel bound needs about 0.72 bits for full
  benefit; trained gates carry 0.004 to 0.006 bits, and an optimal probe
  reaches about 0.034 bits. The manuscript concludes that the gate-less
  baseline is the best routing policy.

These runs trained and validated on a t+2 target: each position predicted
the token two places ahead, not the next token. The probes score next-token
CE on those models. The paragraph above is what the manuscript claims, not a
verified next-token finding. Evidence: ../README.md, section "Reproduction
status", and ../results/v6_routability_summary.md, lines 7 and 80 to 82.

The manuscript and recorded measurements are preserved. Reference-code
corrections do not validate or update the manuscript's scientific claims.
