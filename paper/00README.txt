HSPMN v5 - arXiv source package
================================

Title: Nothing to Route On: A Measured Information Ceiling Explains
       Routing Absorption in Hybrid Mamba-Attention Language Models
Author: Szymon Jędryczko (Tenzan Logic, Kraków)

MAIN FILE (arXiv compiles this): HSPMN_v5_draft.tex
  \input dependencies (must stay in the same dir):
    - appendix_tables.tex        (params / FLOPs / synthetics appendices)
  figures (PDF, in figures/):
    - fig_scaling_advantage.pdf          (140M advantage decays to tie at 350M)
    - fig_gate_null_350m.pdf             (learned vs random gate @350M)
    - fig_mi_threshold.pdf               (gate MI vs full-benefit threshold)
    - fig_gate_ceiling.pdf               (optimal-probe information ceiling)
    - fig_v6_phase.pdf                   (routability phase diagram, Section 5)
    - figures/router_active_fraction_heatmap.pdf  (router modal collapse)

Build (TeX Live): pdflatex HSPMN_v5_draft && pdflatex HSPMN_v5_draft
  (no bibtex run needed - manual thebibliography; natbib[numbers]).

Packages used (all standard TeX Live): amsmath, amssymb, amsthm, graphicx,
booktabs, xcolor, natbib, hyperref, geometry, microtype, inputenc(utf8).

Paper in one paragraph:
  72 training runs at 140M and 350M on one RTX 5090. (1) Block-level
  dual-stream fusion with a learned router loses to plain head fusion by
  +19.5% PPL at matched ingredients; the router modally collapses (7/12
  layers dead). (2) A learned per-token gate ties a frozen random gate at
  both scales (3 seeds each), replicating Aquino-Michaels at 11x their
  scale; head fusion's own -3.4% advantage at 140M decays to a tie at
  350M. (3) A Fano/Pinsker gate-channel bound caps any gate's benefit by
  sqrt(2(I + log2 - H(s))): full benefit needs ~0.72 bits, trained gates
  carry 0.004-0.006 bits. (4) An optimal probe trained on the oracle label
  itself reaches only ~0.034 bits (~21x short; advantage cap ~24% of
  ideal), ablating attention hurts on 256/256 sequences, and 99% of the
  benefit variance is within-sequence: the optimal routing policy is
  always-on, i.e. the gate-less baseline. The null is structural; there is
  nothing to route on.

All numbers reproduce from the accompanying code; the Reproduction
appendix maps each table to its source log or checkpoint.

Note: an earlier draft had the gate-channel theorem as a standalone
theorem_gate_channel.tex section; it is not part of this release. Its
corrected content is inlined directly in Section 6 of the paper.
