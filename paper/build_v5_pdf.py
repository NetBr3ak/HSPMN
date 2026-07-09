"""Compact reportlab preview of the final v5 paper (HSPMN_v5_draft.tex).

The canonical artifact is the LaTeX source; arXiv compiles it. This script
renders a faithful local preview (title, abstract, contributions, all key
tables, the proposition, figures, conclusion) without a TeX install.

Run: python3 build_v5_pdf.py  ->  HSPMN_v5_draft.pdf
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="Body", parent=s["Normal"], fontSize=10,
                         leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5))
    s.add(ParagraphStyle(name="Sect", parent=s["Heading2"], fontSize=12.5,
                         spaceBefore=12, spaceAfter=4))
    s.add(ParagraphStyle(name="Sub", parent=s["Heading3"], fontSize=11,
                         spaceBefore=8, spaceAfter=3))
    s.add(ParagraphStyle(name="Prop", parent=s["Body"], leftIndent=14,
                         rightIndent=14, borderColor=colors.grey,
                         borderWidth=0.6, borderPadding=7,
                         backColor=colors.Color(0.96, 0.96, 0.98)))
    s.add(ParagraphStyle(name="Cap", parent=s["Body"], fontSize=8.7,
                         leading=11, textColor=colors.Color(0.25, 0.25, 0.25)))
    return s


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def fig(name, width, caption, s, story):
    p = os.path.join(HERE, "figures", name.replace(".pdf", ".png"))
    if os.path.exists(p):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(p).getSize()
        story.append(Image(p, width=width, height=width * ih / iw))
        story.append(Paragraph(caption, s["Cap"]))
        story.append(Spacer(1, 6))


def build():
    s = styles()
    out = os.path.join(HERE, "HSPMN_v5_draft.pdf")
    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    st = []

    st.append(Paragraph(
        "<b>Nothing to Route On: A Measured Information Ceiling Explains "
        "Routing Absorption in Hybrid Mamba-Attention Language Models</b>",
        ParagraphStyle("t", parent=s["Body"], fontSize=14.5, leading=18,
                       alignment=TA_CENTER, spaceAfter=8)))
    st.append(Paragraph(
        "Szymon Jędryczko · Tenzan Logic, Kraków · "
        "szymon.jendryczkos@gmail.com · June 2026<br/>"
        "<i>(local preview rendered from HSPMN_v5_draft.tex; arXiv compiles "
        "the LaTeX source)</i>",
        ParagraphStyle("a", parent=s["Body"], fontSize=9.5,
                       alignment=TA_CENTER, spaceAfter=14)))

    st.append(Paragraph("<b>Abstract</b>", s["Sub"]))
    for para in [
        "Small hybrid language models pair an attention stream with a "
        "state-space stream and add a learned gate that decides, token by "
        "token, how much each stream contributes. We ask a simple question: "
        "does the gate actually do anything? Across 134 training runs at 140M "
        "and 350M parameters on a single consumer GPU, the answer is no, and "
        "we can measure why.",
        "First, the architecture. A block-level dual-stream design with a "
        "learned router loses to plain head-level fusion by +19.5% perplexity "
        "at matched parameters and ingredients, and we trace the loss to the "
        "router itself: 6 of 12 layers have a gate that is zero for every "
        "token and two more route under 0.03% of tokens. Second, the gate. "
        "The learned gate fails to beat a frozen random gate at 140M and "
        "again at 350M (3 seeds each), reproducing the routing-absorption "
        "result of Aquino-Michaels at 11x their scale.",
        "Third, the explanation. We treat the gate as a noisy channel between "
        "the token and the event 'the attention stream lowers the loss here', "
        "and prove via Fano and Pinsker inequalities that a gate's benefit "
        "over an always-on baseline is capped by sqrt(2(I + log2 - H(s))). "
        "Realising the full routing benefit needs about 0.72 bits. Trained "
        "gates carry 0.004 to 0.006 bits. An optimal probe trained directly "
        "on the oracle label reaches only 0.034 bits: 21x short. The routing "
        "target is nearly information-free, so no token-local gate of this "
        "class can work, trained or not. Ablating the attention stream hurts "
        "on 256 of 256 held-out sequences and 98.9% of the benefit variance "
        "lies within sequences: the optimal routing policy is simply 'always "
        "on', which is the gate-less baseline. The gates fail not because "
        "routing is hard to learn but because, on this data, there is "
        "nothing to route on. The theory identifies the two levers that "
        "could change this, data heterogeneity and stream non-redundancy, "
        "and we close by pre-registering both as a falsifiable programme "
        "with predictions committed before the results.",
    ]:
        st.append(Paragraph(para, s["Body"]))

    st.append(Paragraph("Contributions", s["Sect"]))
    for c in [
        "<b>1. Block fusion retired, with a mechanism.</b> +19.5% PPL loss vs "
        "head fusion at matched ingredients; router modally collapsed (6 dead "
        "+ 2 effectively dead of 12 layers; one saturated layer at gate mean "
        "47, std 316); 19% more forward FLOPs per token.",
        "<b>2. The gate null, replicated at two scales.</b> Learned vs random "
        "gate: 143.85±9.27 vs 146.24±7.56 at 140M; 82.35±4.78 vs 83.79±1.37 "
        "at 350M (Welch t = -0.41). Gate-less beats both. Head fusion's own "
        "-3.4% edge over dense at 140M decays to a tie at 350M.",
        "<b>3. A gate-channel bound that predicts the null.</b> Advantage cap "
        "sqrt(2(I + log2 - H(s))); full benefit needs ~0.72 bits; measured "
        "gates carry 0.0059±0.0004 (learned) vs 0.0029±0.0006 (random) bits, "
        "capping every gate at <= 15% of the ideal benefit.",
        "<b>4. The ceiling: the null is structural.</b> Optimal probes "
        "trained on the oracle label reach 0.034 bits (21x below threshold; "
        "any token-local gate capped at 24% of ideal). Attention helps on "
        "256/256 sequences; 98.9% of benefit variance is within-sequence. "
        "Optimal policy: always on.",
        "<b>5. A pre-registered programme, executed.</b> Routability phase "
        "diagram over mixture corpora with recall fraction pi in {0, .1, "
        ".25, .5}: the oracle-label ceiling rises monotonically 0.018 -> "
        "0.139 bits (P1 confirmed) yet stays 5x below threshold, and every "
        "gate-vs-random null lands where the bound requires (P2, Welch p "
        ">= 0.52 at all pi). Stream decorrelation fails to raise the "
        "ceiling (P4 falsified). Post-hoc: the parameter-free PC gate beats "
        "the learned gate pooled across the diagram (-2.35 PPL, p = 0.041) "
        "and ties gate-less.",
    ]:
        st.append(Paragraph(c, s["Body"]))

    st.append(PageBreak())
    st.append(Paragraph("Table 1: 140M results (3 seeds, best LR per variant)",
                        s["Sub"]))
    st.append(table([
        ["Variant", "PPL (3 seeds)", "vs dense", "tok/s"],
        ["dense (muP-tuned)", "144.01 ± 5.64", "", "83,154"],
        ["hymba (head fusion, dense attn)", "143.42 ± 9.98", "-0.4%", "66,071"],
        ["hymba-with-nsa (gate-less)", "139.14 ± 8.43", "-3.4%", "50,582"],
        ["  + PC gate (parameter-free)", "140.09 ± 7.21", "-2.7%", "49,204"],
        ["  + param-free select", "141.01 ± 9.75", "-2.1%", "45,764"],
        ["  + learned gate", "143.85 ± 9.27", "-0.1%", "49,720"],
        ["  + random gate (A-M control)", "146.24 ± 7.56", "+1.6%", "50,029"],
        ["  ASA alternation", "149.29 ± 9.42", "+3.7%", "61,795"],
        ["v4-gdn-nsa (block fusion + router)", "166.23 ± 15.12", "+15.4%", "35,578"],
        ["v4-rwkv7-nsa (block fusion)", "169.55 ± 14.03", "+17.7%", "32,551"],
    ]))
    st.append(Spacer(1, 8))

    st.append(Paragraph("Table 2: 350M results (3 seeds each, 983M tokens)",
                        s["Sub"]))
    st.append(table([
        ["Variant (350M)", "PPL by seed 42/1337/2026", "mean ± std"],
        ["hymba-with-nsa (348.8M)", "87.38 / 86.99 / 81.60", "85.32 ± 2.64"],
        ["dense, muP (368.8M)", "84.23 / 80.18 / 86.82", "83.75 ± 2.73"],
        ["hymba-with-nsa-gated", "78.25 / 89.06 / 79.75", "82.35 ± 4.78"],
        ["hymba-with-nsa-randgate", "82.93 / 85.73 / 82.73", "83.79 ± 1.37"],
    ]))
    st.append(Paragraph(
        "Neither contrast significant (Welch t = 0.72 and -0.41). The seed-42 "
        "gated run (78.25) looked like a 10% win until seeds 1337/2026 "
        "arrived; seed luck, told in full in the paper.", s["Cap"]))
    st.append(Spacer(1, 8))

    st.append(Paragraph("Proposition (gate-channel advantage bound)", s["Sub"]))
    st.append(Paragraph(
        "With entropies in nats, any routing decision derived from the gate "
        "has advantage over chance capped by (1 - 2 P_err)+ <= "
        "sqrt(2(I(g;s) + log2 - H(s))) =: A(I), and its benefit over the "
        "always-on baseline is capped by (sigma_C^2 / 2L) * A(I). Realising "
        "the full ideal benefit requires I >= 1/2 + H(s) - log2 nats, about "
        "0.72 bits for a near-balanced label. Proof: Fano's inequality plus "
        "Pinsker's inequality (the chord bound H_b(q) <= 2q log2 is false by "
        "concavity; Pinsker gives the valid direction). Corollary: at I = 0 "
        "and balanced s, the cap is exactly zero, recovering the "
        "Aquino-Michaels null.", s["Prop"]))
    st.append(Paragraph(
        "Measured: learned 0.0059, random 0.0029, PC 0.0072 bits at 140M "
        "(non-circular control confirms); 0.0040 / 0.0021 at 350M. Advantage "
        "caps: 14% / 13% / 15% of ideal. Ceiling probes (linear on h_t: "
        "0.0315; MLP on both streams' outputs: 0.0341 bits) cap ANY "
        "token-local gate at 24% of ideal.", s["Body"]))

    st.append(PageBreak())
    st.append(Paragraph("Figures", s["Sect"]))
    fig("fig_scaling_advantage.pdf", 4.6 * inch,
        "Fig 1: Head fusion's PPL advantage decays from -3.4% (140M) to a "
        "tie at 350M.", s, st)
    fig("fig_gate_null_350m.pdf", 4.6 * inch,
        "Fig 2: Learned vs random gate at 350M: means tie; learned is the "
        "higher-variance way to get nothing.", s, st)
    fig("fig_mi_threshold.pdf", 4.6 * inch,
        "Fig 3: Gate mutual information vs the full-benefit threshold (log "
        "scale): two decades short at both scales.", s, st)
    fig("fig_gate_ceiling.pdf", 4.6 * inch,
        "Fig 4: Optimal-probe ceiling per layer: ~0.034 bits, 21x below "
        "threshold.", s, st)
    fig("fig_v6_phase.pdf", 6.4 * inch,
        "Fig 5: The executed routability phase diagram. Left: ceiling and "
        "gate MI vs recall fraction pi (log scale) against the 0.72-bit "
        "threshold. Right: PPL by variant; learned ties random at every pi.",
        s, st)
    fig("router_active_fraction_heatmap.pdf", 4.6 * inch,
        "Fig 6: Router modal collapse in block fusion: per-layer "
        "all-or-nothing despite on-target batch averages.", s, st)

    st.append(Paragraph("Conclusion (abridged)", s["Sect"]))
    st.append(Paragraph(
        "We set out to build a better router and ended up measuring why no "
        "such router can exist on this data. A gate needs about 0.72 bits "
        "about where attention helps; trained gates carry 0.005, and the "
        "oracle-label ceiling is 0.034, because attention helps everywhere "
        "and the exceptions are noise. The optimal policy is always-on, the "
        "gate-less baseline wins by construction, and the right response to "
        "routing absorption at this scale is no gate, plus data on which the "
        "question is worth asking. Probe the ceiling before training the "
        "router.", s["Body"]))
    st.append(Paragraph(
        "The constructive programme (pre-registered, now executed): mixture "
        "corpora with recall fraction pi in {0, 0.1, 0.25, 0.5} dial the "
        "routable structure up from zero. P1 confirmed: the probe ceiling "
        "I_max(pi) rises 0.018 -> 0.058 -> 0.108 -> 0.139 bits. P2 "
        "consistent: still 5x below threshold, and the learned gate ties "
        "random at every pi, exactly as the bound requires. P3: no gate "
        "benefit anywhere; the recall task itself was not solved at this "
        "budget. P4 falsified: stream decorrelation does not raise the "
        "ceiling (0.094 vs 0.108 bits). Bonus: the parameter-free PC gate "
        "beats the learned gate pooled across the diagram and ties the "
        "gate-less baseline. Artifacts: run_v6_queue.py, "
        "results/v6_routability_summary.md.",
        s["Body"]))

    doc.build(st)
    print(f"Built: {out}  ({os.path.getsize(out)/1024:.1f} KB)")


if __name__ == "__main__":
    build()
