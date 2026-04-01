"""
clinician_figures.py — Publication-quality figures for bssentinel
Nature journal formatting:
  - Double-column width: 7.2 in (183 mm)
  - Font: sans-serif (Helvetica/Arial), 8 pt body
  - Resolution: 300 dpi
  - Colorblind-safe palette (Wong 2011)
  - constrained_layout=True throughout
  - Panel labels: bold uppercase A, B, ... via _panel_label() ONLY
    (never repeated inside set_title)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import roc_curve, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "lines.linewidth":    1.4,
    "lines.markersize":   4,
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

# ── Wong 2011 colorblind-safe palette ─────────────────────────────────────────
W = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "gray":    "#999999",
    "lgray":   "#F5F5F5",
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
with open("artifacts/metrics.json") as f:
    metrics = json.load(f)

oof = pd.read_parquet("artifacts/oof_predictions.parquet")
fi  = pd.read_csv("artifacts/feature_importance.csv")

out_dir = Path("artifacts/figures/clinician")
out_dir.mkdir(parents=True, exist_ok=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _minimal_spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=3, width=0.6)

def _panel_label(ax, letter, x=-0.13, y=1.06):
    """Bold panel letter — the ONLY place the letter appears (never in set_title)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            color=W["black"])


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Study design / patient monitoring schematic
# ══════════════════════════════════════════════════════════════════════════════

def fig1_study_design():
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    fig.patch.set_facecolor("white")

    # Layout: tall timeline on top, two info panels below
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[2.2, 1.0],
                           hspace=0.55, wspace=0.30,
                           left=0.04, right=0.97,
                           top=0.93, bottom=0.06)

    ax_tl = fig.add_subplot(gs[0, :])   # full-width timeline
    ax_in = fig.add_subplot(gs[1, 0])   # inputs
    ax_ou = fig.add_subplot(gs[1, 1])   # output / performance

    TL_Y = 0.3

    # ── Panel A: timeline ─────────────────────────────────────────────────────
    for sp in ax_tl.spines.values():
        sp.set_visible(False)
    ax_tl.set_xlim(0, 14)
    ax_tl.set_ylim(-1.9, 2.2)   # extra headroom for arrow text
    ax_tl.set_yticks([])
    ax_tl.set_xticks([])

    _panel_label(ax_tl, "A", x=-0.01, y=1.04)
    ax_tl.set_title("Patient monitoring — 6-hourly deterioration risk assessment",
                    loc="left", fontsize=9, fontweight="bold", pad=4, x=0.03)

    # Admission bar
    ax_tl.barh(TL_Y, 13, left=0.5, height=0.16,
               color=W["lgray"], edgecolor=W["gray"], linewidth=0.6, zorder=1)

    # Admission marker
    ax_tl.plot(0.5, TL_Y, "^", color=W["green"], ms=7, zorder=4)
    ax_tl.text(0.5, TL_Y + 0.32, "Admission", ha="center", va="bottom",
               fontsize=7, color=W["green"], fontweight="bold")

    # Discharge marker
    ax_tl.plot(13.5, TL_Y, "s", color=W["gray"], ms=6, zorder=4)
    ax_tl.text(13.5, TL_Y + 0.32, "Discharge /\nICU transfer", ha="center",
               va="bottom", fontsize=6.5, color=W["gray"])

    # Observation windows
    window_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]
    alert_set    = {9.5, 11.5}

    for x in window_times:
        is_alert = x in alert_set
        fc = W["vermil"] if is_alert else W["skyblue"]
        ec = W["vermil"] if is_alert else W["blue"]
        ax_tl.plot(x, TL_Y, "o", color=ec, ms=8, zorder=5,
                   markerfacecolor=fc, markeredgewidth=1.2)
        ax_tl.plot([x, x], [TL_Y - 0.08, TL_Y - 0.24], "-",
                   color=ec, lw=0.8, zorder=3)
        label = "Alert" if is_alert else "Check"
        ax_tl.text(x, TL_Y - 0.35, label, ha="center", va="top",
                   fontsize=6.5,
                   color=W["vermil"] if is_alert else W["gray"],
                   fontweight="bold" if is_alert else "normal")
        hour = int((x - 0.5) * 3)
        ax_tl.text(x, TL_Y + 0.32, f"+{hour}h", ha="center", va="bottom",
                   fontsize=6, color=W["gray"])

    # Deterioration event star
    event_x = 12.6
    ax_tl.plot(event_x, TL_Y, "*", color=W["vermil"], ms=12, zorder=6)
    ax_tl.text(event_x, TL_Y - 0.35, "Deterioration\nevent", ha="center",
               va="top", fontsize=6.5, color=W["vermil"], fontweight="bold")

    # Lead-time double arrow — placed above the window labels (+h row)
    arr_y = TL_Y + 1.05
    ax_tl.annotate("", xy=(event_x, arr_y), xytext=(9.5, arr_y),
                   arrowprops=dict(arrowstyle="<->", color=W["orange"],
                                   lw=1.2, mutation_scale=10))
    ax_tl.text((event_x + 9.5) / 2, arr_y + 0.16, "~9 h early warning",
               ha="center", va="bottom", fontsize=7,
               color=W["orange"], fontweight="bold")

    # 6-h lookback bracket — below the tick row
    bx = 7.5
    ax_tl.annotate("", xy=(bx - 1.0, TL_Y - 0.95), xytext=(bx + 0.8, TL_Y - 0.95),
                   arrowprops=dict(arrowstyle="<->", color=W["blue"],
                                   lw=0.9, mutation_scale=8))
    ax_tl.text(bx - 0.1, TL_Y - 1.10, "6 h lookback\nwindow", ha="center",
               va="top", fontsize=6, color=W["blue"])

    # 24-h prediction bracket
    ax_tl.annotate("", xy=(bx + 0.8, TL_Y - 0.95), xytext=(bx + 4.6, TL_Y - 0.95),
                   arrowprops=dict(arrowstyle="<->", color=W["vermil"],
                                   lw=0.9, mutation_scale=8))
    ax_tl.text(bx + 2.7, TL_Y - 1.10, "24 h prediction\nhorizon", ha="center",
               va="top", fontsize=6, color=W["vermil"])

    # ── Panel B: inputs ───────────────────────────────────────────────────────
    for sp in ax_in.spines.values():
        sp.set_visible(False)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    ax_in.set_xlim(0, 1); ax_in.set_ylim(0, 5.8)

    _panel_label(ax_in, "B", x=-0.06, y=1.06)
    ax_in.set_title("Model inputs  (83 features)", loc="left",
                    fontsize=8, fontweight="bold", pad=5, x=0.04)

    categories = [
        (W["skyblue"],  "Vital signs",  "HR, BP, SpO2, RR, Temp, GCS"),
        (W["green"],    "Laboratory",   "Lactate, creatinine, WBC, INR, +7 more"),
        (W["orange"],   "Medications",  "Vasopressors, antibiotics, anticoagulants"),
        (W["purple"],   "Temporal",     "Hours since admission, admission type"),
        (W["gray"],     "Missingness",  "Whether each test was ordered"),
    ]

    for i, (col, cat, detail) in enumerate(categories):
        row_y = 5.2 - i * 1.05
        ax_in.plot(0.03, row_y, "s", color=col, ms=7, clip_on=False)
        ax_in.text(0.10, row_y + 0.13, cat, va="bottom", fontsize=7.5,
                   fontweight="bold", color=W["black"])
        ax_in.text(0.10, row_y - 0.11, detail, va="top", fontsize=6.5,
                   color=W["gray"])

    # ── Panel C: output / performance ─────────────────────────────────────────
    for sp in ax_ou.spines.values():
        sp.set_visible(False)
    ax_ou.set_xticks([]); ax_ou.set_yticks([])
    ax_ou.set_xlim(0, 1); ax_ou.set_ylim(0, 5.8)

    _panel_label(ax_ou, "C", x=-0.06, y=1.06)
    ax_ou.set_title("Model output at threshold 0.31",
                    loc="left", fontsize=8, fontweight="bold", pad=5, x=0.04)

    stats = [
        (W["blue"],    "AUROC",              "0.758  (95% CI 0.755-0.761)"),
        (W["green"],   "Sensitivity",        "90%  — catches 9 in 10 events"),
        (W["vermil"],  "False alert rate",   "62%  of stable windows flagged"),
        (W["orange"],  "Neg. pred. value",   "99.5%  — silence is reassuring"),
        (W["purple"],  "Gain over NEWS2",    "+0.190 AUROC  (p < 0.0001)"),
    ]

    for i, (col, label, val) in enumerate(stats):
        row_y = 5.2 - i * 1.05
        ax_ou.plot(0.03, row_y, "s", color=col, ms=7, clip_on=False)
        ax_ou.text(0.10, row_y + 0.13, label, va="bottom", fontsize=7.5,
                   fontweight="bold", color=W["black"])
        ax_ou.text(0.10, row_y - 0.11, val, va="top", fontsize=6.5,
                   color=W["gray"])

    fig.savefig(out_dir / "fig1_study_design.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved fig1_study_design.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Model discrimination
#   Panel A: ROC curves    Panel B: AUROC bar comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig2_discrimination():
    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                          constrained_layout=True)
    fig.patch.set_facecolor("white")

    y        = oof["label"].values
    xgb_prob = oof["xgb_proba"].values
    lr_prob  = oof["lr_proba"].values
    news2    = oof["news2_total"].values if "news2_total" in oof.columns else None

    # ── Panel A: ROC curves ───────────────────────────────────────────────────
    _minimal_spine(ax_roc)
    _panel_label(ax_roc, "A")
    ax_roc.set_title("Receiver operating characteristic",
                     loc="left", fontsize=9, fontweight="bold", pad=14)

    fpr_x, tpr_x, _ = roc_curve(y, xgb_prob)
    fpr_l, tpr_l, _ = roc_curve(y, lr_prob)

    ax_roc.plot(fpr_x, tpr_x, color=W["blue"],   lw=1.8,
                label="bssentinel XGBoost  (AUROCu00a0=u00a00.758, 95%u00a0CI 0.755-0.761)")
    ax_roc.plot(fpr_l, tpr_l, color=W["purple"], lw=1.4, linestyle="--",
                label="Logistic regression  (AUROCu00a0=u00a00.708, 95%u00a0CI 0.705-0.711)")

    if news2 is not None:
        valid = ~np.isnan(news2)
        fpr_n, tpr_n, _ = roc_curve(y[valid], news2[valid])
        ax_roc.plot(fpr_n, tpr_n, color=W["orange"], lw=1.4, linestyle=":",
                    label="NEWS2 score only  (AUROCu00a0=u00a00.568)")

    ax_roc.plot([0, 1], [0, 1], "--", color=W["gray"], lw=0.8, alpha=0.6,
                label="No discrimination  (AUROCu00a0=u00a00.500)")

    # Operating point — annotate in upper-left area to avoid legend
    hs     = metrics["high_sensitivity_point"]
    op_fpr = 1 - hs["specificity"]
    op_tpr = hs["sensitivity"]
    ax_roc.plot(op_fpr, op_tpr, "o", color=W["vermil"], ms=6, zorder=6,
                label=f"Operating point (90% sensitivity, thresholdu00a0=u00a00.31)")
    ax_roc.annotate("Operating\npoint", xy=(op_fpr, op_tpr),
                    xytext=(op_fpr + 0.12, op_tpr - 0.14),
                    fontsize=6.5, color=W["vermil"],
                    arrowprops=dict(arrowstyle="->", color=W["vermil"],
                                    lw=0.8, mutation_scale=8))

    ax_roc.set_xlabel("1 - Specificity  (false positive rate)")
    ax_roc.set_ylabel("Sensitivity  (true positive rate)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.04)
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.92,
                  edgecolor=W["gray"], handlelength=1.6, labelspacing=0.35,
                  fontsize=6.5)
    ax_roc.grid(True, linewidth=0.4, color=W["lgray"], zorder=0)

    # ── Panel B: AUROC bar chart ───────────────────────────────────────────────
    _minimal_spine(ax_bar)
    _panel_label(ax_bar, "B")
    ax_bar.set_title("AUROC comparison",
                     loc="left", fontsize=9, fontweight="bold", pad=14)

    models = ["NEWS2\n(current standard)", "Logistic\nregression",
              "bssentinel\n(XGBoost)"]
    aucs   = [0.5683, 0.7081, 0.7583]
    cis_lo = [None,   0.7048, 0.7555]
    cis_hi = [None,   0.7112, 0.7612]
    colors = [W["orange"], W["purple"], W["blue"]]

    y_pos = np.arange(len(models))
    bars  = ax_bar.barh(y_pos, aucs, color=colors, height=0.5,
                        edgecolor="white", linewidth=0.5, zorder=2)

    for i, (lo, hi) in enumerate(zip(cis_lo, cis_hi)):
        if lo is not None:
            ax_bar.errorbar(aucs[i], y_pos[i],
                            xerr=[[aucs[i] - lo], [hi - aucs[i]]],
                            fmt="none", color="white", capsize=3, lw=1.2, zorder=3)

    for bar, auc in zip(bars, aucs):
        ax_bar.text(auc + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", va="center", fontsize=7.5,
                    fontweight="bold", color=W["black"])

    ax_bar.axvline(0.5, color=W["gray"], lw=0.8, linestyle="--", zorder=1)
    ax_bar.text(0.502, -0.55, "Chance", fontsize=6.5, color=W["gray"], va="top")

    # AUROC gain annotation — placed above the top bar, away from labels
    gain = 0.7583 - 0.5683
    ax_bar.annotate("", xy=(0.7583, 2.45), xytext=(0.5683, 2.45),
                    arrowprops=dict(arrowstyle="<->", color=W["green"],
                                    lw=1.3, mutation_scale=9))
    ax_bar.text((0.7583 + 0.5683) / 2, 2.58,
                f"+{gain:.2f} AUROC\npu00a0<u00a00.0001 (DeLong)",
                ha="center", fontsize=6.5, color=W["green"], fontweight="bold")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(models, fontsize=7.5)
    ax_bar.set_xlabel("Area under the ROC curve (AUROC)")
    ax_bar.set_xlim(0, 0.94)
    ax_bar.set_ylim(-0.65, 3.2)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75])
    ax_bar.grid(True, axis="x", linewidth=0.4, color=W["lgray"], zorder=0)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.tick_params(left=False)

    fig.savefig(out_dir / "fig2_discrimination.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved fig2_discrimination.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Clinical utility
#   Panel A: confusion matrix per 1,000 windows
#   Panel B: threshold operating characteristics
# ══════════════════════════════════════════════════════════════════════════════

def fig3_clinical_utility():
    fig, (ax_freq, ax_thr) = plt.subplots(1, 2, figsize=(7.2, 3.9),
                                           constrained_layout=True)
    fig.patch.set_facecolor("white")

    # ── Panel A: confusion matrix ──────────────────────────────────────────────
    ax_freq.axis("off")
    ax_freq.set_xlim(0, 10)
    ax_freq.set_ylim(-0.6, 9.2)   # extra bottom margin for footnote

    _panel_label(ax_freq, "A", x=-0.04, y=1.06)
    ax_freq.set_title("Per 1,000 monitoring windows  (thresholdu00a0=u00a00.31)",
                      loc="left", fontsize=9, fontweight="bold", pad=12, x=0.04)

    # Counts (scaled to 1,000 windows from MIMIC rates)
    tp = 18;  fn = 2
    fp = 608; tn = 372
    n_pos = tp + fn
    n_neg = fp + tn
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    col0_x, col1_x, col2_x = 0.0, 3.3, 6.7
    col_w0, col_w1 = 3.1, 3.1
    row0_y, row1_y, row2_y = 7.0, 4.0, 1.0
    row_h0, row_h_data     = 1.4, 2.8

    def cell(ax, x, y, w, h, fc, ec, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="square,pad=0",
                                   facecolor=fc, edgecolor=ec,
                                   linewidth=lw, zorder=2))

    def ctext(ax, x, w, y, h, lines, colors, sizes, weights):
        n = len(lines)
        step = h / (n + 1)
        for i, (txt, col, sz, wt) in enumerate(zip(lines, colors, sizes, weights)):
            ax.text(x + w / 2, y + h - step * (i + 1), txt,
                    ha="center", va="center",
                    fontsize=sz, color=col, fontweight=wt)

    # Header row
    cell(ax_freq, col1_x, row0_y, col_w1, row_h0, "#E8F4FD", W["blue"])
    cell(ax_freq, col2_x, row0_y, col_w1, row_h0, "#FDF3E7", W["orange"])
    ax_freq.text(col1_x + col_w1/2, row0_y + row_h0/2, "ALERT FIRED",
                 ha="center", va="center", fontsize=8, fontweight="bold", color=W["blue"])
    ax_freq.text(col2_x + col_w1/2, row0_y + row_h0/2, "NO ALERT",
                 ha="center", va="center", fontsize=8, fontweight="bold", color=W["orange"])

    # Row labels
    cell(ax_freq, col0_x, row1_y, col_w0, row_h_data, "#FFF1F0", W["vermil"])
    ctext(ax_freq, col0_x, col_w0, row1_y, row_h_data,
          ["Pre-deterioration", f"nu00a0=u00a0{n_pos}  (2%)"],
          [W["vermil"], W["gray"]], [8, 7], ["bold", "normal"])

    cell(ax_freq, col0_x, row2_y, col_w0, row_h_data, "#F0FBF7", W["green"])
    ctext(ax_freq, col0_x, col_w0, row2_y, row_h_data,
          ["Patient-stable", f"nu00a0=u00a0{n_neg}  (98%)"],
          [W["green"], W["gray"]], [8, 7], ["bold", "normal"])

    # Data cells
    cell(ax_freq, col1_x, row1_y, col_w1, row_h_data, "#FFF1F0", W["vermil"], lw=1.5)
    ctext(ax_freq, col1_x, col_w1, row1_y, row_h_data,
          [str(tp), "True positive", "Caught"],
          [W["vermil"], W["black"], W["gray"]], [20, 8, 6.5], ["bold", "bold", "normal"])

    cell(ax_freq, col2_x, row1_y, col_w1, row_h_data, "#FFF8F0", W["orange"], lw=1.5)
    ctext(ax_freq, col2_x, col_w1, row1_y, row_h_data,
          [str(fn), "False negative", "Missed"],
          [W["orange"], W["black"], W["gray"]], [20, 8, 6.5], ["bold", "bold", "normal"])

    cell(ax_freq, col1_x, row2_y, col_w1, row_h_data, "#FFF8F0", W["orange"], lw=1.5)
    ctext(ax_freq, col1_x, col_w1, row2_y, row_h_data,
          [str(fp), "False positive", "False alarm"],
          [W["orange"], W["black"], W["gray"]], [20, 8, 6.5], ["bold", "bold", "normal"])

    cell(ax_freq, col2_x, row2_y, col_w1, row_h_data, "#F0FBF7", W["green"], lw=1.5)
    ctext(ax_freq, col2_x, col_w1, row2_y, row_h_data,
          [str(tn), "True negative", "Correctly silent"],
          [W["green"], W["black"], W["gray"]], [20, 8, 6.5], ["bold", "bold", "normal"])

    # PPV / NPV below grid — well-spaced to avoid overlap
    ax_freq.text(col1_x + col_w1/2, row2_y - 0.18,
                 f"PPVu00a0=u00a0{ppv:.1%}  (1 in 35 alerts is a true event)",
                 ha="center", va="top", fontsize=7,
                 fontweight="bold", color=W["orange"])
    ax_freq.text(col2_x + col_w1/2, row2_y - 0.18,
                 f"NPVu00a0=u00a0{npv:.1%}  (silence is highly reassuring)",
                 ha="center", va="top", fontsize=7,
                 fontweight="bold", color=W["green"])

    # Footnote
    ax_freq.text(5.0, -0.45,
                 "Counts scaled to 1,000 windows from MIMIC-IV event rate (1.96%).",
                 ha="center", va="top", fontsize=5.5, color=W["gray"],
                 style="italic")

    # ── Panel B: threshold characteristics ───────────────────────────────────
    _minimal_spine(ax_thr)
    _panel_label(ax_thr, "B", x=-0.16)
    ax_thr.set_title("Sensitivity-specificity tradeoff by threshold",
                     loc="left", fontsize=9, fontweight="bold", pad=14)

    thr_data = metrics["threshold_analysis"]
    ths  = np.array([r["threshold"]   for r in thr_data])
    sens = np.array([r["sensitivity"] for r in thr_data])
    spec = np.array([r["specificity"] for r in thr_data])
    ppv_arr = np.array([r["ppv"]      for r in thr_data])

    ax_thr.plot(ths, sens * 100, "o-", color=W["blue"],   lw=1.4, ms=4,
                label="Sensitivity")
    ax_thr.plot(ths, spec * 100, "s-", color=W["green"],  lw=1.4, ms=4,
                label="Specificity")
    ax_thr.plot(ths, ppv_arr * 100, "^-", color=W["vermil"], lw=1.4, ms=4,
                label="PPV")

    ax_thr.axvline(0.31, color=W["orange"], lw=1.0, linestyle="--", zorder=1)
    ax_thr.text(0.315, 55, "Recommended\nthreshold\n(0.31)",
                fontsize=6.5, color=W["orange"], va="center")

    ax_thr.set_xlabel("Decision threshold")
    ax_thr.set_ylabel("Metric value (%)")
    ax_thr.set_xlim(0.08, 0.53)
    ax_thr.set_ylim(0, 108)
    ax_thr.set_yticks([0, 25, 50, 75, 100])
    ax_thr.legend(loc="upper right", frameon=True, framealpha=0.92,
                  edgecolor=W["gray"], handlelength=1.5, fontsize=7)
    ax_thr.grid(True, linewidth=0.4, color=W["lgray"], zorder=0)

    fig.savefig(out_dir / "fig3_clinical_utility.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved fig3_clinical_utility.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Feature importance (top 15)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "lactate_missing":                      "Lactate not measured (informative absence)",
    "spo2_worst":                           "Minimum SpO2 in window",
    "gcs_verbal_worst":                     "Worst verbal response (GCS)",
    "gcs_motor_worst":                      "Worst motor response (GCS)",
    "news2_gcs":                            "NEWS2 consciousness subscore",
    "gcs_motor":                            "GCS motor score (most recent)",
    "respiratory_rate_worst":               "Highest respiratory rate in window",
    "admit_offset_hours":                   "Time since hospital admission (h)",
    "vasopressors":                         "Vasopressor therapy active",
    "news2_total":                          "Total NEWS2 score",
    "adm_type_OBSERVATION ADMIT":           "Admission type: observation",
    "adm_type_SURGICAL SAME DAY ADMISSION": "Admission type: same-day surgical",
    "antibiotics":                          "Antibiotic therapy active",
    "anticoagulants":                       "Anticoagulant therapy active",
    "gcs_total":                            "Glasgow Coma Scale total",
    "news2_hr":                             "NEWS2 heart rate subscore",
    "lactate":                              "Lactate concentration",
}

CATEGORY = {
    "lactate_missing":              ("Laboratory",         W["green"]),
    "spo2_worst":                   ("Respiratory/SpO2", W["skyblue"]),
    "gcs_verbal_worst":             ("Neurology/GCS",     W["purple"]),
    "gcs_motor_worst":              ("Neurology/GCS",     W["purple"]),
    "news2_gcs":                    ("Neurology/GCS",     W["purple"]),
    "respiratory_rate_worst":       ("Respiratory/SpO2", W["skyblue"]),
    "admit_offset_hours":           ("Clinical context",  W["gray"]),
    "vasopressors":                 ("Medications",       W["orange"]),
    "news2_total":                  ("Clinical context",  W["gray"]),
    "adm_type_OBSERVATION ADMIT":   ("Clinical context",  W["gray"]),
    "antibiotics":                  ("Medications",       W["orange"]),
    "anticoagulants":               ("Medications",       W["orange"]),
    "gcs_total":                    ("Neurology/GCS",     W["purple"]),
    "news2_hr":                     ("Clinical context",  W["gray"]),
    "lactate":                      ("Laboratory",        W["green"]),
}


def fig4_feature_importance():
    top_n = 15
    top = (fi[fi["importance"] > 0]
           .nlargest(top_n, "importance")
           .reset_index(drop=True))

    top["label"] = top["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    top["cat"]   = top["feature"].map(lambda f: CATEGORY.get(f, ("Other", W["gray"]))[0])
    top["color"] = top["feature"].map(lambda f: CATEGORY.get(f, ("Other", W["gray"]))[1])
    top["pct"]   = top["importance"] * 100
    top = top.iloc[::-1].reset_index(drop=True)   # bottom-to-top

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _minimal_spine(ax)
    _panel_label(ax, "A", x=-0.01, y=1.04)
    ax.set_title("Top 15 predictors — bssentinel XGBoost model",
                 loc="left", fontsize=9, fontweight="bold", pad=14, x=0.03)

    bars = ax.barh(np.arange(top_n), top["pct"], color=top["color"],
                   height=0.68, edgecolor="white", linewidth=0.5, zorder=2)

    for bar, val in zip(bars, top["pct"]):
        ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7, color=W["black"])

    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(top["label"], fontsize=7.5)
    ax.set_xlabel("Feature importance  (% of XGBoost gain, sum-normalised)")
    ax.set_xlim(0, 9.5)
    ax.grid(True, axis="x", linewidth=0.4, color=W["lgray"], zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Legend — unique categories only
    seen = {}
    for _, row in top.iterrows():
        if row["cat"] not in seen:
            seen[row["cat"]] = row["color"]
    handles = [mpatches.Patch(facecolor=c, label=cat, edgecolor="white")
               for cat, c in seen.items()]
    ax.legend(handles=handles, title="Feature category", title_fontsize=7.5,
              fontsize=7, loc="lower right", frameon=True,
              edgecolor=W["gray"], framealpha=0.95,
              handlelength=1.2, labelspacing=0.4)

    # Callout box for #1 feature
    ax.annotate(
        "Absence of lactate\nmeasurement is the\nstrongest predictor\n(informative missingness)",
        xy=(top.iloc[-1]["pct"], top_n - 1),
        xytext=(6.5, top_n - 5.0),
        fontsize=6.5, color=W["green"],
        va="center", ha="left",
        arrowprops=dict(arrowstyle="->", color=W["green"],
                        lw=0.9, mutation_scale=8,
                        connectionstyle="arc3,rad=-0.25"),
        bbox=dict(facecolor="#F0FBF7", edgecolor=W["green"],
                  boxstyle="round,pad=0.35", lw=0.8),
    )

    fig.savefig(out_dir / "fig4_feature_importance.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved fig4_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Decision Curve Analysis
#   Panel A: full 0.5–8% clinical range
#   Panel B: zoomed 1–5% sweet spot
# ══════════════════════════════════════════════════════════════════════════════

def _isotonic_calibrate(y, scores):
    ir = IsotonicRegression(out_of_bounds="clip")
    return ir.fit_transform(scores, y.astype(float))

def _logistic_calibrate(y, scores):
    X = scores.reshape(-1, 1)
    sc = StandardScaler()
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(sc.fit_transform(X), y)
    return lr.predict_proba(sc.transform(X))[:, 1]

def _net_benefit(y, p_cal, thresholds):
    n = len(y)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = p_cal >= t
        tp   = np.sum(pred & (y == 1))
        fp   = np.sum(pred & (y == 0))
        nb[i] = tp / n - fp / n * (t / (1.0 - t + 1e-15))
    return nb

def _nb_treat_all(y, thresholds):
    prev = y.mean()
    return prev - (1.0 - prev) * thresholds / (1.0 - thresholds + 1e-15)


def fig5_dca():
    y         = oof["label"].values.astype(int)
    p_xgb_raw = oof["xgb_proba"].values.astype(float)
    news2_raw = oof["news2_total"].fillna(0).values.astype(float)

    p_xgb  = _isotonic_calibrate(y, p_xgb_raw)
    p_news = _logistic_calibrate(y, news2_raw)

    t_clin = np.linspace(0.005, 0.08, 300)
    t_pct  = t_clin * 100

    nb_xgb  = _net_benefit(y, p_xgb,  t_clin)
    nb_news = _net_benefit(y, p_news,  t_clin)
    nb_all  = _nb_treat_all(y, t_clin)
    nb_none = np.zeros(len(t_clin))

    # NEWS2 ≥ 5 binary rule curve
    flag5 = (news2_raw >= 5)
    tp5   = np.sum(flag5 & (y == 1))
    fp5   = np.sum(flag5 & (y == 0))
    n     = len(y)
    nb_news5 = np.array([tp5/n - fp5/n * (t/(1-t+1e-15)) for t in t_clin])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                    constrained_layout=True)
    fig.patch.set_facecolor("white")

    def _style(ax):
        _minimal_spine(ax)
        ax.grid(True, linewidth=0.4, color=W["lgray"], zorder=0)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}%"))

    # ── Panel A: full range 0.5–8% ────────────────────────────────────────────
    _style(ax1)
    _panel_label(ax1, "A")
    ax1.set_title("Decision curve analysis  (0.5--8% threshold range)",
                  loc="left", fontsize=9, fontweight="bold", pad=14)

    ax1.plot(t_pct, nb_xgb,    color=W["blue"],   lw=2.0,
             label="bssentinel (XGBoost)")
    ax1.plot(t_pct, nb_news,   color=W["orange"], lw=1.6, ls="--",
             label="NEWS2 (calibrated)")
    ax1.plot(t_pct, nb_news5,  color=W["orange"], lw=1.2, ls=":",
             label="NEWS2u00a0>=u00a05 (binary rule)")
    ax1.plot(t_pct, nb_all,    color=W["green"],  lw=1.4, ls=(0, (4, 2)),
             label="Treat-all")
    ax1.axhline(0, color=W["gray"], lw=0.9, label="Treat-none")

    # Shade bssentinel advantage
    adv = nb_xgb > np.maximum(nb_news, 0)
    ax1.fill_between(t_pct[adv], 0, nb_xgb[adv],
                     alpha=0.10, color=W["blue"])

    ax1.set_xlabel("Threshold probability")
    ax1.set_ylabel("Net benefit (per patient)")
    ax1.set_xlim(0.5, 8.0)
    ypad = abs(min(nb_all.min(), nb_news.min())) * 0.15
    ax1.set_ylim(min(nb_all.min(), nb_news.min()) - ypad, None)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92,
               edgecolor=W["gray"], handlelength=1.6, fontsize=6.5,
               labelspacing=0.35)

    # ── Panel B: zoomed 1–5% ──────────────────────────────────────────────────
    _style(ax2)
    _panel_label(ax2, "B")
    ax2.set_title("Zoomed view  (1-5% clinical sweet spot)",
                  loc="left", fontsize=9, fontweight="bold", pad=14)

    mask = (t_clin >= 0.01) & (t_clin <= 0.05)
    tz   = t_pct[mask]

    ax2.plot(tz, nb_xgb[mask],  color=W["blue"],   lw=2.2,
             label="bssentinel")
    ax2.plot(tz, nb_news[mask], color=W["orange"], lw=1.8, ls="--",
             label="NEWS2 (calibrated)")
    ax2.plot(tz, nb_all[mask],  color=W["green"],  lw=1.4, ls=(0, (4, 2)),
             label="Treat-all")
    ax2.axhline(0, color=W["gray"], lw=0.9, label="Treat-none")

    # Fill bssentinel > NEWS2 region
    adv2 = nb_xgb[mask] > nb_news[mask]
    ax2.fill_between(tz[adv2],
                     nb_news[mask][adv2],
                     nb_xgb[mask][adv2],
                     alpha=0.18, color=W["blue"],
                     label="bssentinel advantage")

    # Annotate ΔNB at 2%
    t2i   = np.argmin(np.abs(t_clin[mask] - 0.02))
    nb_x2 = nb_xgb[mask][t2i]
    nb_n2 = nb_news[mask][t2i]
    if nb_x2 > nb_n2:
        mid = (nb_x2 + nb_n2) / 2
        ax2.annotate(
            f"DeltaNBu00a0=u00a0{nb_x2 - nb_n2:.4f}\n(per patient at 2%)",
            xy=(2.0, mid),
            xytext=(3.2, mid + abs(nb_x2 - nb_n2) * 1.2),
            fontsize=7, color=W["blue"],
            arrowprops=dict(arrowstyle="->", color=W["blue"], lw=0.9,
                            mutation_scale=8),
            bbox=dict(facecolor="#EEF4FF", edgecolor=W["blue"],
                      boxstyle="round,pad=0.3", lw=0.7),
        )

    ax2.set_xlabel("Threshold probability")
    ax2.set_ylabel("Net benefit (per patient)")
    ax2.set_xlim(1, 5)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92,
               edgecolor=W["gray"], handlelength=1.6, fontsize=6.5,
               labelspacing=0.35)

    fig.savefig(out_dir / "fig5_dca.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved fig5_dca.png")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent)
    print("Generating publication-quality figures...")
    fig1_study_design()
    fig2_discrimination()
    fig3_clinical_utility()
    fig4_feature_importance()
    fig5_dca()
    print("Done.")
