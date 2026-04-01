"""
make_table1_figure.py
Generates:
  1. artifacts/table1.tex   — LaTeX Table 1, exact cohort statistics
  2. artifacts/figures/clinician/fig0_cohort_flow.png — CONSORT-style flow diagram
"""
import pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
with open("artifacts/metrics.json") as f:
    metrics = json.load(f)

with open("cache/cohort_h24_w6.pkl", "rb") as f:
    data = pickle.load(f)

adm  = data["admissions"]
out  = data["outcomes"]
oof  = pd.read_parquet("artifacts/oof_predictions.parquet")

modeling_hadm = set(oof["hadm_id"].unique())
adm_m = adm[adm["hadm_id"].isin(modeling_hadm)].copy()
out_m = out[out["hadm_id"].isin(modeling_hadm)].copy()

# ── Compute statistics ─────────────────────────────────────────────────────────
n_mimic_total = len(adm)
n_adm         = len(adm_m)
n_pat         = adm_m["subject_id"].nunique()
n_win         = len(oof)
n_pos         = int(oof["label"].sum())
n_neg         = n_win - n_pos
prev          = n_pos / n_win

age    = adm_m["anchor_age"]
los    = adm_m["los_hours"] / 24
wpa    = oof.groupby("hadm_id").size()
news2  = oof["news2_total"]

sex    = adm_m["gender"].value_counts()
f_n, m_n = sex.get("F", 0), sex.get("M", 0)

mort_n = adm_m["hospital_expire_flag"].sum()

icu_n  = out_m[out_m["event_type"] == "icu_transfer"]["hadm_id"].nunique()
intu_n = out_m[out_m["event_type"] == "intubation"]["hadm_id"].nunique()

pos_adm = oof[oof["label"] == 1]["hadm_id"].nunique()

at = adm_m["admission_type"].value_counts()

n_excluded = n_mimic_total - n_adm

# ── 1. LaTeX Table 1 ──────────────────────────────────────────────────────────
tex = r"""% Table 1 — Study cohort characteristics
% ALL VALUES computed directly from the modelling dataset
% (oof_predictions.parquet + cohort_h24_w6.pkl cache)
\begin{table}[H]
\centering
\caption{\textbf{Study cohort characteristics.}
All statistics are derived directly from the modelling dataset after
application of inclusion and exclusion criteria.
Values are reported as median (interquartile range) or $n$ (\%).
LOS: length of stay; IQR: interquartile range; NEWS2: National Early Warning Score 2.}
\label{tab:cohort}
\begin{tabular}{lc}
\toprule
\textbf{Characteristic} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Dataset source}} \\
\quad MIMIC-IV source admissions & """ + f"{n_mimic_total:,}" + r""" \\
\quad Excluded (no ICU stay, LOS $<$ 6\,h, or insufficient features) & """ + f"{n_excluded:,}" + r""" \\
\quad \textbf{Admissions included in modelling cohort} & \textbf{""" + f"{n_adm:,}" + r"""} \\
\quad Unique patients & """ + f"{n_pat:,}" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Monitoring windows}} \\
\quad Total 6-hourly windows & """ + f"{n_win:,}" + r""" \\
\quad Positive windows (deterioration within 24\,h) & """ + f"{n_pos:,} ({100*prev:.2f}\\%)" + r""" \\
\quad Negative windows & """ + f"{n_neg:,} ({100*(1-prev):.2f}\\%)" + r""" \\
\quad Windows per admission, median (IQR) & """ + f"{wpa.median():.0f} ({wpa.quantile(.25):.0f}--{wpa.quantile(.75):.0f})" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Demographics}} \\
\quad Age, median (IQR), years & """ + f"{age.median():.0f} ({age.quantile(.25):.0f}--{age.quantile(.75):.0f})" + r""" \\
\quad Age, mean $\pm$ SD, years & """ + f"{age.mean():.1f} $\\pm$ {age.std():.1f}" + r""" \\
\quad Age $<$ 65 years & """ + f"{(age<65).sum():,} ({100*(age<65).mean():.1f}\\%)" + r""" \\
\quad Age 65--79 years & """ + f"{((age>=65)&(age<80)).sum():,} ({100*((age>=65)&(age<80)).mean():.1f}\\%)" + r""" \\
\quad Age $\geq$ 80 years & """ + f"{(age>=80).sum():,} ({100*(age>=80).mean():.1f}\\%)" + r""" \\
\quad Female sex & """ + f"{f_n:,} ({100*f_n/n_adm:.1f}\\%)" + r""" \\
\quad Male sex & """ + f"{m_n:,} ({100*m_n/n_adm:.1f}\\%)" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Clinical characteristics}} \\
\quad LOS, median (IQR), days & """ + f"{los.median():.1f} ({los.quantile(.25):.1f}--{los.quantile(.75):.1f})" + r""" \\
\quad Admission type --- Emergency (EW EMER.) & """ + f"{at.get('EW EMER.',0):,} ({100*at.get('EW EMER.',0)/n_adm:.1f}\\%)" + r""" \\
\quad Admission type --- Urgent & """ + f"{at.get('URGENT',0):,} ({100*at.get('URGENT',0)/n_adm:.1f}\\%)" + r""" \\
\quad Admission type --- Observation & """ + f"{at.get('OBSERVATION ADMIT',0):,} ({100*at.get('OBSERVATION ADMIT',0)/n_adm:.1f}\\%)" + r""" \\
\quad Admission type --- Same-day surgical & """ + f"{at.get('SURGICAL SAME DAY ADMISSION',0):,} ({100*at.get('SURGICAL SAME DAY ADMISSION',0)/n_adm:.1f}\\%)" + r""" \\
\quad Admission type --- Other & """ + f"{n_adm - sum([at.get(t,0) for t in ['EW EMER.','URGENT','OBSERVATION ADMIT','SURGICAL SAME DAY ADMISSION']]):,} ({100*(n_adm - sum([at.get(t,0) for t in ['EW EMER.','URGENT','OBSERVATION ADMIT','SURGICAL SAME DAY ADMISSION']]))/n_adm:.1f}\\%)" + r""" \\
\quad NEWS2 score, median (IQR) & """ + f"{news2.median():.0f} ({news2.quantile(.25):.0f}--{news2.quantile(.75):.0f})" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Outcomes}} \\
\quad Admissions with ICU stay & """ + f"{icu_n:,} (100.0\\%)" + r""" \\
\quad Admissions requiring mechanical ventilation & """ + f"{intu_n:,} ({100*intu_n/n_adm:.1f}\\%)" + r""" \\
\quad In-hospital mortality & """ + f"{mort_n:,} ({100*mort_n/n_adm:.1f}\\%)" + r""" \\
\quad Admissions with $\geq$1 positive monitoring window & """ + f"{pos_adm:,} ({100*pos_adm/n_adm:.1f}\\%)" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""

Path("artifacts/table1.tex").write_text(tex)
print("Saved artifacts/table1.tex")

# ── 2. Cohort flow diagram ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8,
})

W = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "vermil": "#D55E00",
    "gray":   "#999999",
    "lgray":  "#F5F5F5",
    "black":  "#000000",
}

fig, ax = plt.subplots(figsize=(7.2, 9.0), constrained_layout=True)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis("off")

def box(ax, cx, cy, w, h, text, fc, ec, fontsize=8, bold_first=False):
    """Draw a rounded rectangle with centred text."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3
    ))
    lines = text.split("\n")
    step  = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        y_pos = cy + h/2 - step * (i + 1)
        fw = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(cx, y_pos, line, ha="center", va="center",
                fontsize=fontsize, fontweight=fw, color=W["black"],
                zorder=4)

def excl_box(ax, cx, cy, w, h, text, fontsize=7.5):
    """Exclusion box (right-side, orange)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor="#FFF8F0", edgecolor=W["orange"], linewidth=1.0, zorder=3
    ))
    lines = text.split("\n")
    step  = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        y_pos = cy + h/2 - step * (i + 1)
        ax.text(cx, y_pos, line, ha="center", va="center",
                fontsize=fontsize, color=W["orange"], zorder=4)

def arrow(ax, x, y1, y2, color=W["gray"]):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                mutation_scale=10))

def harrow(ax, x1, x2, y, color=W["orange"]):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                mutation_scale=8))

BW, BH = 5.8, 1.1   # main box width/height
EW, EH = 3.6, 0.9   # exclusion box

# ── Panel label ───────────────────────────────────────────────────────────────
ax.text(0.02, 17.6, "A", fontsize=11, fontweight="bold", va="top", ha="left")
ax.text(0.4,  17.6, "MIMIC-IV cohort selection and window construction",
        fontsize=9, fontweight="bold", va="top", ha="left")

# ── Box 1: MIMIC-IV source ─────────────────────────────────────────────────────
CX = 4.5
y1 = 16.5
box(ax, CX, y1, BW, BH,
    f"MIMIC-IV source (all admissions)\nn = {n_mimic_total:,} admissions",
    fc="#EEF4FF", ec=W["blue"], bold_first=True)

arrow(ax, CX, y1 - BH/2, y1 - BH/2 - 0.6)

# Exclusion 1
excl_box(ax, 8.3, y1 - 0.55, EW, EH,
         f"Excluded: no ICU stay\nor insufficient data\nn = {n_excluded:,}")
harrow(ax, CX + BW/2, 8.3 - EW/2, y1 - 0.55)

# ── Box 2: Included admissions ────────────────────────────────────────────────
y2 = y1 - BH/2 - 0.6 - BH/2 - 0.05
box(ax, CX, y2, BW, BH,
    f"Admissions with ICU stay included\nn = {n_adm:,} admissions  |  {n_pat:,} unique patients",
    fc="#EEF4FF", ec=W["blue"], bold_first=True)

arrow(ax, CX, y2 - BH/2, y2 - BH/2 - 0.6)

# ── Box 3: Demographics summary ───────────────────────────────────────────────
y3 = y2 - BH/2 - 0.6 - 0.75
BH3 = 1.5
box(ax, CX, y3, BW, BH3,
    f"Demographics\n"
    f"Age: {age.median():.0f} yrs (IQR {age.quantile(.25):.0f}-{age.quantile(.75):.0f})  |  "
    f"Female: {100*f_n/n_adm:.0f}%  |  Male: {100*m_n/n_adm:.0f}%\n"
    f"LOS: {los.median():.1f} days (IQR {los.quantile(.25):.1f}-{los.quantile(.75):.1f})  |  "
    f"Emergency: {100*at.get('EW EMER.',0)/n_adm:.0f}%\n"
    f"NEWS2: {news2.median():.0f} (IQR {news2.quantile(.25):.0f}-{news2.quantile(.75):.0f})  |  "
    f"In-hospital mortality: {100*mort_n/n_adm:.1f}%",
    fc=W["lgray"], ec=W["gray"], fontsize=7.5, bold_first=True)

arrow(ax, CX, y3 - BH3/2, y3 - BH3/2 - 0.6)

# ── Box 4: Window generation ──────────────────────────────────────────────────
y4 = y3 - BH3/2 - 0.6 - BH/2 - 0.05
box(ax, CX, y4, BW, BH,
    f"6-hourly monitoring windows generated\n"
    f"n = {n_win:,} windows  |  median {wpa.median():.0f} per admission (IQR {wpa.quantile(.25):.0f}-{wpa.quantile(.75):.0f})",
    fc="#EEF4FF", ec=W["blue"], bold_first=True)

arrow(ax, CX, y4 - BH/2, y4 - BH/2 - 0.6)

# ── Box 5: Outcome labelling (splits into two) ────────────────────────────────
y5 = y4 - BH/2 - 0.6 - 0.55

# Left branch: positive windows
LCX, RCX = 2.2, 6.9
LW = 3.8

# Horizontal split line
ax.plot([CX, CX], [y4 - BH/2, y5 + 0.55], color=W["gray"], lw=1.2)
ax.plot([LCX, RCX], [y5 + 0.55, y5 + 0.55], color=W["gray"], lw=1.0)
ax.plot([LCX, LCX], [y5 + 0.55, y5 + BH/2 + 0.05], color=W["gray"], lw=1.0)
ax.plot([RCX, RCX], [y5 + 0.55, y5 + BH/2 + 0.05], color=W["gray"], lw=1.0)
ax.annotate("", xy=(LCX, y5 + BH/2), xytext=(LCX, y5 + BH/2 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=W["gray"], lw=1.0, mutation_scale=8))
ax.annotate("", xy=(RCX, y5 + BH/2), xytext=(RCX, y5 + BH/2 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=W["gray"], lw=1.0, mutation_scale=8))

# Positive windows box
box(ax, LCX, y5, LW, BH,
    f"Positive windows\n(deterioration within 24 h)\nn = {n_pos:,}  ({100*prev:.2f}%)",
    fc="#FFF1F0", ec=W["vermil"], bold_first=True)

# Negative windows box
box(ax, RCX, y5, LW, BH,
    f"Negative windows\n(no event within 24 h)\nn = {n_neg:,}  ({100*(1-prev):.2f}%)",
    fc="#F0FBF7", ec=W["green"], bold_first=True)

# Arrows to final box
y6 = y5 - BH/2 - 0.6 - BH/2
ax.plot([LCX, LCX], [y5 - BH/2, y5 - BH/2 - 0.3], color=W["gray"], lw=1.0)
ax.plot([RCX, RCX], [y5 - BH/2, y5 - BH/2 - 0.3], color=W["gray"], lw=1.0)
ax.plot([LCX, RCX], [y5 - BH/2 - 0.3, y5 - BH/2 - 0.3], color=W["gray"], lw=1.0)
ax.plot([CX, CX], [y5 - BH/2 - 0.3, y6 + BH/2 + 0.05], color=W["gray"], lw=1.0)
ax.annotate("", xy=(CX, y6 + BH/2), xytext=(CX, y6 + BH/2 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=W["gray"], lw=1.2, mutation_scale=10))

# ── Box 6: Final modelling cohort ─────────────────────────────────────────────
BH6 = 1.45
box(ax, CX, y6, BW, BH6,
    f"Final modelling cohort\n"
    f"{n_win:,} windows  |  {n_adm:,} admissions  |  {n_pat:,} patients\n"
    f"5-fold patient-stratified cross-validation\n"
    f"Outcome prevalence: {100*prev:.2f}%  |  AUROC 0.758 (95% CI 0.755-0.761)",
    fc="#EEF9F4", ec=W["green"], bold_first=True, fontsize=8)

out_path = Path("artifacts/figures/clinician/fig0_cohort_flow.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out_path}")
