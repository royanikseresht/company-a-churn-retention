"""
GCI World 2026 September - Final Assignment (Company A, telecom)
Churn early-warning model + reason-based retention playbook

HOW TO RUN (Terminal, from the folder that contains the `telecom` folder):
    python telecom_analysis.py

The script looks for telecom/Client.csv and telecom/Record.csv next to it.
Runtime: roughly 10-20 minutes on a laptop (set FAST = True for a 1-2 minute test run).

OUTPUT (folder `results/`, small enough to share):
    summary.txt            every number used in the slides
    fig01 ... fig14 .png   charts for the slides (plus fig02b / fig03b detail charts)
    segment_tables.csv     churn rate by customer segment (reference)

Requirements: pandas, numpy, scikit-learn, matplotlib, lightgbm  (xgboost optional)

Sources / citation:
  - Per-customer explanations use TreeSHAP via LightGBM's built-in `pred_contrib`
    (Lundberg et al., 2020, "From local explanations to global understanding with
    explainable AI for trees", Nature Machine Intelligence 2, 56-67).
  - Prior-shift correction of predicted probabilities follows Saerens et al., 2002,
    "Adjusting the outputs of a classifier to new a priori probabilities", Neural Computation 14(1).
"""
import sys, time, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("LightGBM is required. Install it with:  pip install lightgbm")
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# =============================== SETTINGS ===============================
SEED = 42
N_FOLDS = 5
FAST = False          # True = quick test on a 20,000-row sample
TOP_SHARE = 0.20      # "high-risk group" = top 20% of customers by predicted churn risk

# Fairness: features that must not drive who receives offers. They are removed from all models;
# FAIRNESS_CHECK re-trains LightGBM once WITH them to show how little accuracy is lost.
EXCLUDE_FEATURES = ["ethnic"]
FAIRNESS_CHECK = True

# ---- Business assumptions for the ROI slide (US market, sources in comments) ----
ASSUME = dict(
    # Monthly churn. The dataset is a ~50/50 sample, so a real-world rate is needed.
    # Big 3 postpaid phone churn Q2 2026: AT&T 0.86%, Verizon 0.92%, T-Mobile 0.99% (account)
    #   [TelecomLead, 29 Sep 2026]. Cable operators (Comcast, Charter) estimated near 2%/month
    #   [R. Entner, Recon Analytics, in Light Reading, 31 Jul 2026].
    # Company A is a smaller operator without in-house analytics -> 2% base case, 1% / 3% in sensitivity.
    BASE_CHURN_RATE=0.02,
    # Outbound call-centre labour $10-$50 per hour [Twilio, Nov 2025] -> ~10 minutes per retention call ~ $5.
    CONTACT_COST=5.0,
    # Offer value paid only if accepted, e.g. $10/month bill credit for 6 months (assumption).
    INCENTIVE_COST=60.0,
    # Share of contacted would-be churners who stay (assumption, tested 15-35%).
    # Cross-check: McKinsey reports analytics-led base management cutting churn by up to 15% (Jain & Surana, Dec 2017).
    SAVE_RATE=0.25,
    TAKEUP_LOYAL=0.10,       # share of loyal customers contacted who also take the offer (wasted incentive; assumption)
    HORIZON_MONTHS=12,       # months of revenue counted per saved customer (conservative; US device credits run 36 months)
    CUSTOMER_BASE=100_000,   # results are reported per this many customers
)
# =========================================================================

ID, TARGET = "Customer_ID", "churn"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)
for old in OUT.glob("fig*.png"):
    old.unlink()
T0 = time.time()
SUMMARY = []


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def say(line=""):
    print(line, flush=True)
    SUMMARY.append(str(line))


def money(x):
    s = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{s}${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{s}${x / 1e3:.0f}K"
    return f"{s}${x:.0f}"


# ------------------------------ chart style ------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, GRAY = C[0], C[1], "#a8a79f"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def blended(ax):
    """x in axes fraction, y in data units (for labels in the right margin)."""
    return matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)


def titles(ax, title, sub=None):
    """Chart title with an optional subtitle line between the title and the plot."""
    ax.set_title(title, pad=30 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


LABELS = {  # readable names for charts (raw column names stay in summary.txt)
    "eqpdays": "Handset age (days)", "eqp_vs_tenure": "Handset age vs tenure", "hnd_price": "Handset price",
    "hnd_price_vs_rev": "Handset price vs monthly bill", "hnd_webcap": "Handset web capability",
    "refurb_new": "Refurbished handset", "phones": "Handsets issued", "models": "Handset models issued",
    "phones_per_year": "Handsets per year", "dualband": "Dual-band handset",
    "change_mou": "Change in minutes vs 3-mo avg", "change_rev": "Change in revenue vs 3-mo avg",
    "usage_trend_mou": "Minutes trend (% vs 3-mo avg)", "usage_trend_rev": "Revenue trend (% vs 3-mo avg)",
    "mou_recent_vs_life": "Recent vs lifetime minutes", "rev_recent_vs_life": "Recent vs lifetime revenue",
    "qty_recent_vs_life": "Recent vs lifetime calls", "mou_6m_vs_life": "6-month vs lifetime minutes",
    "months": "Months in service", "uniqsubs": "Subscribers in household", "actvsubs": "Active subscribers",
    "subs_ratio": "Active share of subscribers", "new_cell": "New cell phone user",
    "rev_Mean": "Monthly revenue", "totmrc_Mean": "Monthly recurring charge", "ovrmou_Mean": "Overage minutes",
    "ovrrev_Mean": "Overage revenue", "overage_share": "Overage share of bill", "rev_per_min": "Revenue per minute",
    "mrc_share": "Recurring-charge share of bill", "totrev": "Lifetime revenue", "adjrev": "Adjusted lifetime revenue",
    "avgrev": "Avg monthly revenue (lifetime)", "avg3rev": "Avg revenue, last 3 months",
    "avg6rev": "Avg revenue, last 6 months", "roam_Mean": "Roaming calls", "da_Mean": "Directory-assisted calls",
    "vceovr_Mean": "Voice overage revenue", "datovr_Mean": "Data overage revenue",
    "mou_Mean": "Monthly minutes of use", "avg3mou": "Avg minutes, last 3 months", "avg6mou": "Avg minutes, last 6 months",
    "avgmou": "Avg minutes (lifetime)", "avgqty": "Avg calls (lifetime)", "avg3qty": "Avg calls, last 3 months",
    "avg6qty": "Avg calls, last 6 months", "totcalls": "Lifetime calls", "totmou": "Lifetime minutes",
    "adjmou": "Adjusted lifetime minutes", "adjqty": "Adjusted lifetime calls", "recv_sms_Mean": "Received SMS",
    "threeway_Mean": "Three-way calls", "inonemin_Mean": "Inbound calls under 1 min", "complete_Mean": "Completed calls",
    "attempt_Mean": "Attempted calls", "peak_vce_Mean": "Peak voice calls", "opk_vce_Mean": "Off-peak voice calls",
    "mou_peav_Mean": "Peak voice minutes", "mou_opkv_Mean": "Off-peak voice minutes",
    "mou_cvce_Mean": "Completed voice minutes", "mou_rvce_Mean": "Received voice minutes",
    "recv_vce_Mean": "Received voice calls", "owylis_vce_Mean": "Outbound wireless-to-wireless calls",
    "iwylis_vce_Mean": "Inbound wireless-to-wireless calls", "mouowylisv_Mean": "Outbound wireless-to-wireless min",
    "mouiwylisv_Mean": "Inbound wireless-to-wireless min", "callwait_Mean": "Call-waiting calls",
    "drop_vce_Mean": "Dropped voice calls", "blck_vce_Mean": "Blocked voice calls",
    "unan_vce_Mean": "Unanswered voice calls", "drop_blk_Mean": "Dropped or blocked calls",
    "drop_rate": "Dropped-call rate", "block_rate": "Blocked-call rate", "complete_rate": "Call completion rate",
    "unanswered_rate": "Unanswered-call rate", "custcare_Mean": "Customer-care calls",
    "ccrndmou_Mean": "Customer-care minutes", "cc_mou_Mean": "Customer-care minutes (exact)",
    "care_per_100min": "Care calls per 100 minutes", "crclscod": "Credit class", "asl_flag": "Spending limit",
    "area": "Geographic area", "prizm_social_one": "Social group", "ethnic": "Ethnicity code",
    "income": "Estimated income", "marital": "Marital status", "adults": "Adults in household",
    "lor": "Length of residence", "creditcd": "Has credit card", "HHstatin": "Household status",
    "dwllsize": "Dwelling size", "dwlltype": "Dwelling type", "ownrent": "Owner or renter",
    "numbcars": "Number of vehicles", "infobase": "InfoBase match", "forgntvl": "Foreign travel",
    "truck": "Owns a truck", "rv": "Owns an RV",
}


def lab(f):
    return LABELS.get(f, f)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved results/{name}")


# =============================== 1. LOAD ===============================
def find_file(name):
    for d in [BASE / "telecom", BASE, BASE.parent / "telecom"]:
        if (d / name).exists():
            return d / name
    for pattern in [f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"]:
        hits = sorted(BASE.glob(pattern))
        if hits:
            return hits[0]
    sys.exit(f"Could not find {name}. Put this script in the folder that contains the `telecom` folder.")


client_path, record_path = find_file("Client.csv"), find_file("Record.csv")
log(f"loading {record_path.parent}")
client = pd.read_csv(client_path, low_memory=False)
record = pd.read_csv(record_path, low_memory=False)
dup = [c for c in client.columns if c in record.columns and c != ID]
df = record.merge(client.drop(columns=dup), on=ID, how="inner")
df = df[df[TARGET].notna()].reset_index(drop=True)
df[TARGET] = df[TARGET].astype(int)
if FAST:
    df = df.sample(n=min(20_000, len(df)), random_state=SEED).reset_index(drop=True)
y = df[TARGET].values
n = len(df)
cat_cols = [c for c in df.columns if c not in (ID, TARGET) and not pd.api.types.is_numeric_dtype(df[c])]
num_raw = [c for c in df.columns if c not in (ID, TARGET) and c not in cat_cols]
log(f"Client {client.shape}, Record {record.shape}, merged {df.shape}")

say("=" * 78)
say("COMPANY A - CHURN ANALYSIS SUMMARY")
say("=" * 78)
say(f"Run date: {time.strftime('%Y-%m-%d %H:%M')}   FAST mode: {FAST}")
say()
say("1. DATA OVERVIEW")
say(f"  Client.csv {client.shape[0]:,} rows x {client.shape[1]} cols | Record.csv {record.shape[0]:,} rows x {record.shape[1]} cols")
say(f"  Merged on {ID}: {n:,} customers x {df.shape[1]} columns ({len(num_raw)} numeric, {len(cat_cols)} text)")
churn_rate = y.mean()
say(f"  Churn rate in data: {churn_rate:.1%}  ({y.sum():,} churners / {n - y.sum():,} stayers)")
rev = df["rev_Mean"].fillna(df["rev_Mean"].median()).clip(lower=0).values
say(f"  Avg monthly revenue per customer (rev_Mean): mean ${df['rev_Mean'].mean():.2f}, median ${df['rev_Mean'].median():.2f}")
say(f"  ARPU churners ${df.loc[y == 1, 'rev_Mean'].mean():.2f} vs stayers ${df.loc[y == 0, 'rev_Mean'].mean():.2f}")
say(f"  Monthly revenue of churners (sum rev_Mean): {money(rev[y == 1].sum())} "
    f"= {rev[y == 1].sum() / rev.sum():.1%} of total monthly revenue {money(rev.sum())}")
if "months" in df:
    say(f"  Tenure (months): min {df['months'].min():.0f}, median {df['months'].median():.0f}, max {df['months'].max():.0f}")
say()
say("  Data quality notes:")
for col, cond, txt in [("rev_Mean", df.get("rev_Mean", pd.Series(dtype=float)) < 0, "negative revenue (billing adjustments)"),
                       ("eqpdays", df.get("eqpdays", pd.Series(dtype=float)) < 0, "negative equipment age"),
                       ("mou_Mean", df.get("mou_Mean", pd.Series(dtype=float)) == 0, "zero minutes of use")]:
    if col in df:
        say(f"    {col}: {int(cond.sum()):,} rows with {txt}")

# =============================== 2. EDA ===============================
miss = df.drop(columns=[ID]).isna().mean().sort_values(ascending=False)
say()
say("2. MISSING VALUES (top 15, % of customers)")
for c, v in miss.head(15).items():
    say(f"  {c:<22}{v:7.1%}")
say(f"  Columns with any missing values: {(miss > 0).sum()} of {len(miss)}")

top = miss.head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index, top.values * 100, color=BLUE, height=0.6)
for i, v in enumerate(top.values):
    ax.text(v * 100 + 0.8, i, f"{v:.0%}", va="center", color=INK2, fontsize=9.5)
ax.set_xlabel("% of customers with no value")
ax.grid(axis="y", visible=False)
titles(ax, "Missing values by column", "Top 15 columns by share of customers with no value")
save(fig, "fig01_missing_values.png")


def rate_table(groups, label):
    t = pd.DataFrame({"group": groups, "churn": y, "rev": rev}).dropna(subset=["group"])
    g = t.groupby("group", observed=True)
    out = pd.DataFrame({
        "customers": g.size(),
        "churn_rate": g["churn"].mean(),
        "avg_monthly_rev": g["rev"].mean(),
        "monthly_rev_of_churners": g.apply(lambda d: d.loc[d["churn"] == 1, "rev"].sum()),
    })
    out.insert(0, "feature", label)
    return out


def plot_rate(tbl, title, sub, xlabel, fname):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(tbl))
    vals = tbl["churn_rate"].values * 100
    ax.bar(xs, vals, color=BLUE, width=0.55)
    ax.axhline(churn_rate * 100, color=INK2, linewidth=1)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", color=INK, fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.text(1.01, churn_rate * 100, f"All\ncustomers\n{churn_rate:.0%}", transform=blended(ax),
            ha="left", va="center", color=INK2, fontsize=9)
    ax.set_xticks(xs, [f"{i}\nn={int(nn):,}" for i, nn in zip(tbl.index, tbl["customers"].values)])
    ax.set_ylim(0, max(vals.max() * 1.18, churn_rate * 100 * 1.25))
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    titles(ax, title, sub)
    save(fig, fname)


tables = []
say()
say("3. CHURN RATE BY CANDIDATE DRIVER  (rate | customers | avg monthly revenue)")


def report(tbl, name):
    tables.append(tbl)
    say(f"  {name}")
    for idx, r in tbl.iterrows():
        say(f"    {str(idx):<26}{r['churn_rate']:6.1%} | {int(r['customers']):>7,} | ${r['avg_monthly_rev']:6.2f}")
    spread = tbl["churn_rate"].max() - tbl["churn_rate"].min()
    say(f"    -> spread between highest and lowest group: {spread * 100:.1f} percentage points")


if "eqpdays" in df:
    g = pd.cut(df["eqpdays"], [-np.inf, 180, 365, 540, 730, np.inf],
               labels=["< 6 mo", "6-12 mo", "12-18 mo", "18-24 mo", "> 24 mo"])
    t = rate_table(g, "Handset age (eqpdays)")
    report(t, "Handset age (eqpdays)")
    plot_rate(t, "Churn rate by handset age", "Age of the customer's current handset (n = customers in group)",
              "Age of current handset", "fig02_churn_by_handset_age.png")

if "eqpdays" in df:
    # detail: churn by handset age in 30-day steps
    d = pd.DataFrame({"age": (df["eqpdays"] // 30).clip(lower=0), "churn": y}).dropna()
    gd = d.groupby("age")["churn"].agg(["mean", "size"])
    gd = gd[(gd.index <= 30) & (gd["size"] >= 300)]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axvspan(10 - 0.5, 12.5, color="#fde2d6", zorder=0, lw=0)
    ax.plot(gd.index, gd["mean"] * 100, color=BLUE, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1)
    ax.axhline(churn_rate * 100, color=INK2, linewidth=1)
    ax.text(1.01, churn_rate * 100, f"All\ncustomers\n{churn_rate:.0%}", transform=blended(ax),
            ha="left", va="center", color=INK2, fontsize=9)
    ax.text(11, ax.get_ylim()[1] * 0.985, "months 10-12", ha="center", va="top", color=INK2, fontsize=9.5)
    ax.set_xlabel("Age of current handset (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Churn rate by handset age, month by month", "Groups with at least 300 customers")
    save(fig, "fig02b_churn_by_handset_age_detail.png")
    say("  Handset age month by month (churn rate | customers): " +
        ", ".join(f"{int(k)}m {v['mean']:.0%}" for k, v in gd.iterrows() if 6 <= k <= 15))

if "months" in df:
    # detail: churn by exact month in service, months 10-12 highlighted
    d = pd.DataFrame({"m": df["months"], "churn": y})
    gm = d.groupby("m")["churn"].agg(["mean", "size"])
    gm = gm[(gm.index <= 36) & (gm["size"] >= 300)]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cols_ = [ORANGE if 10 <= m <= 12 else BLUE for m in gm.index]
    ax.bar(gm.index, gm["mean"] * 100, color=cols_, width=0.7)
    ax.axhline(churn_rate * 100, color=INK2, linewidth=1)
    ax.text(1.01, churn_rate * 100, f"All\ncustomers\n{churn_rate:.0%}", transform=blended(ax),
            ha="left", va="center", color=INK2, fontsize=9)
    peak = gm.loc[10:12, "mean"]
    if len(peak):
        ax.text(11, peak.max() * 100 + 1.5, f"months 10-12: {peak.mean():.0%}", ha="center", color=INK, fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, gm["mean"].max() * 100 * 1.15)
    ax.set_xlabel("Months in service")
    ax.set_ylabel("Churn rate (%)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Churn rate by month in service", "Each bar = customers with that many months of service (groups >= 300)")
    save(fig, "fig03b_churn_by_month_detail.png")
    say("  Tenure month by month (churn rate): " + ", ".join(f"{int(k)}m {v['mean']:.0%}" for k, v in gm.iterrows() if k <= 15))

if "months" in df:
    g = pd.cut(df["months"], [0, 9, 12, 15, 18, 24, 36, np.inf],
               labels=["6-9", "10-12", "13-15", "16-18", "19-24", "25-36", "37+"])
    t = rate_table(g, "Tenure (months)")
    report(t, "Tenure in months")
    plot_rate(t, "Churn rate by months in service", "Length of the customer relationship (n = customers in group)",
              "Months in service", "fig03_churn_by_tenure.png")

if "change_mou" in df:
    basis = df["avg3mou"] if "avg3mou" in df else df["mou_Mean"]
    pct = 100 * df["change_mou"] / basis.where(basis > 0)
    g = pd.cut(pct, [-np.inf, -50, -20, 20, 50, np.inf],
               labels=["Fell > 50%", "Fell 20-50%", "Stable", "Rose 20-50%", "Rose > 50%"])
    t = rate_table(g, "Usage change vs previous 3 months")
    report(t, "Usage change (change_mou vs 3-month average)")
    plot_rate(t, "Churn rate by change in usage",
              "Minutes of use in the latest month vs the previous 3-month average",
              "Change in monthly minutes of use", "fig04_churn_by_usage_change.png")

if "custcare_Mean" in df:
    g = pd.cut(df["custcare_Mean"], [-0.001, 0, 1, 3, np.inf], labels=["None", "0-1", "1-3", "3+"])
    t = rate_table(g, "Customer care calls / month")
    report(t, "Customer-care calls per month")
    plot_rate(t, "Churn rate by customer-care calls", "Average customer-care calls per month",
              "Customer-care calls per month", "fig05_churn_by_care_calls.png")

if "drop_blk_Mean" in df:
    g = pd.cut(df["drop_blk_Mean"], [-0.001, 2, 5, 10, 20, np.inf], labels=["0-2", "2-5", "5-10", "10-20", "20+"])
    t = rate_table(g, "Dropped or blocked calls / month")
    report(t, "Dropped or blocked calls per month")
    plot_rate(t, "Churn rate by dropped or blocked calls", "Average dropped or blocked calls per month",
              "Dropped or blocked calls per month", "fig06_churn_by_dropped_calls.png")

if "ovrrev_Mean" in df:
    g = pd.cut(df["ovrrev_Mean"], [-np.inf, 0, 10, 30, np.inf], labels=["No overage", "$0-10", "$10-30", "$30+"])
    t = rate_table(g, "Overage charges / month")
    report(t, "Overage charges per month (ovrrev_Mean)")
    plot_rate(t, "Churn rate by overage charges", "Average monthly overage revenue per customer",
              "Overage charges per month", "fig07_churn_by_overage.png")

if "hnd_price" in df:
    g = pd.cut(df["hnd_price"], [-np.inf, 40, 80, 130, np.inf], labels=["< $40", "$40-80", "$80-130", "$130+"])
    report(rate_table(g, "Handset price"), "Handset price")

# Revenue concentration: share of churners vs share of lost revenue by ARPU band
g = pd.cut(pd.Series(rev), [-np.inf, 30, 50, 75, 100, np.inf], labels=["< $30", "$30-50", "$50-75", "$75-100", "$100+"])
t = rate_table(g, "Monthly revenue band")
report(t, "Monthly revenue band (rev_Mean)")
share_churners = t["churn_rate"] * t["customers"] / (t["churn_rate"] * t["customers"]).sum()
share_lost_rev = t["monthly_rev_of_churners"] / t["monthly_rev_of_churners"].sum()
say("    share of churners vs share of churners' revenue:")
for idx in t.index:
    say(f"    {str(idx):<26}{share_churners[idx]:6.1%} of churners | {share_lost_rev[idx]:6.1%} of lost revenue")
fig, ax = plt.subplots(figsize=(8, 4.6))
xs = np.arange(len(t))
w = 0.36
ax.bar(xs - w / 2 - 0.01, share_churners.values * 100, width=w, color=GRAY, label="Share of churners")
ax.bar(xs + w / 2 + 0.01, share_lost_rev.values * 100, width=w, color=BLUE, label="Share of revenue lost")
for x, a, b in zip(xs, share_churners.values * 100, share_lost_rev.values * 100):
    ax.text(x + w / 2 + 0.01, b + 0.8, f"{b:.0f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
ax.set_xticks(xs, [str(i) for i in t.index])
ax.set_ylabel("% of total")
ax.set_xlabel("Customer's average monthly bill")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")
titles(ax, "Share of churners vs share of revenue lost", "Grouped by the customer's average monthly bill")
save(fig, "fig08_revenue_at_risk_by_bill.png")

# Categorical segments (tables only)
for c in ["crclscod", "area", "hnd_webcap", "refurb_new", "asl_flag", "new_cell", "prizm_social_one",
          "dualband", "marital", "creditcd"]:
    if c in df:
        s = df[c].astype(object).where(df[c].notna(), "Missing")
        keep = s.value_counts().head(12).index
        tables.append(rate_table(s.where(s.isin(keep)), c))
seg = pd.concat(tables)
seg.to_csv(OUT / "segment_tables.csv")
say()
say("  Categorical segments with the biggest churn gaps (groups with >= 1,000 customers):")
for c in ["crclscod", "area", "hnd_webcap", "refurb_new", "asl_flag", "new_cell", "prizm_social_one", "dualband"]:
    s = seg[(seg["feature"] == c) & (seg["customers"] >= 1000)]
    if len(s) >= 2:
        hi, lo = s["churn_rate"].idxmax(), s["churn_rate"].idxmin()
        say(f"    {c:<18} highest {hi} {s.loc[hi, 'churn_rate']:.1%} | lowest {lo} {s.loc[lo, 'churn_rate']:.1%}")

# Leakage check: single-feature AUC
say()
say("4. LEAKAGE CHECK - how well does each single numeric column separate churners? (AUC, 0.5 = useless)")
uni = {}
for c in num_raw:
    v = df[c].fillna(df[c].median()).values
    if np.nanstd(v) > 0:
        a = roc_auc_score(y, v)
        uni[c] = max(a, 1 - a)
uni = pd.Series(uni).sort_values(ascending=False)
for c, v in uni.head(10).items():
    say(f"  {c:<22}{v:.3f}")
flag = uni[uni > 0.80]
say("  -> no single column is suspiciously predictive (all < 0.80): no obvious target leakage"
    if flag.empty else f"  -> CHECK FOR LEAKAGE: {list(flag.index)}")

# ======================= 3. FEATURE ENGINEERING =======================
log("feature engineering")


def ratio(a, b):
    return a / b.where(b != 0)


excluded = [c for c in EXCLUDE_FEATURES if c in df.columns]
X = df.drop(columns=[ID, TARGET] + excluded).copy()
cat_cols = [c for c in cat_cols if c not in excluded]
num_raw = [c for c in num_raw if c not in excluded]
new = {}
if {"eqpdays", "months"} <= set(X): new["eqp_vs_tenure"] = ratio(X["eqpdays"], X["months"] * 30.4)
if {"change_mou", "avg3mou"} <= set(X): new["usage_trend_mou"] = ratio(X["change_mou"], X["avg3mou"])
if {"change_rev", "avg3rev"} <= set(X): new["usage_trend_rev"] = ratio(X["change_rev"], X["avg3rev"])
if {"avg3mou", "avgmou"} <= set(X): new["mou_recent_vs_life"] = ratio(X["avg3mou"], X["avgmou"])
if {"avg3rev", "avgrev"} <= set(X): new["rev_recent_vs_life"] = ratio(X["avg3rev"], X["avgrev"])
if {"avg3qty", "avgqty"} <= set(X): new["qty_recent_vs_life"] = ratio(X["avg3qty"], X["avgqty"])
if {"avg6mou", "avgmou"} <= set(X): new["mou_6m_vs_life"] = ratio(X["avg6mou"], X["avgmou"])
if {"ovrrev_Mean", "rev_Mean"} <= set(X): new["overage_share"] = ratio(X["ovrrev_Mean"], X["rev_Mean"])
if {"rev_Mean", "mou_Mean"} <= set(X): new["rev_per_min"] = ratio(X["rev_Mean"], X["mou_Mean"])
if {"totmrc_Mean", "rev_Mean"} <= set(X): new["mrc_share"] = ratio(X["totmrc_Mean"], X["rev_Mean"])
if {"drop_vce_Mean", "plcd_vce_Mean"} <= set(X): new["drop_rate"] = ratio(X["drop_vce_Mean"], X["plcd_vce_Mean"])
if {"blck_vce_Mean", "plcd_vce_Mean"} <= set(X): new["block_rate"] = ratio(X["blck_vce_Mean"], X["plcd_vce_Mean"])
if {"complete_Mean", "attempt_Mean"} <= set(X): new["complete_rate"] = ratio(X["complete_Mean"], X["attempt_Mean"])
if {"unan_vce_Mean", "plcd_vce_Mean"} <= set(X): new["unanswered_rate"] = ratio(X["unan_vce_Mean"], X["plcd_vce_Mean"])
if {"custcare_Mean", "mou_Mean"} <= set(X): new["care_per_100min"] = 100 * ratio(X["custcare_Mean"], X["mou_Mean"])
if {"hnd_price", "rev_Mean"} <= set(X): new["hnd_price_vs_rev"] = ratio(X["hnd_price"], X["rev_Mean"])
if {"phones", "months"} <= set(X): new["phones_per_year"] = ratio(X["phones"], X["months"] / 12)
if {"actvsubs", "uniqsubs"} <= set(X): new["subs_ratio"] = ratio(X["actvsubs"], X["uniqsubs"])
X = pd.concat([X, pd.DataFrame(new)], axis=1).replace([np.inf, -np.inf], np.nan)
for c in cat_cols:
    X[c] = X[c].astype("category")
num_cols = [c for c in X.columns if c not in cat_cols]
features = list(X.columns)
say()
say(f"5. FEATURES: {len(features)} total = {len(num_raw)} numeric + {len(cat_cols)} categorical + {len(new)} engineered")
say(f"  Engineered: {', '.join(new)}")
say(f"  Excluded for fairness (never used by any model): {', '.join(excluded) if excluded else 'none'}")

# ============================ 4. MODELS ============================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

lr_pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), cat_cols),
])
rf_pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                      ("oe", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
])


def as_object(frame):
    f = frame.copy()
    for c in cat_cols:
        f[c] = f[c].astype(object).where(f[c].notna(), np.nan)
    return f


LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  cat_smooth=20, verbosity=-1, seed=SEED, n_jobs=-1)

def fit_lgb(Xtr, ytr, cats):
    """LightGBM with early stopping on an inner 10% split of the training fold (validation fold stays unseen)."""
    itr, ies = train_test_split(np.arange(len(Xtr)), test_size=0.1, stratify=ytr, random_state=SEED)
    dtr = lgb.Dataset(Xtr.iloc[itr], ytr[itr], categorical_feature=cats)
    des = lgb.Dataset(Xtr.iloc[ies], ytr[ies], categorical_feature=cats, reference=dtr)
    return lgb.train(dict(LGB_PARAMS, metric="auc"), dtr, num_boost_round=5000, valid_sets=[des],
                     callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])


models = ["Logistic Regression", "Random Forest"] + (["XGBoost (tutorial settings)"] if HAS_XGB else []) + ["LightGBM"]
oof = {m: np.zeros(n) for m in models}
fold_auc = {m: [] for m in models}
shap = np.zeros((n, len(features)))
lgb_iters = []
log(f"training {len(models)} models x {N_FOLDS} folds - this takes ~10-15 minutes. "
    "Progress is printed after each model. Please don't press Ctrl+C.")
for k, (tr, va) in enumerate(folds):
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]
    # Logistic regression
    log(f"fold {k + 1}/{N_FOLDS}: training Logistic Regression ...")
    m = Pipeline([("pre", lr_pre), ("lr", LogisticRegression(max_iter=3000, C=0.5))])
    m.fit(as_object(Xtr), ytr)
    oof["Logistic Regression"][va] = m.predict_proba(as_object(Xva))[:, 1]
    # Random forest
    log(f"fold {k + 1}/{N_FOLDS}: training Random Forest ...")
    m = Pipeline([("pre", rf_pre), ("rf", RandomForestClassifier(
        n_estimators=100 if FAST else 300, min_samples_leaf=20, max_features="sqrt",
        n_jobs=-1, random_state=SEED))])
    m.fit(as_object(Xtr), ytr)
    oof["Random Forest"][va] = m.predict_proba(as_object(Xva))[:, 1]
    # XGBoost with the tutorial's settings (on our engineered features)
    if HAS_XGB:
        log(f"fold {k + 1}/{N_FOLDS}: training XGBoost ...")
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric="logloss",
                              tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        oof["XGBoost (tutorial settings)"][va] = m.predict_proba(Xva)[:, 1]
    # LightGBM: early stopping on an inner 10% split of the training fold (validation fold stays unseen)
    log(f"fold {k + 1}/{N_FOLDS}: training LightGBM ...")
    booster = fit_lgb(Xtr, ytr, cat_cols)
    oof["LightGBM"][va] = booster.predict(Xva, num_iteration=booster.best_iteration)
    shap[va] = booster.predict(Xva, num_iteration=booster.best_iteration, pred_contrib=True)[:, :-1]
    lgb_iters.append(booster.best_iteration)
    for mname in models:
        fold_auc[mname].append(roc_auc_score(yva, oof[mname][va]))
    log(f"fold {k + 1}/{N_FOLDS} done - AUC: " + " | ".join(f"{mn.split(' (')[0]} {fold_auc[mn][-1]:.4f}" for mn in models))

say()
say(f"6. MODEL COMPARISON  ({N_FOLDS}-fold stratified cross-validation, out-of-fold predictions, seed {SEED})")
say("  Metric: ROC-AUC = how well the model ranks churners above stayers (0.5 random, 1.0 perfect).")
say("  Accuracy and F1 use a 0.5 threshold (classes are ~balanced in this data).")
res = []
for mname in models:
    p = oof[mname]
    pred = (p >= 0.5).astype(int)
    order = np.argsort(-p)
    cap = {q: y[order[: int(q * n)]].sum() / y.sum() for q in (0.1, 0.2, 0.3)}
    res.append(dict(model=mname, auc=roc_auc_score(y, p), auc_sd=np.std(fold_auc[mname]),
                    acc=accuracy_score(y, pred), f1=f1_score(y, pred), cap10=cap[0.1], cap20=cap[0.2], cap30=cap[0.3],
                    prec20=y[order[: int(0.2 * n)]].mean()))
res = pd.DataFrame(res).set_index("model")
say(f"  {'Model':<30}{'AUC':>8}{'(+/-)':>8}{'Acc':>8}{'F1':>8}{'Top10%':>9}{'Top20%':>9}")
for mname, r in res.iterrows():
    say(f"  {mname:<30}{r['auc']:8.4f}{r['auc_sd']:8.4f}{r['acc']:8.4f}{r['f1']:8.4f}{r['cap10']:9.1%}{r['cap20']:9.1%}")
say("  (Top10%/Top20% = share of all churners found when contacting the 10%/20% highest-risk customers;")
say("   a random list finds 10%/20%.)")
best = res["auc"].idxmax()
say(f"  Best model: {best} (AUC {res.loc[best, 'auc']:.4f}, accuracy {res.loc[best, 'acc']:.4f})")
say(f"  Tutorial baseline for reference: XGBoost on label-encoded raw data, 70/30 split, accuracy 0.6368")
say(f"  LightGBM trees per fold (early stopping): {lgb_iters}")
say(f"  Precision in top 20%: {res.loc['LightGBM', 'prec20']:.1%} of contacted customers are real churners "
    f"(vs {churn_rate:.1%} at random) -> lift {res.loc['LightGBM', 'prec20'] / churn_rate:.2f}x")

auc_with_excluded = None
if FAIRNESS_CHECK and excluded:
    log(f"fairness check: re-training LightGBM WITH {excluded} for comparison ...")
    Xf = X.copy()
    for c in excluded:
        Xf[c] = df[c].astype("category") if c not in num_cols and not pd.api.types.is_numeric_dtype(df[c]) else df[c]
    cats_f = cat_cols + [c for c in excluded if str(Xf[c].dtype) == "category"]
    oof_f = np.zeros(n)
    for k, (tr, va) in enumerate(folds):
        bst = fit_lgb(Xf.iloc[tr], y[tr], cats_f)
        oof_f[va] = bst.predict(Xf.iloc[va], num_iteration=bst.best_iteration)
    auc_with_excluded = roc_auc_score(y, oof_f)
    say(f"  Fairness check: LightGBM AUC WITH {', '.join(excluded)} = {auc_with_excluded:.4f} vs "
        f"WITHOUT = {res.loc['LightGBM', 'auc']:.4f} (difference {auc_with_excluded - res.loc['LightGBM', 'auc']:+.4f})")

# fig09 model comparison (dot plot; axis starts at 0.5 = random guessing)
fig, ax = plt.subplots(figsize=(8, 3.6))
r = res.sort_values("auc")
ys = np.arange(len(r))
cols = [BLUE if m == best else GRAY for m in r.index]
ax.hlines(ys, 0.5, r["auc"], color=GRID, linewidth=2)
ax.scatter(r["auc"], ys, s=110, color=cols, zorder=3, edgecolor="white", linewidth=2)
for yy, v in zip(ys, r["auc"]):
    ax.text(v + 0.004, yy, f"{v:.3f}", va="center", color=INK, fontsize=10.5, fontweight="bold")
ax.set_yticks(ys, r.index)
ax.set_xlim(0.5, max(0.8, r["auc"].max() + 0.04))
ax.set_xlabel("ROC-AUC  (0.5 = random guessing, 1.0 = perfect)")
ax.grid(axis="y", visible=False)
titles(ax, "Model comparison: ROC-AUC", f"{N_FOLDS}-fold cross-validation on {n:,} customers; best model highlighted")
save(fig, "fig09_model_comparison.png")

# fig10 gains chart
p_best = oof["LightGBM"]
order = np.argsort(-p_best)
frac = np.arange(1, n + 1) / n
gain = np.cumsum(y[order]) / y.sum()
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(frac * 100, gain * 100, color=BLUE, linewidth=2, label="Contact by model risk score")
ax.plot([0, 100], [0, 100], color=GRAY, linewidth=2, label="Contact at random")
q = TOP_SHARE
gq = gain[int(q * n) - 1]
ax.scatter([q * 100], [gq * 100], s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
ax.annotate(f"Top {q:.0%} contacted:\n{gq:.0%} of churners found", (q * 100, gq * 100), xytext=(8, 28),
            textcoords="offset points", color=INK, fontsize=10)
ax.set_xlabel("% of customers contacted (highest risk first)")
ax.set_ylabel("% of all churners found")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.legend(loc="lower right")
titles(ax, "Cumulative gains: churners found vs customers contacted", "LightGBM out-of-fold predictions")
save(fig, "fig10_gains_chart.png")

# ===================== 5. DRIVERS (SHAP) & REASON CODES =====================
FAMILIES = ["Handset age & device", "Usage drop", "Network quality", "Customer care",
            "Price & overage", "Tenure & contract", "Usage level", "Customer profile"]
FAM_COLOR = dict(zip(FAMILIES, C))
ACTIONS = {
    "Handset age & device": "Early handset-upgrade offer",
    "Usage drop": "Usage check-in + personalised bonus",
    "Network quality": "Network fix + service credit",
    "Customer care": "Priority callback to resolve the issue",
    "Price & overage": "Plan right-sizing (end overage bills)",
    "Tenure & contract": "Renewal offer before contract end",
}
PROFILE = {"crclscod", "prizm_social_one", "area", "truck", "rv", "ownrent", "lor", "dwlltype", "marital",
           "adults", "infobase", "income", "numbcars", "HHstatin", "dwllsize", "forgntvl", "ethnic",
           "creditcd", "asl_flag", "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17"}
PRICE = {"rev_Mean", "totmrc_Mean", "ovrmou_Mean", "ovrrev_Mean", "vceovr_Mean", "datovr_Mean", "roam_Mean",
         "da_Mean", "totrev", "adjrev", "avgrev", "avg3rev", "avg6rev", "overage_share", "rev_per_min", "mrc_share"}


def family_of(c):
    lc = c.lower()
    if c in {"eqpdays", "hnd_price", "refurb_new", "hnd_webcap", "models", "phones", "dualband",
             "eqp_vs_tenure", "hnd_price_vs_rev", "phones_per_year"}:
        return "Handset age & device"
    if c in {"change_mou", "change_rev"} or "trend" in lc or "recent_vs" in lc or "6m_vs" in lc:
        return "Usage drop"
    if lc.startswith(("drop_", "blck_", "unan_")) or c in {"block_rate", "complete_rate", "unanswered_rate"}:
        return "Network quality"
    if lc.startswith(("custcare", "ccrndmou", "cc_mou", "care_")):
        return "Customer care"
    if c in PRICE:
        return "Price & overage"
    if c in {"months", "uniqsubs", "actvsubs", "new_cell", "subs_ratio"}:
        return "Tenure & contract"
    if c in PRICE or c in PROFILE or lc.startswith("kid"):
        return "Customer profile"
    return "Usage level"


fam = pd.Series({f: family_of(f) for f in features})
imp = pd.Series(np.abs(shap).mean(axis=0), index=features).sort_values(ascending=False)
fam_imp = imp.groupby(fam).sum().sort_values(ascending=False)
say()
say("7. WHAT DRIVES CHURN - mean |SHAP| of LightGBM (out-of-fold), top 20 features")
for f, v in imp.head(20).items():
    say(f"  {f:<24}{v:.4f}   [{fam[f]}]  {lab(f)}")
say("  Importance by driver family (share of total):")
for f, v in fam_imp.items():
    say(f"  {f:<24}{v / fam_imp.sum():6.1%}{'   (actionable)' if f in ACTIONS else ''}")

top15 = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.barh([lab(f) for f in top15.index], top15.values, color=[FAM_COLOR[fam[f]] for f in top15.index], height=0.62)
shown = [f for f in FAMILIES if f in set(fam[top15.index])]
for f in shown:
    ax.barh([np.nan], [np.nan], color=FAM_COLOR[f], label=f)
ax.legend(loc="lower right", fontsize=9.5)
ax.set_xlabel("Average impact on predicted churn risk (mean |SHAP|, log-odds)")
ax.grid(axis="y", visible=False)
titles(ax, "Top 15 churn drivers (SHAP importance)", "LightGBM, out-of-fold; colour = business lever (driver family)")
save(fig, "fig11_feature_importance.png")

# fig12 SHAP dependence for the 3 strongest actionable numeric drivers
act_num = [f for f in imp.index if fam[f] in ACTIONS and f in num_cols][:3]
if act_num:
    fig, axes = plt.subplots(1, len(act_num), figsize=(4 * len(act_num), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=min(5000, n), replace=False)
    for ax, f in zip(axes, act_num):
        x = X[f].values[idx].astype(float)
        s = shap[idx, features.index(f)]
        ok = ~np.isnan(x)
        lo, hi = np.nanpercentile(x, [1, 99])
        ok &= (x >= lo) & (x <= hi)
        ax.scatter(x[ok], s[ok], s=6, alpha=0.25, color=FAM_COLOR[fam[f]], linewidth=0)
        bins = np.unique(np.nanpercentile(x[ok], np.linspace(0, 100, 16)))
        if len(bins) > 3:
            mid = pd.Series(s[ok]).groupby(pd.cut(x[ok], bins, include_lowest=True), observed=True).mean()
            centers = [iv.mid for iv in mid.index]
            ax.plot(centers, mid.values, color=INK, linewidth=2)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(lab(f), fontsize=12, pad=8)
        ax.set_xlabel(f"{f}  ({fam[f]})", fontsize=9.5)
    axes[0].set_ylabel("Effect on churn risk\n(+ raises, - lowers)")
    fig.suptitle("Effect of the strongest actionable drivers on churn risk (SHAP)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig12_driver_effects.png")
say("  Strongest actionable numeric drivers (see fig12): " + ", ".join(act_num))

# Reason codes for the high-risk group
fam_shap = pd.DataFrame(shap, columns=features).T.groupby(fam).sum().T
act = fam_shap[[f for f in ACTIONS if f in fam_shap]]
reason = act.idxmax(axis=1).where(act.max(axis=1) > 0, "No single lever")
risk_rank = pd.Series(p_best).rank(ascending=False, method="first")
high = (risk_rank <= TOP_SHARE * n).values
hr = pd.DataFrame({"reason": reason[high].values, "p": p_best[high], "churn": y[high], "rev": rev[high]})
g = hr.groupby("reason")
tab = pd.DataFrame({"customers": g.size(), "share": g.size() / len(hr), "actual_churn_rate": g["churn"].mean(),
                    "avg_monthly_rev": g["rev"].mean(),
                    "expected_monthly_rev_at_risk": g.apply(lambda d: (d["p"] * d["rev"]).sum())}
                   ).sort_values("customers", ascending=False)
tab["action"] = [ACTIONS.get(r, "General loyalty offer") for r in tab.index]
say()
say(f"8. REASON CODES - main actionable driver for each customer in the top {TOP_SHARE:.0%} risk group "
    f"({high.sum():,} customers, actual churn rate {y[high].mean():.1%})")
say(f"  {'Main reason':<24}{'Customers':>10}{'Share':>8}{'Churn':>8}{'ARPU':>9}{'Exp. rev at risk/mo':>21}   Action")
for r_, row in tab.iterrows():
    say(f"  {r_:<24}{int(row['customers']):>10,}{row['share']:8.1%}{row['actual_churn_rate']:8.1%}"
        f"{row['avg_monthly_rev']:9.2f}{money(row['expected_monthly_rev_at_risk']):>21}   {row['action']}")
tab.to_csv(OUT / "reason_codes_high_risk.csv")

t2 = tab[tab["customers"] > 0][::-1]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.barh(np.arange(len(t2)), t2["share"] * 100,
        color=[FAM_COLOR.get(r_, GRAY) for r_ in t2.index], height=0.6)
for i, (r_, row) in enumerate(t2.iterrows()):
    pct = f"{row['share']:.0%}" if row["share"] >= 0.01 else "<1%"
    ax.text(row["share"] * 100 + 0.8, i, f"{pct}  ->  {row['action']}", va="center", color=INK2, fontsize=9.5)
ax.set_yticks(np.arange(len(t2)), t2.index)
xmax = t2["share"].max() * 100 * 2.1
ax.set_xlim(0, xmax)
ax.set_xticks([v for v in range(0, 101, 20) if v <= xmax])
ax.set_xlabel(f"% of high-risk customers (top {TOP_SHARE:.0%} by model score)")
ax.grid(axis="y", visible=False)
titles(ax, "Main actionable churn reason in the high-risk group",
       "Per-customer SHAP drivers; label = matching retention action")
save(fig, "fig13_reason_codes.png")

# ============================ 6. BUSINESS IMPACT ============================
A = ASSUME
p1 = churn_rate


def prior_shift(p, r):
    """Re-calibrate probabilities from the sample churn rate to the real churn rate r (Saerens et al., 2002)."""
    a = p * (r / p1)
    b = (1 - p) * ((1 - r) / (1 - p1))
    return a / (a + b)


def simulate(score, weights, A_):
    """Contact customers from highest score down; returns net benefit etc. on a grid of contact shares."""
    o = np.argsort(-score, kind="mergesort")
    w_, y_, r_ = weights[o], y[o], rev[o]
    cw = np.cumsum(w_)
    churn_reached = np.cumsum(w_ * y_)
    value_reached = np.cumsum(w_ * y_ * r_) * A_["HORIZON_MONTHS"]
    grid = np.arange(0.005, 1.0001, 0.005)
    i = np.clip(np.searchsorted(cw / cw[-1], grid), 0, len(cw) - 1)
    contacted, churners = cw[i], churn_reached[i]
    loyal = contacted - churners
    saved = A_["SAVE_RATE"] * churners
    revenue = A_["SAVE_RATE"] * value_reached[i]
    cost = A_["CONTACT_COST"] * contacted + A_["INCENTIVE_COST"] * (saved + A_["TAKEUP_LOYAL"] * loyal)
    return pd.DataFrame(dict(share=grid, contacted=contacted, churners=churners, saved=saved,
                             revenue=revenue, cost=cost, net=revenue - cost)).set_index("share")


def random_curve(weights, A_):
    tot_w, tot_c = weights.sum(), (weights * y).sum()
    tot_v = (weights * y * rev).sum() * A_["HORIZON_MONTHS"]
    grid = np.arange(0.005, 1.0001, 0.005)
    contacted, churners = grid * tot_w, grid * tot_c
    saved = A_["SAVE_RATE"] * churners
    revenue = A_["SAVE_RATE"] * grid * tot_v
    cost = A_["CONTACT_COST"] * contacted + A_["INCENTIVE_COST"] * (saved + A_["TAKEUP_LOYAL"] * (contacted - churners))
    return pd.DataFrame(dict(share=grid, contacted=contacted, churners=churners, saved=saved,
                             revenue=revenue, cost=cost, net=revenue - cost)).set_index("share")


def scenario(name, r, A_, make_plot=None, xmax=0.6):
    if r is None:
        w = np.full(n, A_["CUSTOMER_BASE"] / n)
        p = p_best
    else:
        w = np.where(y == 1, r / p1, (1 - r) / (1 - p1)) * A_["CUSTOMER_BASE"] / n
        p = prior_shift(p_best, r)
    curves = {"Random list (no model)": random_curve(w, A_),
              "Model: highest churn risk first": simulate(p, w, A_),
              "Model: highest revenue at risk first": simulate(p * rev, w, A_)}
    say()
    say(f"  Scenario: {name}  (per {A_['CUSTOMER_BASE']:,} customers, one campaign)")
    say(f"    churners in base: {(w * y).sum():,.0f}")
    say(f"    {'Strategy':<40}{'Best share':>11}{'Contacted':>11}{'Saved':>8}{'Revenue kept':>14}{'Cost':>10}{'Net':>10}{'ROI':>7}")
    best_row = None
    for sname, cv in curves.items():
        cvr = cv[cv.index <= 0.5]
        b = cvr["net"].idxmax()
        row = cvr.loc[b]
        roi = row["net"] / row["cost"] if row["cost"] > 0 else np.nan
        say(f"    {sname:<40}{b:11.1%}{row['contacted']:11,.0f}{row['saved']:8,.0f}{money(row['revenue']):>14}"
            f"{money(row['cost']):>10}{money(row['net']):>10}{roi:7.0%}")
        if sname != "Random list (no model)" and (best_row is None or row["net"] > best_row[2]["net"]):
            best_row = (sname, b, row)
    sname, b, row = best_row
    say("    (search limited to contacting at most 50% of customers)")
    rnd = curves["Random list (no model)"].loc[b]
    say(f"    -> Recommended: '{sname}', contact top {b:.1%}: net {money(row['net'])} per campaign "
        f"vs {money(rnd['net'])} for a random list of the same size")
    say("    Detail for the recommended strategy:")
    say(f"    {'Share':>8}{'Contacted':>11}{'Churners reached':>18}{'Saved':>8}{'Revenue kept':>14}{'Cost':>10}{'Net':>10}")
    cv = curves[sname]
    for sh in sorted({0.02, 0.05, 0.10, 0.20, 0.30, round(b, 3)}):
        rr = cv.loc[cv.index[np.argmin(np.abs(cv.index - sh))]]
        say(f"    {sh:8.1%}{rr['contacted']:11,.0f}{rr['churners']:18,.0f}{rr['saved']:8,.0f}"
            f"{money(rr['revenue']):>14}{money(rr['cost']):>10}{money(rr['net']):>10}")
    if make_plot:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        colors = {"Random list (no model)": GRAY, "Model: highest churn risk first": BLUE,
                  "Model: highest revenue at risk first": ORANGE}
        for s_, cv_ in curves.items():
            cv_ = cv_[cv_.index <= xmax]
            ax.plot(cv_.index * 100, cv_["net"] / 1e3, color=colors[s_], linewidth=2, label=s_)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.scatter([b * 100], [row["net"] / 1e3], s=80, color=colors[sname], zorder=3, edgecolor="white", linewidth=2)
        ax.annotate(f"Best: contact top {b:.1%}\nnet {money(row['net'])} per campaign", (b * 100, row["net"] / 1e3),
                    xytext=(0.58, 0.93), textcoords="axes fraction", va="top", color=INK, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
        ax.set_xlabel("% of customers contacted")
        ax.set_ylabel("Net benefit per campaign ($ thousands)")
        ax.legend(loc="lower left", fontsize=9.5)
        titles(ax, make_plot[1], f"Per {A_['CUSTOMER_BASE']:,} customers; {name}")
        save(fig, make_plot[0])
    return sname, b, row, (w * y).sum()


say()
say("9. BUSINESS IMPACT SIMULATION")
say("  Assumptions (EDIT in the script, cite sources on the slide):")
for k_, v_ in A.items():
    say(f"    {k_:<16}= {v_}")
say("  Logic: contact customers from the top of the list. Saved customers = SAVE_RATE x churners reached.")
say("  Revenue kept = saved customers x their monthly bill x HORIZON_MONTHS.")
say("  Cost = CONTACT_COST x contacted + INCENTIVE_COST x (saved customers + TAKEUP_LOYAL x loyal customers contacted).")
say("  Sources: churn benchmarks - AT&T 2Q26 release, TelecomLead (29 Sep 2026), Recon Analytics (27 Jul 2026);")
say("           contact cost - Twilio call-centre pricing (Nov 2025); churn-reduction cross-check - McKinsey (2017).")
ds_name, ds_b, ds_row, _ = scenario("REFERENCE ONLY - churn rate as in the dataset (~50%, unrealistic)", None, A)
sname = None
if A["BASE_CHURN_RATE"]:
    sname, b, row, tot_churn = scenario(f"realistic churn rate {A['BASE_CHURN_RATE']:.1%} (probabilities re-calibrated)",
                                        A["BASE_CHURN_RATE"], A,
                                        ("fig14_roi_net_benefit.png", "Net benefit of a retention campaign by targeting strategy"),
                                        xmax=0.3)
    say(f"    Share of ALL churn prevented: {row['saved'] / tot_churn:.1%} "
        f"(McKinsey cross-check: analytics-led programmes cut churn by up to 15%)")
    say()
    say(f"  Sensitivity (realistic rate {A['BASE_CHURN_RATE']:.1%}): best net benefit per campaign")
    say(f"    {'':<14}" + "".join(f"{'offer $' + str(int(c)):>14}" for c in (30, 60, 90)))
    for sr in (0.15, 0.25, 0.35):
        cells = []
        for ic in (30, 60, 90):
            A2 = dict(A, SAVE_RATE=sr, INCENTIVE_COST=ic)
            w = np.where(y == 1, A["BASE_CHURN_RATE"] / p1, (1 - A["BASE_CHURN_RATE"]) / (1 - p1)) * A["CUSTOMER_BASE"] / n
            p = prior_shift(p_best, A["BASE_CHURN_RATE"])
            best_net = max(simulate(sc, w, A2).loc[lambda d: d.index <= 0.5, "net"].max() for sc in (p, p * rev))
            cells.append(money(best_net))
        say(f"    save {sr:.0%}{'':<5}" + "".join(f"{c_:>14}" for c_ in cells))
    say("    (negative = no profitable targeting share exists under those assumptions)")
    say()
    say("  Sensitivity: best net benefit per campaign by real churn rate x save rate (offer $60)")
    say(f"    {'':<14}" + "".join(f"{'save ' + str(int(sr * 100)) + '%':>14}" for sr in (0.15, 0.25, 0.35)))
    for r_ in (0.01, 0.02, 0.03):
        cells = []
        for sr in (0.15, 0.25, 0.35):
            A2 = dict(A, SAVE_RATE=sr, BASE_CHURN_RATE=r_)
            w = np.where(y == 1, r_ / p1, (1 - r_) / (1 - p1)) * A["CUSTOMER_BASE"] / n
            p = prior_shift(p_best, r_)
            best_net = max(simulate(sc, w, A2).loc[lambda d: d.index <= 0.5, "net"].max() for sc in (p, p * rev))
            cells.append(money(best_net))
        say(f"    churn {r_:.0%}{'':<6}" + "".join(f"{c_:>14}" for c_ in cells))

say()
say("10. EXECUTIVE SNAPSHOT (numbers for the title / summary slide)")
say(f"  Customers analysed: {n:,} | churn rate in data: {churn_rate:.1%}")
say(f"  Monthly revenue of churners: {money(rev[y == 1].sum())} ({rev[y == 1].sum() / rev.sum():.1%} of monthly revenue)")
say(f"  Best model: {best}, ROC-AUC {res.loc[best, 'auc']:.3f}, accuracy {res.loc[best, 'acc']:.3f}")
say(f"  Top 10% highest-risk customers contain {res.loc['LightGBM', 'cap10']:.0%} of churners "
    f"(lift {res.loc['LightGBM', 'cap10'] / 0.10:.2f}x vs random)")
say(f"  Top driver families: " + ", ".join(f"{f} {v / fam_imp.sum():.0%}" for f, v in fam_imp.head(3).items()))
say(f"  Largest reason group in high-risk segment: {tab.index[0]} ({tab['share'].iloc[0]:.0%}) -> {tab['action'].iloc[0]}")
if auc_with_excluded is not None:
    say(f"  Fairness: removing {', '.join(excluded)} changes AUC by {res.loc['LightGBM', 'auc'] - auc_with_excluded:+.4f}")
if sname:
    say(f"  ROI at {A['BASE_CHURN_RATE']:.1%} monthly churn: '{sname}', contact top {b:.1%} "
        f"({row['contacted']:,.0f} customers): net {money(row['net'])} per monthly campaign per {A['CUSTOMER_BASE']:,} customers")
    say(f"    revenue kept {money(row['revenue'])}, campaign cost {money(row['cost'])}, ROI {row['net'] / row['cost']:.0%}, "
        f"customers saved {row['saved']:,.0f}")
    say(f"    if repeated every month: ~{money(12 * row['net'])} per year per {A['CUSTOMER_BASE']:,} customers; "
        f"~{money(120 * row['net'])} per year for a 1-million-customer operator (illustrative)")
say()
say("=" * 78)
say(f"Finished in {time.time() - T0:.0f}s. Figures and tables are in: {OUT}")
(OUT / "summary.txt").write_text("\n".join(SUMMARY), encoding="utf-8")
print(f"\nDONE - send results/summary.txt and the results/fig*.png files.")
