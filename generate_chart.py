import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUTPUT_IMG = "nav_apy_chart.png"

def parse_pct(series):
    """Parse '3.52800%' strings or plain floats into float."""
    def _p(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().rstrip("%")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_p)

xls = pd.ExcelFile("nav_volatility_analysis.xlsx", engine="openpyxl")

stac  = pd.read_excel(xls, "STAC")
acred = pd.read_excel(xls, "ACRED")
vbill = pd.read_excel(xls, "VBILL")

for df in (stac, acred, vbill):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

stac["apy"]  = parse_pct(stac["APY 30D"])
acred["apy"] = parse_pct(acred["APY 30D"])
vbill["apy"] = parse_pct(vbill["APY 7D"])

# Align to common date range where at least one series has data
# Use STAC start (first non-null 30D APY) as the chart start
chart_start = stac.loc[stac["apy"].notna(), "Date"].iloc[0]

stac  = stac[stac["Date"] >= chart_start].reset_index(drop=True)
acred = acred[acred["Date"] >= chart_start].reset_index(drop=True)
vbill = vbill[vbill["Date"] >= chart_start].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(22, 9))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

BLUE   = "#4c9be8"
RED    = "#e85c5c"
YELLOW = "#f0b429"

def plot_series(ax, df, col_label, color, label):
    d = df[df["apy"].notna()]
    ax.plot(d["Date"], d["apy"], color=color, linewidth=1.5,
            marker="o", markersize=3, label=label, zorder=3)
    # label every 7th point to avoid clutter
    for i, (_, row) in enumerate(d.iterrows()):
        if i % 7 == 0 or i == len(d) - 1:
            ax.annotate(
                f"{row['apy']:.2f}%",
                xy=(row["Date"], row["apy"]),
                xytext=(0, 6), textcoords="offset points",
                fontsize=6.5, color=color, ha="center", va="bottom",
            )

plot_series(ax, vbill, "APY 7D",  BLUE,   "vbill_7d_apy")
plot_series(ax, stac,  "APY 30D", RED,    "stac_30d_apy")
plot_series(ax, acred, "APY 30D", YELLOW, "acred_30d_apy")

ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha="right", fontsize=7, color="#cccccc")
plt.yticks(fontsize=7, color="#cccccc")

ax.spines[:].set_color("#333333")
ax.tick_params(colors="#cccccc", which="both")
ax.yaxis.label.set_color("#cccccc")
ax.xaxis.label.set_color("#cccccc")

ax.set_xlabel("Date", color="#cccccc", fontsize=9)
ax.set_ylabel("APY (%)", color="#cccccc", fontsize=9)

legend = ax.legend(
    loc="upper left", framealpha=0.3, facecolor="#1a1a1a",
    edgecolor="#444444", labelcolor="linecolor", fontsize=8
)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUTPUT_IMG}")
