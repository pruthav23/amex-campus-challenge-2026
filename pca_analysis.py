"""Principal component analysis (PCA) of the 22 usable features.

Why: to test whether the feature groups found with correlations hold up,
and whether a compressed, unsupervised view of the data could replace the
business-driven profit model.

Steps:
  1. Drop f23 (88% missing); floor f7's few negative values (refunds) at 0.
  2. Fill missing values with each feature's median.
  3. log(1 + x) on right-skewed, non-negative features, so a few extreme
     members don't dominate.
  4. StandardScaler: every feature to mean 0, standard deviation 1, so
     large-unit features (spend in thousands) don't outweigh counts.
  5. PCA on all components.

Usage:
    python pca_analysis.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from profitability_framework import prepare_data, score_members

DATA_PATH = "data/campus_challenge_r1_data.csv"
FEATURES = [f"f{i}" for i in range(1, 24) if i != 23]
TOP_SHARE = 0.20
NAMES = {
    "f1": "revolve balance", "f2": "cancel calls", "f3": "collection calls",
    "f4": "points balance", "f5": "total spend (f5)", "f6": "airlines",
    "f7": "other spend", "f8": "entertainment", "f9": "lodging", "f10": "dining",
    "f11": "risk score", "f12": "logins", "f13": "lounge visits",
    "f14": "airline credits", "f15": "cab benefits", "f16": "entertain. credits",
    "f17": "lend line", "f18": "consumer lend line", "f19": "supp. accounts",
    "f20": "charge cards", "f21": "points redeemed", "f22": "emails opened",
}


def prepare(raw):
    features = raw[FEATURES].copy()
    features["f7"] = features["f7"].clip(lower=0)
    features = features.fillna(features.median())
    skewed = [c for c in FEATURES if features[c].skew() > 1 and features[c].min() >= 0]
    features[skewed] = np.log1p(features[skewed])
    return StandardScaler().fit_transform(features), skewed


def top_flags(values, count):
    return values >= np.sort(values)[-count]


def plot_variance(explained):
    cumulative = np.cumsum(explained)
    count = len(explained)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(1, count + 1), explained * 100, color="#2e6f8e", label="Each component")
    axis.plot(range(1, count + 1), cumulative * 100, color="#c62828", marker="o",
              markersize=4, label="Cumulative")
    axis.axhline(80, color="grey", linestyle="--", linewidth=1)
    needed = int(np.searchsorted(cumulative, 0.80) + 1)
    axis.annotate(f"{needed} components to reach 80%", (needed, 80),
                  xytext=(needed + 1.2, 62), fontsize=9,
                  arrowprops={"arrowstyle": "->", "color": "grey"})
    axis.set_xticks(range(1, count + 1))
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Variance explained (%)")
    axis.set_ylim(0, 105)
    axis.set_title(f"No dominant structure: the first component explains only "
                   f"{explained[0]:.0%}", fontsize=13, pad=10)
    axis.legend(loc="center right")
    plt.tight_layout()
    plt.savefig("figures/pca_variance.png", dpi=130, bbox_inches="tight")
    plt.close()


def plot_loadings(loadings, explained, shown=7):
    table = loadings.iloc[:, :shown].copy()
    table.index = [f"{c}  {NAMES[c]}" for c in table.index]
    table.columns = [f"PC{i + 1}\n{explained[i]:.0%}" for i in range(shown)]
    plt.figure(figsize=(8.5, 9))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.8,
                vmax=0.8, linewidths=0.5, annot_kws={"size": 8}, cbar_kws={"shrink": 0.6})
    plt.title("PCA loadings: how strongly each feature drives each component",
              fontsize=13, pad=12)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/pca_loadings.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    scaled, skewed = prepare(raw)
    pca = PCA().fit(scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[f"PC{i + 1}" for i in range(len(FEATURES))])
    scores = pca.transform(scaled)

    print("=" * 70)
    print("1) HOW MUCH STRUCTURE IS THERE?")
    print("=" * 70)
    print(f"  Log-transformed before scaling: {', '.join(skewed)}")
    print("  Variance explained, PC1-PC7: "
          + ", ".join(f"{v:.1%}" for v in explained[:7]))
    print(f"  Components needed for 80%: {np.searchsorted(cumulative, 0.80) + 1}"
          f" of {len(FEATURES)};  for 90%: {np.searchsorted(cumulative, 0.90) + 1}")
    print(f"  Components with eigenvalue > 1: {(pca.explained_variance_ > 1).sum()}")

    print("\n" + "=" * 70)
    print("2) WHAT EACH COMPONENT REPRESENTS (top 5 loadings)")
    print("=" * 70)
    for i, pc in enumerate(loadings.columns[:7]):
        strongest = loadings[pc].abs().sort_values(ascending=False).head(5).index
        print(f"  {pc} ({explained[i]:.1%}): "
              + ", ".join(f"{c} {loadings.loc[c, pc]:+.2f}" for c in strongest))
    home = loadings.loc["f5"].abs().idxmax()
    print(f"  f5 'total spend': loading on PC1 (spending) {loadings.loc['f5', 'PC1']:+.2f};"
          f" strongest on {home} ({loadings.loc['f5', home]:+.2f})")

    print("\n" + "=" * 70)
    print("3) CAN PCA FIND THE PROFITABLE MEMBERS?")
    print("=" * 70)
    data = prepare_data(DATA_PATH)
    profit = score_members(data).to_numpy()
    count = int(len(profit) * TOP_SHARE)
    top = top_flags(profit, count)
    pc1 = scores[:, 0] * np.sign(np.corrcoef(scores[:, 0], profit)[0, 1])
    print(f"  Ranking by PC1 (spending) alone recovers "
          f"{(top_flags(pc1, count) & top).sum() / count:.1%} of the profit model's top 20%"
          f" (random: {TOP_SHARE:.0%})")
    revolver = (data["f1"] > 0).to_numpy()
    print(f"  Members carrying a balance: {revolver[top].mean():.0%} of the top 20%,"
          f" {revolver[~top].mean():.0%} of the rest")
    print(f"  Mean risk score (f11): top 20% {data['f11'][top].mean():.3f},"
          f" rest {data['f11'][~top].mean():.3f}")
    risky = (data["f11"] > data.loc[revolver, "f11"].median()).to_numpy()
    low, high = np.median(profit[revolver & ~risky]), np.median(profit[revolver & risky])
    print(f"  Among revolvers, median profit: low-risk {low:,.0f} vs high-risk {high:,.0f}"
          f" ({low / high:.1f}x)")
    print("  -> PCA puts revolve balance and risk on the same component (PC2), but they")
    print("     push profit in opposite directions. The profit model separates them.")

    plot_variance(explained)
    plot_loadings(loadings, explained)
    print("\n  saved figures/pca_variance.png and figures/pca_loadings.png")


if __name__ == "__main__":
    main()
