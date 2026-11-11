import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = [f"f{i}" for i in range(1, 24)]

def save_heatmap(matrix, output, title, size, mask=None):
    plt.figure(figsize=size)
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=16, pad=12)
    plt.tight_layout()
    plt.savefig(output, dpi=130, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs("figures", exist_ok=True)
    data = pd.read_csv("data/campus_challenge_r1_data.csv")
    correlation = data.loc[:, FEATURES].corr()

    sns.set_style("white")

    save_heatmap(
        correlation,
        "figures/correlation_heatmap.png",
        "Correlation Heatmap of American Express Dataset",
        (14, 12)
    )

    lower_mask = np.triu(np.ones(correlation.shape, dtype=bool))
    save_heatmap(
        correlation,
        "figures/correlation_heatmap_triangle.png",
        "Feature Correlation Heatmap (lower triangle)",
        (13, 11),
        lower_mask
    )

    connectedness = correlation.abs().sum().sub(1).sort_values(ascending=False)
    leading = connectedness.iloc[:15]

    plt.figure(figsize=(10, 5))
    plt.bar(leading.index, leading.to_numpy(), color="#1f77b4")
    plt.ylabel("Total Absolute Correlation")
    plt.title("Most Connected Features", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/most_connected_features.png", dpi=130, bbox_inches="tight")
    plt.close()

    print("Wrote 3 figures to figures/")
    print("\nMost connected features (total |correlation|):")
    for feature, value in leading.items():
        print(f"  {feature:>4}: {value:.2f}")

if __name__ == "__main__":
    main()
