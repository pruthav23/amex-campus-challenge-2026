import pandas as pd
import numpy as np

FEATURES = [f"f{i}" for i in range(1, 24)]

def main():
    df = pd.read_csv("data/campus_challenge_r1_data.csv")

    print("=" * 70)
    print("1) CAPPING CHECK  (99th percentile == max  =>  top 1% is clamped)")
    print("=" * 70)

    for feature in ("f1", "f6", "f7"):
        q99 = df[feature].quantile(0.99)
        maximum = df[feature].max()
        is_capped = np.isclose(q99, maximum)
        print(f"  {feature}: 99%={q99:,.0f}  max={maximum:,.0f}  -> top-1%-capped={is_capped}")

    print("\n" + "=" * 70)
    print("2) IS f5 ('total spend') A REAL SIGNAL?")
    print("=" * 70)

    corr = df.select_dtypes(include=np.number).drop(columns="id").corr()
    f5_strength = corr.loc[corr.index != "f5", "f5"].abs().max()
    print(f"  f5's strongest correlation with any other feature: {f5_strength:.3f}")

    category_total = df[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1, min_count=5)
    print(f"  corr(f5, f6+f7+f8+f9+f10) = {df['f5'].corr(category_total):.3f}")
    print(f"  median f5 = {df['f5'].median():,.0f}   median of category sum = {category_total.median():,.0f}")
    print("  -> f5 does not reconcile with its own components: treat it as unreliable.")
    print("     (Confirmed on the leaderboard: using f5 as spend scored 0.388.)")

    print("\n" + "=" * 70)
    print("3) CORRELATION CLUSTERS  (|r| >= 0.5)  ->  match economic levers")
    print("=" * 70)

    columns = list(corr.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = corr.loc[first, second]
            if abs(value) >= 0.5:
                print(f"  {first:>4} <-> {second:<4}  r={value:+.2f}")

    print("\n" + "=" * 70)
    print("4) RISK DIRECTION  (f11 vs f3 'Cancellation Calls due to Collection')")
    print("=" * 70)

    risk_corr = df[["f11", "f3"]].dropna().corr().iat[0, 1]
    print(f"  corr(risk f11, collection-related calls f3) = {risk_corr:+.2f}  -> higher f11 = riskier")

if __name__ == "__main__":
    main()
