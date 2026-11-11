import pandas as pd
import numpy as np

PARAMS = {
    "discount_rate": 0.037,
    "interest_margin": 0.42,
    "supp_fee": 175,
    "reward_cost_per_pt": 0.006,
    "lounge_cost": 35,
    "cab_cost": 15,
    "lgd_x_pd_weight": 5.1,
    "collection_cost": 400,
    "cancel_cost": 30,
    # Cost per point of unredeemed rewards balance (f4).
    # 0 in the final model. Set to 0.001 to reproduce the 0.875 submission.
    "points_balance_cost": 0.0,
}

SPEND_COLS = ["f6", "f7", "f8", "f9", "f10"]
TRAVEL_5X = ["f6", "f9"]
OTHER_1X = ["f7", "f8", "f10"]

def prepare_data(filepath):
    frame = pd.read_csv(filepath)
    features = [f"f{i}" for i in range(1, 24)]
    frame[features] = frame[features].apply(
        lambda column: column.fillna(column.median())
    )
    return frame

def score_members(frame, params=PARAMS):
    spend_total = frame[SPEND_COLS].sum(axis=1)
    earned_points = (
        frame[TRAVEL_5X].sum(axis=1) * 5
        + frame[OTHER_1X].sum(axis=1)
    )

    income = (
        params["discount_rate"] * spend_total
        + params["interest_margin"] * frame["f1"]
        + params["supp_fee"] * frame["f19"]
    )

    expenses = (
        params["reward_cost_per_pt"] * earned_points
        + params["lounge_cost"] * frame["f13"]
        + frame["f14"]
        + params["cab_cost"] * frame["f15"]
        + frame["f16"]
        + params["lgd_x_pd_weight"] * frame["f11"] * frame["f1"]
        + params["collection_cost"] * frame["f3"]
        + params["cancel_cost"] * frame["f2"]
        + params["points_balance_cost"] * frame["f4"]
    )

    return income - expenses

def main():
    source = "data/campus_challenge_r1_data.csv"
    data = prepare_data(source)
    data["Prediction"] = score_members(data).round(2)

    predictions = data.loc[:, ["id", "Prediction"]].rename(columns={"id": "ID"})
    predictions.to_csv("predictions.csv", index=False)

    cutoff = data["Prediction"].quantile(0.80)
    print(f"Scored {len(data):,} members. Top-20% cutoff score: {cutoff:,.0f}")
    print("Wrote predictions.csv")

if __name__ == "__main__":
    main()
