"""Check which model version produced a past submission.

Compares the top 20% of a submission file with the top 20% of the
current profitability model, and reports how many members they share.

Usage:
    python compare_submission.py path/to/submission.csv
    python compare_submission.py path/to/submission.xlsx --points-balance-cost 0.001

The submission may hold either 0/1 flags (1 = predicted top 20%) or raw
profitability scores; scores are converted to a top-20% flag first.
"""

import argparse

import pandas as pd

from profitability_framework import PARAMS, prepare_data, score_members

DATA_PATH = "data/campus_challenge_r1_data.csv"
TOP_SHARE = 0.20


def top_flags(scores, count):
    ranks = scores.rank(method="first", ascending=False)
    return (ranks <= count).to_numpy()


def load_submission(path, member_ids):
    if path.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path, sheet_name=0)
    else:
        frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    values = frame.set_index("ID")["Prediction"].reindex(member_ids)
    return values.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--points-balance-cost", type=float,
                        default=PARAMS["points_balance_cost"])
    parser.add_argument("--discount-rate", type=float,
                        default=PARAMS["discount_rate"])
    args = parser.parse_args()

    data = prepare_data(DATA_PATH)
    count = int(len(data) * TOP_SHARE)

    submitted = load_submission(args.submission, data["id"])
    if set(submitted.dropna().unique()) <= {0, 1}:
        submitted_top = submitted.fillna(0).astype(int).to_numpy().astype(bool)
    else:
        submitted_top = top_flags(submitted, count)

    params = dict(PARAMS,
                  points_balance_cost=args.points_balance_cost,
                  discount_rate=args.discount_rate)
    model_top = top_flags(score_members(data, params), count)

    shared = (model_top & submitted_top).sum()
    print(f"Members in submission's top 20%: {submitted_top.sum():,}")
    print(f"Shared with model's top 20%:     {shared:,} "
          f"({shared / submitted_top.sum():.2%})")
    print("Above ~99.5% means this model version produced the submission.")


if __name__ == "__main__":
    main()
