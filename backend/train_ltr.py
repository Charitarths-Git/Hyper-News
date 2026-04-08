"""
Minimal LambdaMART training entry point for vNext ranking.

This script intentionally trains on lightweight exported feature rows rather
than trying to infer a full training dataset from raw app events in-line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from ltr import DEFAULT_LTR_MODEL_PATH


_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HyperNews LambdaMART scorer.")
    parser.add_argument("--features-csv", default="", help="CSV file containing feature rows.")
    parser.add_argument("--auto-export", action="store_true", help="Generate features first when --features-csv is not provided.")
    parser.add_argument("--export-source", choices=["auto", "db", "mind"], default="auto")
    parser.add_argument("--limit-impressions", type=int, default=2000)
    parser.add_argument("--label-col", default="label", help="Column containing binary or graded relevance labels.")
    parser.add_argument("--group-col", default="group_id", help="Column identifying ranking groups.")
    parser.add_argument("--output-model", default=DEFAULT_LTR_MODEL_PATH, help="Path to save the LightGBM model.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise SystemExit(f"lightgbm is required to train the LTR model: {exc}") from exc

    features_csv = str(args.features_csv or "").strip()
    if not features_csv:
        if not args.auto_export:
            raise SystemExit("Provide --features-csv or pass --auto-export.")
        features_csv = os.path.join(_BASE, "data", "ltr_features.auto.csv")
        export_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_ltr_features.py")
        command = [
            sys.executable,
            export_script,
            "--source",
            args.export_source,
            "--output-csv",
            features_csv,
            "--limit-impressions",
            str(args.limit_impressions),
        ]
        subprocess.run(command, check=True)

    frame = pd.read_csv(features_csv)
    if args.label_col not in frame.columns or args.group_col not in frame.columns:
        raise SystemExit(
            f"Expected columns '{args.label_col}' and '{args.group_col}' in {features_csv}"
        )

    excluded_cols = {args.label_col, args.group_col, "article_id", "candidate_source"}
    feature_cols = [column for column in frame.columns if column not in excluded_cols]
    train_x = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_y = frame[args.label_col].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    group_sizes = frame.groupby(args.group_col).size().tolist()

    dataset = lgb.Dataset(
        train_x.to_numpy(dtype=np.float32),
        label=train_y.to_numpy(dtype=np.float32),
        group=group_sizes,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": ["ndcg"],
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": 1,
    }
    booster = lgb.train(params, dataset, num_boost_round=200)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    booster.save_model(args.output_model)
    gain_importance = booster.feature_importance(importance_type="gain")
    total_gain = float(np.sum(gain_importance)) or 1.0
    signed_weights: dict[str, float] = {}
    for index, feature_name in enumerate(feature_cols):
        gain = float(gain_importance[index]) / total_gain
        column = train_x[feature_name]
        correlation = 0.0
        try:
            if float(column.std()) > 0.0:
                correlation = float(np.corrcoef(column.to_numpy(dtype=float), train_y.to_numpy(dtype=float))[0, 1])
        except Exception:
            correlation = 0.0
        sign = -1.0 if correlation < 0 else 1.0
        signed_weights[feature_name] = float(gain * sign)

    weights_path = f"{args.output_model}.weights.json"
    Path(weights_path).write_text(
        json.dumps({"weights": signed_weights, "features_csv": features_csv}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"saved_model": args.output_model, "saved_weights": weights_path, "features_csv": features_csv, "features": feature_cols}, indent=2))


if __name__ == "__main__":
    main()
