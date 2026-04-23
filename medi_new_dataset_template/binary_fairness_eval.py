#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairness_eval import FairnessEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness evaluation for a binary task from a CSV file.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="CSV with labels, scores, and group columns.")
    parser.add_argument("--target-column", type=str, default="y_true", help="Ground-truth label column.")
    parser.add_argument("--score-column", type=str, default="score", help="Predicted score / probability column.")
    parser.add_argument(
        "--group-columns",
        type=str,
        nargs="+",
        required=True,
        help="One or more metadata columns used as protected groups.",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default=None,
        help="If labels are not already 0/1, this value will be treated as positive.",
    )
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        default="best_f1",
        choices=["best_f1", "youden", "fixed"],
        help="Threshold selection strategy.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5, help="Threshold used when strategy=fixed.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write fairness outputs.")
    return parser.parse_args()


def normalize_binary_target(series: pd.Series, positive_label: str | None) -> pd.Series:
    if positive_label is not None:
        return (series.astype(str) == positive_label).astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique_values = sorted(set(numeric.astype(int).tolist()))
        if set(unique_values).issubset({0, 1}):
            return numeric.astype(int)

    raise ValueError(
        "Target labels are not binary 0/1. Please pass --positive-label to specify the positive class."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    work = df.copy()
    work["_binary_target"] = normalize_binary_target(work[args.target_column], args.positive_label)

    evaluator = FairnessEvaluator(task="binary")
    result = evaluator.evaluate_from_dataframe(
        work,
        schema={
            "target_column": "_binary_target",
            "score_column": args.score_column,
            "group_columns": args.group_columns,
        },
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
    )

    evaluator.export_json(result, args.output_dir / "fairness_summary.json")
    evaluator.export_tables(result, args.output_dir, stem="fairness_report")
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "predictions_csv": str(args.predictions_csv),
                "target_column": args.target_column,
                "score_column": args.score_column,
                "group_columns": args.group_columns,
                "positive_label": args.positive_label,
                "threshold_strategy": args.threshold_strategy,
                "fixed_threshold": args.fixed_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
