#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairness_eval import FairnessEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-vs-rest fairness evaluation for single-label multiclass predictions."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True, help="CSV with y_true and score_<class> columns.")
    parser.add_argument("--target-column", type=str, default="y_true", help="Ground-truth label column.")
    parser.add_argument(
        "--score-prefix",
        type=str,
        default="score_",
        help="Prefix for per-class score columns such as score_normal.",
    )
    parser.add_argument(
        "--group-columns",
        type=str,
        nargs="+",
        required=True,
        help="One or more metadata columns used as protected groups.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Optional JSON mapping class name to a fixed threshold.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write fairness outputs.")
    return parser.parse_args()


def is_scalar_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def extract_metric_keys(result: dict) -> tuple[list[str], list[str]]:
    exclude_overall = {
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "tpr",
        "tnr",
        "fpr",
        "fnr",
    }
    overall_keys = [
        key
        for key, value in result.get("overall", {}).items()
        if is_scalar_number(value) and key not in exclude_overall
    ]
    fairness_keys = [key for key, value in result.get("fairness", {}).items() if is_scalar_number(value)]
    return overall_keys, fairness_keys


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    score_columns = [col for col in df.columns if col.startswith(args.score_prefix)]
    if not score_columns:
        raise ValueError(f"No score columns found with prefix {args.score_prefix!r}")

    classes = [col[len(args.score_prefix) :] for col in score_columns]
    thresholds = {}
    if args.thresholds_json is not None and args.thresholds_json.exists():
        thresholds = json.loads(args.thresholds_json.read_text(encoding="utf-8"))

    evaluator = FairnessEvaluator(task="binary")
    per_class: dict[str, dict] = {}
    summary_rows = []
    class_support = df[args.target_column].value_counts().to_dict()
    overall_metric_keys: list[str] = []
    fairness_metric_keys: list[str] = []

    for class_name in classes:
        work = df.copy()
        work["binary_target"] = (work[args.target_column] == class_name).astype(int)
        work["binary_score"] = work[f"{args.score_prefix}{class_name}"]
        threshold = float(thresholds.get(class_name, 0.5))

        result = evaluator.evaluate_from_dataframe(
            work,
            schema={
                "target_column": "binary_target",
                "score_column": "binary_score",
                "group_columns": args.group_columns,
            },
            threshold_strategy="fixed",
            fixed_threshold=threshold,
        )

        per_class[class_name] = result
        if not overall_metric_keys and not fairness_metric_keys:
            overall_metric_keys, fairness_metric_keys = extract_metric_keys(result)

        row = {
            "class_name": class_name,
            "support": int(class_support.get(class_name, 0)),
        }
        for key in overall_metric_keys:
            row[f"overall_{key}"] = result["overall"].get(key, float("nan"))
        for key in fairness_metric_keys:
            row[key] = result["fairness"].get(key, float("nan"))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("class_name").reset_index(drop=True)
    weights = summary_df["support"].to_numpy(dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)

    aggregate_keys = [f"overall_{key}" for key in overall_metric_keys] + fairness_metric_keys
    macro = {}
    weighted = {}
    for key in aggregate_keys:
        values = summary_df[key].to_numpy(dtype=float)
        macro[key] = float(np.nanmean(values))
        weighted[key] = float(np.nansum(values * weights))

    overall = {
        "classes": classes,
        "group_columns": args.group_columns,
        "overall_metric_keys": overall_metric_keys,
        "fairness_metric_keys": fairness_metric_keys,
        "macro": macro,
        "weighted": weighted,
    }

    (args.output_dir / "ovr_summary.csv").write_text(summary_df.to_csv(index=False), encoding="utf-8")
    try:
        markdown_text = summary_df.to_markdown(index=False)
    except ImportError:
        markdown_text = summary_df.to_string(index=False)
    (args.output_dir / "ovr_summary.md").write_text(markdown_text, encoding="utf-8")
    (args.output_dir / "ovr_summary.json").write_text(
        json.dumps({"overall": overall, "per_class": per_class}, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
