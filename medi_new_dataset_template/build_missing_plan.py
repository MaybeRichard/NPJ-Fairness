#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MeDi-style missing subgroup plan from train/test CSV files."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Training split CSV.")
    parser.add_argument("--test-csv", type=Path, required=True, help="Test split CSV.")
    parser.add_argument(
        "--class-column",
        type=str,
        default="primary_class",
        help="Main class column. Default: primary_class",
    )
    parser.add_argument(
        "--group-columns",
        type=str,
        nargs="+",
        required=True,
        help="Metadata columns used to define missing subgroups.",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Output missing_plan.csv path.")
    parser.add_argument(
        "--quota-column-prefixes",
        type=str,
        nargs="*",
        default=[],
        help="Optional prefixes used to duplicate n_generate into <prefix>_quota columns.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    value = str(text).strip().lower()
    value = value.replace("+", "plus")
    value = value.replace("<", "lt")
    value = value.replace(">", "gt")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def main() -> None:
    args = parse_args()
    group_columns = [args.class_column, *args.group_columns]

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    missing_columns = [col for col in group_columns if col not in train_df.columns or col not in test_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in train/test CSV: {missing_columns}")

    train_groups = set(map(tuple, train_df[group_columns].drop_duplicates().itertuples(index=False, name=None)))
    test_counts = test_df.groupby(group_columns).size().reset_index(name="n_generate")

    missing = test_counts[
        test_counts.apply(
            lambda row: tuple(row[col] for col in group_columns) not in train_groups,
            axis=1,
        )
    ].copy()

    if missing.empty:
        raise RuntimeError("No missing test groups were found relative to train.")

    def build_group_id(row: pd.Series) -> str:
        parts = [slugify(row[args.class_column])]
        for col in args.group_columns:
            parts.append(f"{slugify(col)}-{slugify(row[col])}")
        return "__".join(parts)

    missing["group_id"] = missing.apply(build_group_id, axis=1)

    for prefix in args.quota_column_prefixes:
        missing[f"{prefix}_quota"] = missing["n_generate"]

    missing = missing.sort_values(group_columns).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output_csv, index=False)
    print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
