"""Standalone fairness evaluation in a single file.

Implements every named metric from four benchmark papers plus NC 2023 PFD.
Pure numpy / pandas / scikit-learn; no PyTorch, fairlearn, or aif360.

Main entry point
----------------
    FairnessEvaluator(task="binary" | "multilabel" | "pairwise" | "segmentation")

Metric catalog (paper source in brackets)
-----------------------------------------

Binary classification
    Utility (higher is better unless noted):
        auroc        - Area under ROC curve [Harvard-GF, FairMedFM, FairFedMed]
        ece          - Expected calibration error (lower is better) [FairMedFM]
        accuracy     - Overall accuracy [Harvard-GF]
        sensitivity  - TPR at chosen threshold [Fairer-AI]
        specificity  - TNR at chosen threshold [Fairer-AI]
    Fairness (lower is better unless noted):
        auc_gap_maxmin        - max-min subgroup AUROC. Paper name: AUC_Delta [FairMedFM]
        ece_gap_maxmin        - max-min subgroup ECE. Paper name: ECE_Delta [FairMedFM]
        equal_opportunity_gap - max-min subgroup TPR. Paper name: DEO [Harvard-GF]
        equalized_odds_gap    - max of TPR gap and FPR gap. Paper names:
                                DEOdds [Harvard-GF], EqOdds [FairMedFM], EOD [FairFedMed]
        dpd                   - max-min selection rate (tp+fp)/count.
                                Paper names: DPD [Harvard-GF], SPD [FairFedMed]
        es_auc      (higher)  - AUC / (1 + sum_g |AUC - AUC_g|).
                                Paper names: ES-AUC [Harvard-GF], AUC_ES [FairMedFM]
        es_accuracy (higher)  - same template on accuracy. Paper name: ES-Acc [Harvard-GF]

Multilabel classification (Fairer-AI, Nat Commun 2024)
    Utility:
        mAP, avg_auroc, avg_acc, avg_sensitivity, avg_specificity
        ap_per_class, auroc_per_class, avg_acc_per_class
    Fairness (per-class variants also returned):
        deltaD_per_class  - (max-min)/mean of AP across groups. Paper name: Delta D
        deltaA            - mean of deltaD. Paper name: Delta A (average screening disparity)
        deltaM            - max of deltaD. Paper name: Delta M (max screening disparity)
        PQD               - min/max of avg_acc across groups (higher is better)
        DPM               - mean over classes of min/max of (tp+fp)/positive_count
        EOM               - mean over classes of min/max of TPR (tp/positive_count)

Pairwise fairness (Nat Commun 2023 PFD paper)
    Utility:
        pairwise_overall               - AUC using all positives vs all negatives
        pairwise_within_group          - within-group AUC per subgroup
    Fairness:
        pairwise_subgroup_vs_all_negative - per subgroup: group positives vs all negatives
        pairwise_fairness_difference_pfd  - max-min of the above. Paper name: PFD

Segmentation (FairMedFM)
    Utility:
        mean_dice, min_dice, max_dice. Paper name: DSC
    Fairness:
        delta_dice     - max-min (or |g0-g1| for 2 groups). Paper name: DSC_Delta
        std_dice       - stddev across group means. Paper name: DSC_STD
        skewness_dice  - (1-min_dice) / (1-max_dice). Paper name: DSC_Skew
        es_dice        - mean_dice / (1 + std_dice) (higher is better). Paper name: DSC_ES

Input helpers (same metric set, different I/O)
    evaluator.evaluate(y_true=..., y_score=..., groups=...)
    evaluator.evaluate_from_dataframe(df, schema={...})
    evaluator.evaluate_from_csv(path, schema={...})
    evaluator.evaluate_from_npz(path, key_map={...})
    evaluator.export_json(result, path)
    evaluator.export_tables(result, output_dir)

Schema keys by task
    binary / pairwise:
        target_column, score_column, group_column or group_columns
    multilabel:
        target_columns + score_columns (multi-col form), or
        target_column + score_column with array_columns=[...] (list-in-cell form);
        plus group_column / group_columns
    segmentation:
        dice_column (precomputed dice per sample), or
        pred_column + true_column (list-in-cell form);
        plus group_column / group_columns

References (cited inline as short keys)
---------------------------------------
[Harvard-GF]   Luo et al., "Harvard Glaucoma Fairness: A Retinal Nerve Disease
               Dataset for Fairness Learning and Fair Identity Normalization",
               IEEE TMI 2023/2024. arXiv:2306.09264.
               Official code: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF
               (src/modules.py for equity_scaled_AUC, equity_scaled_accuracy).
[FairMedFM]    Jin et al., "FairMedFM: Fairness Benchmarking for Medical Imaging
               Foundation Models", NeurIPS 2024 D&B. arXiv:2407.00983.
               Official code: https://github.com/FairMedFM/FairMedFM
               (utils/metrics.py for organize_results, evaluate_seg).
[Fairer-AI]    Tan et al., "Fairer AI in ophthalmology via implicit fairness
               learning for mitigating sexism and ageism", Nat Commun 2024,
               15:4750. doi:10.1038/s41467-024-48972-0.
               Official code: https://github.com/mintanwei/Fairer-AI
               (metrics/multi_evalute.py::cal_metrics).
[FairFedMed]   Li et al., "FairFedMed: Benchmarking Group Fairness in Federated
               Medical Imaging With FairLoRA", IEEE TMI 2026, 45(4):1337.
               Official code: https://github.com/Harvard-AI-and-Robotics-Lab/FairFedMed
               (evaluation/metrics.py for equity_scaled_AUC, compute_between_group_disparity).
[NC2023-PFD]   Weng et al., "Improving model fairness in image-based computer-aided
               diagnosis", Nat Commun 2023. Introduces Pairwise Fairness Difference.

Direction cheat sheet
---------------------
Higher-is-better (performance or equity-scaled utility):
    auroc, accuracy, sensitivity, specificity, mAP, avg_acc, avg_auroc,
    avg_sensitivity, avg_specificity, mean_dice, min_dice, max_dice,
    pairwise_overall, pairwise_within_group, pairwise_subgroup_vs_all_negative,
    PQD, DPM, EOM, es_auc, es_accuracy, es_dice.
Lower-is-better (gap / disparity, closer to 0 means more fair):
    ece, auc_gap_maxmin, ece_gap_maxmin, equal_opportunity_gap,
    equalized_odds_gap, dpd, delta_dice, std_dice, skewness_dice,
    deltaD_per_class, deltaA, deltaM, pairwise_fairness_difference_pfd.

Minimal example
---------------
    import numpy as np
    from fairness_eval import FairnessEvaluator

    y_true  = np.array([0,1,0,1,1,0,1,0])
    y_score = np.array([.1,.9,.3,.7,.8,.2,.6,.4])
    groups  = np.array(["M","M","M","M","F","F","F","F"])
    result = FairnessEvaluator("binary").evaluate(
        y_true=y_true, y_score=y_score, groups=groups,
    )
    result["overall"]["auroc"]
    result["fairness"]["equalized_odds_gap"]
    result["fairness"]["es_auc"]
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

SUPPORTED_METRICS: dict[str, list[str]] = {
    "binary": [
        "auroc",                  # [Harvard-GF] [FairMedFM] [FairFedMed] [Fairer-AI]
        "ece",                    # [FairMedFM] §3.4 predictive-alignment fairness (base for ECE_Δ)
        "accuracy",               # [Harvard-GF] Table II-VIII "Overall Acc"
        "sensitivity",            # [Fairer-AI]
        "specificity",            # [Fairer-AI]
        "auc_gap_maxmin",         # [FairMedFM] §3.4 "AUC_Δ"
        "ece_gap_maxmin",         # [FairMedFM] §3.4 "ECE_Δ"
        "equal_opportunity_gap",  # [Harvard-GF] "DEO"; = fairlearn.equal_opportunity_difference
        "equalized_odds_gap",     # [Harvard-GF] DEOdds / [FairMedFM] EqOdds / [FairFedMed] EOD
        "es_auc",                 # [Harvard-GF] Eq.(7) ES-AUC; [FairMedFM] AUC_ES
        "es_accuracy",            # [Harvard-GF] Table II-VIII "Overall ES-Acc"
        "dpd",                    # [Harvard-GF] DPD; [FairFedMed] SPD
    ],
    "multilabel": [
        "mAP",              # [Fairer-AI]
        "deltaD_per_class", # [Fairer-AI] "ΔD" screening quality disparity
        "deltaA",           # [Fairer-AI] "ΔA" average screening disparity
        "deltaM",           # [Fairer-AI] "ΔM" max screening disparity
        "PQD",              # [Fairer-AI] predictive quality disparity
        "DPM",              # [Fairer-AI] demographic parity metric (code version)
        "EOM",              # [Fairer-AI] equality of opportunity metric
    ],
    "pairwise": [
        "pairwise_subgroup_vs_all_negative",  # [NC2023-PFD] per-subgroup pairwise AUC
        "pairwise_fairness_difference_pfd",   # [NC2023-PFD] PFD
    ],
    "segmentation": [
        "mean_dice",      # [FairMedFM] "DSC"
        "delta_dice",     # [FairMedFM] "DSC_Δ"
        "skewness_dice",  # [FairMedFM] "DSC_Skew"
        "std_dice",       # [FairMedFM] "DSC_STD"
        "es_dice",        # [FairMedFM] "DSC_ES"
    ],
}


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def safe_roc_auc(y_true: Any, y_score: Any) -> float:
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_score_arr))


def safe_average_precision(y_true: Any, y_score: Any) -> float:
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(average_precision_score(y_true_arr, y_score_arr))


def select_threshold(
    y_true: Any,
    y_score: Any,
    strategy: str = "best_f1",
    fixed_threshold: float = 0.5,
) -> float:
    """Choose a decision threshold for binary scores.

    Strategies:
        "best_f1"  - threshold that maximises F1 on the PR curve (FairMedFM default).
        "youden"   - threshold that maximises TPR - FPR on the ROC curve.
        "fixed"    - return `fixed_threshold` without looking at data.
    """
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)

    if strategy == "fixed":
        return float(fixed_threshold)
    if np.unique(y_true_arr).size < 2:
        return float(fixed_threshold)
    if strategy == "best_f1":
        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
        if thresholds.size == 0:
            return float(fixed_threshold)
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        return float(thresholds[int(np.nanargmax(f1))])
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
        if thresholds.size == 0:
            return float(fixed_threshold)
        return float(thresholds[int(np.nanargmax(tpr - fpr))])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def confusion_from_scores(y_true: Any, y_score: Any, threshold: float) -> dict[str, float]:
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)
    y_pred = (y_score_arr >= threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tpr": float(sensitivity),
        "tnr": float(specificity),
        "precision": float(precision),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "accuracy": float(accuracy),
    }


def expected_calibration_error(
    y_true: Any,
    y_score: Any,
    n_bins: int = 10,
    binning: str = "equal_width",
) -> float:
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)
    if y_true_arr.size == 0:
        return float("nan")

    if binning == "quantile":
        edges = np.quantile(y_score_arr, np.linspace(0.0, 1.0, n_bins + 1))
    elif binning == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown binning mode: {binning}")

    ece = 0.0
    total = float(y_true_arr.size)
    for low, high in zip(edges[:-1], edges[1:]):
        if high == edges[-1]:
            mask = (y_score_arr >= low) & (y_score_arr <= high)
        else:
            mask = (y_score_arr >= low) & (y_score_arr < high)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_score_arr[mask]))
        acc = float(np.mean(y_true_arr[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def nanmax_minus_nanmin(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def nanstd(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def group_indices(groups: Any) -> list[Any]:
    return list(np.unique(to_numpy(groups)))


def evaluate_binary_classification(
    y_true: Any,
    y_score: Any,
    groups: Any | None = None,
    threshold_strategy: str = "best_f1",
    fixed_threshold: float = 0.5,
    ece_bins: int = 10,
    ece_binning: str = "equal_width",
) -> dict[str, Any]:
    """Evaluate a binary-classification head with optional subgroup fairness.

    Args:
        y_true: shape (N,) array of 0/1 labels.
        y_score: shape (N,) array of positive-class scores in [0, 1].
        groups: optional shape (N,) array of group labels. If None, only overall
            utility metrics are returned.
        threshold_strategy: see `select_threshold`. A pooled threshold is chosen
            once on the full dataset and reused for every subgroup, matching the
            FairMedFM reporting convention.
        fixed_threshold: used when `threshold_strategy="fixed"`.
        ece_bins, ece_binning: parameters forwarded to
            `expected_calibration_error`.

    Returns a dict with keys:
        overall   - auroc, ece, accuracy, sensitivity, specificity,
                    plus tpr, fpr, tp, fp (kept as inputs to fairness metrics).
        groups    - per-group mirror of `overall` plus sample counts
                    (count, positive_count, negative_count).
        fairness  - auc_gap_maxmin, ece_gap_maxmin, equal_opportunity_gap,
                    equalized_odds_gap, dpd, es_auc, es_accuracy.
        metadata  - threshold details and semantics notes.
    """
    y_true_arr = to_numpy(y_true).astype(int).reshape(-1)
    y_score_arr = to_numpy(y_score).astype(float).reshape(-1)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have the same shape")

    group_arr = None
    if groups is not None:
        group_arr = to_numpy(groups).reshape(-1)
        if group_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError("groups must have the same number of samples as y_true")

    threshold = select_threshold(y_true_arr, y_score_arr, strategy=threshold_strategy, fixed_threshold=fixed_threshold)
    overall_conf = confusion_from_scores(y_true_arr, y_score_arr, threshold)
    # Keep only paper-reported fields + inputs needed for fairness metrics (TPR, FPR for gaps; TP, FP for DPD).
    _paper_conf_keys = {"accuracy", "sensitivity", "specificity", "tpr", "fpr", "tp", "fp"}
    overall = {
        # Source: standard sklearn; reported as "AUC" by [Harvard-GF], [FairMedFM], [FairFedMed], [Fairer-AI].
        "auroc": safe_roc_auc(y_true_arr, y_score_arr),
        # Source: Guo et al. ICML 2017; reported as "ECE" by [FairMedFM] §3.4 (predictive-alignment fairness).
        "ece": expected_calibration_error(y_true_arr, y_score_arr, n_bins=ece_bins, binning=ece_binning),
        # accuracy [Harvard-GF], sensitivity/specificity [Fairer-AI], tpr/fpr/tp/fp kept as
        # inputs to DEO/DEOdds/DPD — tn/fn/tnr/fnr/precision/threshold suppressed to match paper scope.
        **{k: v for k, v in overall_conf.items() if k in _paper_conf_keys},
    }

    if group_arr is None:
        return {"overall": overall, "groups": {}, "fairness": {}, "metadata": {"threshold": threshold}}

    groups_out: dict[Any, dict[str, float]] = {}
    for group_value in group_indices(group_arr):
        mask = group_arr == group_value
        group_conf = confusion_from_scores(y_true_arr[mask], y_score_arr[mask], threshold)
        groups_out[group_value] = {
            "count": float(np.sum(mask)),
            "positive_count": float(np.sum(y_true_arr[mask] == 1)),
            "negative_count": float(np.sum(y_true_arr[mask] == 0)),
            # Per-group AUC is reported by [Harvard-GF] (Asian/Black/White), [FairMedFM] (AUC_Female/AUC_Male), [FairFedMed].
            "auroc": safe_roc_auc(y_true_arr[mask], y_score_arr[mask]),
            "ece": expected_calibration_error(
                y_true_arr[mask], y_score_arr[mask], n_bins=ece_bins, binning=ece_binning
            ),
            **{k: v for k, v in group_conf.items() if k in _paper_conf_keys},
        }

    subgroup_auc = [v["auroc"] for v in groups_out.values()]
    subgroup_acc = [v["accuracy"] for v in groups_out.values()]
    subgroup_ece = [v["ece"] for v in groups_out.values()]
    subgroup_tpr = [v["tpr"] for v in groups_out.values()]
    subgroup_fpr = [v["fpr"] for v in groups_out.values()]
    subgroup_selection = [
        (v["tp"] + v["fp"]) / v["count"] if v["count"] > 0 else float("nan")
        for v in groups_out.values()
    ]

    fairness = {
        # Source: [FairMedFM] §3.4 + Table 2 (AUC_Δ). Paper symbol: AUC_Δ.
        "auc_gap_maxmin": nanmax_minus_nanmin(subgroup_auc),
        # Source: [FairMedFM] §3.4 + Table 2 (ECE_Δ). Paper symbol: ECE_Δ.
        "ece_gap_maxmin": nanmax_minus_nanmin(subgroup_ece),
        # Source: [Harvard-GF] §II "Difference in Equal Opportunity" (DEO);
        #         = fairlearn.metrics.equal_opportunity_difference.
        "equal_opportunity_gap": nanmax_minus_nanmin(subgroup_tpr),
        # Source: [Harvard-GF] DEOdds; [FairMedFM] Table 2 EqOdds; [FairFedMed] §V.A.3 EOD.
        #         = fairlearn.metrics.equalized_odds_difference (worst-case of TPR/FPR gaps).
        "equalized_odds_gap": float(
            np.nanmax([nanmax_minus_nanmin(subgroup_tpr), nanmax_minus_nanmin(subgroup_fpr)])
        ),
        # Source: [Harvard-GF] Fig.1 + Table II-VIII "Overall DPD"; [FairFedMed] Table I-VI "SPD".
        #         = max-min of selection rate τ(A) = E[ŷ=1 | a=A] = (tp+fp)/count.
        "dpd": nanmax_minus_nanmin(subgroup_selection),
        # Source: [Harvard-GF] Eq.(7) ES-AUC; [FairMedFM] Table 2 AUC_ES.
        #         Code: Harvard-GF src/modules.py::equity_scaled_AUC;
        #               FairFedMed evaluation/metrics.py::equity_scaled_AUC.
        "es_auc": float(
            overall["auroc"] / (1.0 + np.nansum(np.abs(np.asarray(subgroup_auc) - overall["auroc"])))
        ),
        # Source: [Harvard-GF] Table II-VIII "Overall ES-Acc".
        #         Code: Harvard-GF src/modules.py::equity_scaled_accuracy.
        "es_accuracy": float(
            overall["accuracy"] / (1.0 + np.nansum(np.abs(np.asarray(subgroup_acc) - overall["accuracy"])))
        ),
    }

    return {
        "overall": overall,
        "groups": groups_out,
        "fairness": fairness,
        "metadata": {
            "threshold_strategy": threshold_strategy,
            "threshold": threshold,
            "eqopp_semantics": "max-min TPR across groups (Harvard-GF DEO, fairlearn equal_opportunity_difference)",
            "eqodds_semantics": "worst-case of TPR and FPR gaps (Harvard-GF DEOdds, FairMedFM EqOdds, FairFedMed EOD)",
            "dpd_semantics": "max-min of selection rate (tp+fp)/count (Harvard-GF DPD, FairFedMed SPD)",
            "es_semantics": "overall_metric / (1 + sum_g |overall - group_metric|) (Harvard-GF ES-M, FairMedFM *_ES)",
        },
    }


def _select_class_threshold(y_true: Any, y_score: Any, strategy: str, fixed_threshold: float) -> float:
    if strategy == "per_group_youden":
        strategy = "youden"
    return select_threshold(y_true, y_score, strategy=strategy, fixed_threshold=fixed_threshold)


def _safe_ratio_of_min_to_max(values: np.ndarray, axis: int) -> np.ndarray:
    out = []
    for idx in range(values.shape[1 - axis] if values.ndim == 2 else 0):
        arr = values[:, idx] if axis == 0 else values[idx]
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        out.append(np.nan if arr.size == 0 else float(np.min(arr) / np.clip(np.max(arr), 1e-15, None)))
    return np.asarray(out, dtype=float)


def _safe_delta_per_class(ap_array: np.ndarray) -> np.ndarray:
    out = []
    for class_idx in range(ap_array.shape[1]):
        arr = ap_array[:, class_idx]
        arr = arr[~np.isnan(arr)]
        out.append(np.nan if arr.size == 0 else float((np.max(arr) - np.min(arr)) / np.clip(np.mean(arr), 1e-15, None)))
    return np.asarray(out, dtype=float)


def evaluate_multilabel_fairness(
    y_true: Any,
    y_score: Any,
    groups: Any,
    threshold_strategy: str = "per_group_youden",
    fixed_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate multilabel classification with Fairer-AI fairness metrics.

    Reproduces the Fairer-AI (Nat Commun 2024) public-code definitions of
    PQD, DPM, EOM, and Delta D / A / M. Per-class and per-group matrices are
    also returned.

    Args:
        y_true: shape (N, C) binary label matrix.
        y_score: shape (N, C) score matrix (probabilities or logits).
        groups: shape (N,) group labels.
        threshold_strategy:
            "per_group_youden" (default, Fairer-AI convention): each (class,
                group) cell picks its own Youden threshold.
            "youden" / "best_f1" / "fixed": a single threshold per class is
                shared across groups.
        fixed_threshold: used when `threshold_strategy="fixed"`.

    Returns a dict with `overall`, `groups`, `fairness`, `metadata`. Paper
    symbol -> code field: Delta D -> deltaD_per_class, Delta A -> deltaA,
    Delta M -> deltaM, PQD / DPM / EOM -> same name.
    """
    y_true_arr = to_numpy(y_true).astype(int)
    y_score_arr = to_numpy(y_score).astype(float)
    group_arr = to_numpy(groups).reshape(-1)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if y_true_arr.ndim != 2:
        raise ValueError("y_true and y_score must be 2D arrays for multilabel evaluation")
    if y_true_arr.shape[0] != group_arr.shape[0]:
        raise ValueError("groups must have the same number of samples as y_true")

    n_samples, n_classes = y_true_arr.shape
    unique_groups = group_indices(group_arr)
    ap_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    auc_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    specificity_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    sensitivity_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    fpr_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    fnr_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    tp_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    fp_array = np.full((len(unique_groups), n_classes), np.nan, dtype=float)
    positive_count_array = np.zeros((len(unique_groups), n_classes), dtype=float)

    pooled_ap = np.full(n_classes, np.nan, dtype=float)
    pooled_auc = np.full(n_classes, np.nan, dtype=float)
    pooled_specificity = np.full(n_classes, np.nan, dtype=float)
    pooled_sensitivity = np.full(n_classes, np.nan, dtype=float)
    thresholds: dict[str, float] = {}

    for class_idx in range(n_classes):
        pooled_ap[class_idx] = safe_average_precision(y_true_arr[:, class_idx], y_score_arr[:, class_idx])
        pooled_auc[class_idx] = safe_roc_auc(y_true_arr[:, class_idx], y_score_arr[:, class_idx])
        pooled_threshold = _select_class_threshold(
            y_true_arr[:, class_idx], y_score_arr[:, class_idx], threshold_strategy, fixed_threshold
        )
        thresholds[f"class_{class_idx}"] = pooled_threshold
        pooled_conf = confusion_from_scores(y_true_arr[:, class_idx], y_score_arr[:, class_idx], pooled_threshold)
        pooled_specificity[class_idx] = pooled_conf["specificity"]
        pooled_sensitivity[class_idx] = pooled_conf["sensitivity"]

        for group_idx, group_value in enumerate(unique_groups):
            mask = group_arr == group_value
            y_true_group = y_true_arr[mask, class_idx]
            y_score_group = y_score_arr[mask, class_idx]
            positive_count_array[group_idx, class_idx] = float(np.sum(y_true_group == 1))
            ap_array[group_idx, class_idx] = safe_average_precision(y_true_group, y_score_group)
            auc_array[group_idx, class_idx] = safe_roc_auc(y_true_group, y_score_group)

            threshold = (
                _select_class_threshold(y_true_group, y_score_group, "youden", fixed_threshold)
                if threshold_strategy == "per_group_youden"
                else pooled_threshold
            )
            conf = confusion_from_scores(y_true_group, y_score_group, threshold)
            specificity_array[group_idx, class_idx] = conf["specificity"]
            sensitivity_array[group_idx, class_idx] = conf["sensitivity"]
            fpr_array[group_idx, class_idx] = conf["fpr"]
            fnr_array[group_idx, class_idx] = conf["fnr"]
            tp_array[group_idx, class_idx] = conf["tp"]
            fp_array[group_idx, class_idx] = conf["fp"]

    map_per_group = np.nanmean(ap_array, axis=1)
    avg_acc_array = 0.5 * (specificity_array + sensitivity_array)
    avg_acc_per_group = np.nanmean(avg_acc_array, axis=1)
    avg_acc_per_class = 0.5 * (pooled_specificity + pooled_sensitivity)
    demo_array = (tp_array + fp_array) / np.clip(positive_count_array, 1e-15, None)
    eo_array = tp_array / np.clip(positive_count_array, 1e-15, None)
    delta_d_per_class = _safe_delta_per_class(ap_array)

    return {
        "overall": {
            # Source: [Fairer-AI] "mAP" — main multilabel utility metric.
            "mAP": float(np.nanmean(pooled_ap)),
            # Source: [Fairer-AI] cal_metrics L229 — per-class 0.5*(spec+sens) then mean.
            "avg_acc": float(np.nanmean(avg_acc_per_class)),
            # Source: [Fairer-AI] reported as "Specificity".
            "avg_specificity": float(np.nanmean(pooled_specificity)),
            # Source: [Fairer-AI] reported as "Sensitivity".
            "avg_sensitivity": float(np.nanmean(pooled_sensitivity)),
            # Source: [Fairer-AI] reported as "AUC" (macro-averaged).
            "avg_auroc": float(np.nanmean(pooled_auc)),
            "ap_per_class": pooled_ap.tolist(),
            "auroc_per_class": pooled_auc.tolist(),
            "avg_acc_per_class": avg_acc_per_class.tolist(),
        },
        "groups": {
            group_value: {
                "mAP": float(map_per_group[group_idx]),
                "avg_acc": float(avg_acc_per_group[group_idx]),
                "auroc_per_class": auc_array[group_idx].tolist(),
                "ap_per_class": ap_array[group_idx].tolist(),
                "specificity_per_class": specificity_array[group_idx].tolist(),
                "sensitivity_per_class": sensitivity_array[group_idx].tolist(),
                "fpr_per_class": fpr_array[group_idx].tolist(),
                "fnr_per_class": fnr_array[group_idx].tolist(),
            }
            for group_idx, group_value in enumerate(unique_groups)
        },
        "fairness": {
            # Source: [Fairer-AI] metrics/multi_evalute.py::cal_metrics L269-270 — "ΔD" (screening quality disparity).
            "deltaD_per_class": delta_d_per_class.tolist(),
            # Source: [Fairer-AI] cal_metrics L274 — "ΔA" (average screening disparity).
            "deltaA": float(np.nanmean(delta_d_per_class)),
            # Source: [Fairer-AI] cal_metrics L275 — "ΔM" (max screening disparity).
            "deltaM": float(np.nanmax(delta_d_per_class)),
            # Source: [Fairer-AI] cal_metrics L232 — "PQD" (predictive quality disparity).
            "PQD": float(np.nanmin(avg_acc_per_group) / np.nanmax(avg_acc_per_group)),
            "PQD_per_class": _safe_ratio_of_min_to_max(avg_acc_array, axis=0).tolist(),
            # Source: [Fairer-AI] cal_metrics L253-258 — "DPM" (demographic parity metric, Fairer-AI code version).
            "DPM": float(np.nanmean(_safe_ratio_of_min_to_max(demo_array, axis=0))),
            "DPM_per_class": _safe_ratio_of_min_to_max(demo_array, axis=0).tolist(),
            # Source: [Fairer-AI] cal_metrics L261-266 — "EOM" (equality of opportunity metric).
            "EOM": float(np.nanmean(_safe_ratio_of_min_to_max(eo_array, axis=0))),
            "EOM_per_class": _safe_ratio_of_min_to_max(eo_array, axis=0).tolist(),
        },
        "metadata": {
            "threshold_strategy": threshold_strategy,
            "thresholds": thresholds,
            "dpm_note": "Matches Fairer-AI public code, not standard demographic parity.",
            "eom_note": "Matches Fairer-AI public code; ratio of group-wise TPR values.",
            "sample_count": n_samples,
            "num_classes": n_classes,
        },
    }


def cross_auc(positive_scores: Any, negative_scores: Any) -> float:
    pos = to_numpy(positive_scores).reshape(-1).astype(float)
    neg = to_numpy(negative_scores).reshape(-1).astype(float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)], axis=0)
    return safe_roc_auc(labels, scores)


def evaluate_pairwise_fairness(y_true: Any, y_score: Any, groups: Any) -> dict[str, Any]:
    """Evaluate pairwise fairness (Nat Commun 2023 PFD convention).

    Treats fairness as a bipartite-ranking problem: for each subgroup, compute
    the AUC of its positives against the global negative pool. PFD is the
    max-min spread of this per-subgroup statistic.

    Args:
        y_true: shape (N,) 0/1 labels.
        y_score: shape (N,) positive-class scores.
        groups: shape (N,) group labels.

    Returns `overall` (pairwise_overall, AUROC), `groups` (per-subgroup
    pairwise_within_group and pairwise_subgroup_vs_all_negative), `fairness`
    (pairwise_fairness_difference_pfd), and `metadata`.
    """
    y_true_arr = to_numpy(y_true).astype(int).reshape(-1)
    y_score_arr = to_numpy(y_score).astype(float).reshape(-1)
    group_arr = to_numpy(groups).reshape(-1)
    if y_true_arr.shape != y_score_arr.shape or y_true_arr.shape[0] != group_arr.shape[0]:
        raise ValueError("y_true, y_score, and groups must align")

    all_positive = y_score_arr[y_true_arr == 1]
    all_negative = y_score_arr[y_true_arr == 0]
    within_group: dict[Any, float] = {}
    subgroup_vs_all_neg: dict[Any, float] = {}
    group_counts: dict[Any, dict[str, int]] = {}

    for group_value in group_indices(group_arr):
        mask = group_arr == group_value
        group_pos = y_score_arr[mask & (y_true_arr == 1)]
        group_neg = y_score_arr[mask & (y_true_arr == 0)]
        within_group[group_value] = cross_auc(group_pos, group_neg)
        subgroup_vs_all_neg[group_value] = cross_auc(group_pos, all_negative)
        group_counts[group_value] = {"positive_count": int(group_pos.size), "negative_count": int(group_neg.size)}

    subgroup_pf_values = np.asarray(list(subgroup_vs_all_neg.values()), dtype=float)
    return {
        "overall": {
            # Source: standard sklearn AUROC over the full dataset.
            "auroc": safe_roc_auc(y_true_arr, y_score_arr),
            # Source: [NC2023-PFD] all-positives vs all-negatives bipartite AUC (equivalent to AUROC).
            "pairwise_overall": cross_auc(all_positive, all_negative),
        },
        "groups": {
            group_value: {
                # Source: within-group positives vs within-group negatives AUC — diagnostic.
                "pairwise_within_group": within_group[group_value],
                # Source: [NC2023-PFD] key per-subgroup fairness statistic:
                #         AUC of subgroup positives ranked against the global negative pool.
                "pairwise_subgroup_vs_all_negative": subgroup_vs_all_neg[group_value],
                **group_counts[group_value],
            }
            for group_value in group_indices(group_arr)
        },
        "fairness": {
            # Source: [NC2023-PFD] Pairwise Fairness Difference =
            #         max_g pairwise_subgroup_vs_all_negative(g) − min_g pairwise_subgroup_vs_all_negative(g).
            "pairwise_fairness_difference_pfd": float(np.nanmax(subgroup_pf_values) - np.nanmin(subgroup_pf_values))
            if subgroup_pf_values.size
            else float("nan")
        },
        "metadata": {"paper_semantics": "[NC2023-PFD] subgroup positives vs all negatives"},
    }


def dice_score(y_pred: Any, y_true: Any, eps: float = 1e-8) -> np.ndarray:
    pred = to_numpy(y_pred).astype(float)
    true = to_numpy(y_true).astype(float)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if pred.ndim < 2:
        raise ValueError("segmentation inputs must be batch-first")
    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)
    return (2.0 * np.sum(pred_flat * true_flat, axis=1) + eps) / (
        np.sum(pred_flat, axis=1) + np.sum(true_flat, axis=1) + eps
    )


def evaluate_segmentation(
    y_pred: Any | None = None,
    y_true: Any | None = None,
    groups: Any | None = None,
    dice_values: Any | None = None,
) -> dict[str, Any]:
    """Evaluate segmentation fairness using FairMedFM's DSC suite.

    Either pass pre-computed Dice scores via `dice_values`, or pass masks
    (`y_pred`, `y_true`) and let `dice_score` compute them.

    Args:
        y_pred: optional shape (N, ...) predicted masks.
        y_true: optional shape (N, ...) ground-truth masks.
        groups: optional shape (N,) group labels.
        dice_values: optional shape (N,) pre-computed per-sample Dice.

    Returns `overall.mean_dice`, `groups[group].mean_dice`, and `fairness`
    with delta_dice (DSC_Delta), std_dice (DSC_STD), skewness_dice (DSC_Skew),
    es_dice (DSC_ES), min_dice, max_dice.
    """
    dice_arr = to_numpy(dice_values).astype(float).reshape(-1) if dice_values is not None else dice_score(y_pred, y_true)
    if groups is None:
        return {"overall": {"mean_dice": float(np.mean(dice_arr))}, "groups": {}, "fairness": {}}

    group_arr = to_numpy(groups).reshape(-1)
    if group_arr.shape[0] != dice_arr.shape[0]:
        raise ValueError("groups must match the number of dice values")
    group_means = {group_value: float(np.mean(dice_arr[group_arr == group_value])) for group_value in group_indices(group_arr)}
    group_values = np.asarray(list(group_means.values()), dtype=float)
    mean_dice = float(np.mean(dice_arr))
    min_dice = float(np.min(group_values))
    max_dice = float(np.max(group_values))
    std_dice = float(np.std(group_values))
    return {
        "overall": {
            # Source: [FairMedFM] Table 2 "DSC" — sample-level mean Dice.
            "mean_dice": mean_dice,
        },
        "groups": {group_value: {"mean_dice": group_means[group_value]} for group_value in group_indices(group_arr)},
        "fairness": {
            # Source: [FairMedFM] utils/metrics.py::evaluate_seg L170 — "DSC_Δ".
            "delta_dice": float(abs(group_values[0] - group_values[1])) if group_values.size == 2 else float(max_dice - min_dice),
            # Source: [FairMedFM] evaluate_seg L171 — "DSC_Skew" (eps-clamp added for stability).
            "skewness_dice": float((1.0 - min_dice) / max(1.0 - max_dice, 1e-8)),
            # Source: [FairMedFM] evaluate_seg L172 — "DSC_STD".
            "std_dice": std_dice,
            # Source: [FairMedFM] evaluate_seg L173 — "DSC_ES" (equity-scaled DSC).
            "es_dice": float(mean_dice / (1.0 + std_dice)),
            # Min / max group means — diagnostic companions (min_dice also reported by [FairMedFM]).
            "min_dice": min_dice,
            "max_dice": max_dice,
        },
    }


def _parse_array_cell(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            return ast.literal_eval(stripped)
    return value


def _combine_group_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if len(columns) == 1:
        return df[columns[0]].to_numpy()
    return df[columns].astype(object).apply(lambda row: tuple(row.tolist()), axis=1).to_numpy()


def _stack_object_series(series: pd.Series) -> np.ndarray:
    return np.asarray([_parse_array_cell(v) for v in series.tolist()])


def inputs_from_dataframe(task: str, df: pd.DataFrame, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = schema or {}
    kwargs: dict[str, Any] = {}
    if "group_columns" in schema:
        kwargs["groups"] = _combine_group_columns(df, list(schema["group_columns"]))
    elif "group_column" in schema:
        kwargs["groups"] = df[schema["group_column"]].to_numpy()

    if task in {"binary", "pairwise"}:
        kwargs["y_true"] = df[schema["target_column"]].to_numpy()
        kwargs["y_score"] = df[schema["score_column"]].to_numpy()
        return kwargs
    if task == "multilabel":
        if "target_columns" in schema and "score_columns" in schema:
            kwargs["y_true"] = df[list(schema["target_columns"])].to_numpy()
            kwargs["y_score"] = df[list(schema["score_columns"])].to_numpy()
        elif "target_column" in schema and "score_column" in schema:
            kwargs["y_true"] = _stack_object_series(df[schema["target_column"]])
            kwargs["y_score"] = _stack_object_series(df[schema["score_column"]])
        else:
            raise ValueError("Multilabel schema must provide target/score columns")
        return kwargs
    if task == "segmentation":
        if "dice_column" in schema:
            kwargs["dice_values"] = df[schema["dice_column"]].to_numpy()
        elif "pred_column" in schema and "true_column" in schema:
            kwargs["y_pred"] = _stack_object_series(df[schema["pred_column"]])
            kwargs["y_true"] = _stack_object_series(df[schema["true_column"]])
        else:
            raise ValueError("Segmentation schema must provide dice_column or pred/true columns")
        return kwargs
    raise ValueError(f"Unknown task: {task}")


def inputs_from_csv(
    task: str,
    path: str | Path,
    schema: dict[str, Any] | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
    array_columns: list[str] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(path, **(read_csv_kwargs or {}))
    if array_columns:
        for col in array_columns:
            df[col] = df[col].apply(_parse_array_cell)
    return inputs_from_dataframe(task=task, df=df, schema=schema)


def inputs_from_npz(task: str, path: str | Path, key_map: dict[str, Any] | None = None) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    key_map = key_map or {}

    def get_value(arg_name: str, default_key: str | None = None) -> Any:
        if arg_name in key_map:
            key = key_map[arg_name]
            if isinstance(key, (list, tuple)):
                arrays = [data[k] for k in key]
                if arg_name == "groups":
                    return np.asarray(list(zip(*arrays)), dtype=object)
                return np.stack(arrays, axis=1)
            return data[key]
        if default_key is not None and default_key in data:
            return data[default_key]
        return None

    kwargs: dict[str, Any] = {}
    if task in {"binary", "pairwise", "multilabel"}:
        kwargs["y_true"] = get_value("y_true", "y_true")
        kwargs["y_score"] = get_value("y_score", "y_score")
        groups = get_value("groups", "groups")
        if groups is not None:
            kwargs["groups"] = groups
        return kwargs
    if task == "segmentation":
        dice_values = get_value("dice_values", "dice_values")
        if dice_values is not None:
            kwargs["dice_values"] = dice_values
        else:
            kwargs["y_pred"] = get_value("y_pred", "y_pred")
            kwargs["y_true"] = get_value("y_true", "y_true")
        groups = get_value("groups", "groups")
        if groups is not None:
            kwargs["groups"] = groups
        return kwargs
    raise ValueError(f"Unknown task: {task}")


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str, np.generic)) and not isinstance(value, (list, tuple))


def _scalar_table(section_name: str, mapping: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"section": section_name, "metric": key, "value": _to_serializable(value)} for key, value in mapping.items() if _is_scalar(value)]
    )


def _group_table(groups: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for group_name, metrics in groups.items():
        row = {"group": str(group_name)}
        for key, value in metrics.items():
            if _is_scalar(value):
                row[key] = _to_serializable(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_paper_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if "overall" in result and isinstance(result["overall"], dict):
        tables["overall"] = _scalar_table("overall", result["overall"])
    if "fairness" in result and isinstance(result["fairness"], dict):
        tables["fairness"] = _scalar_table("fairness", result["fairness"])
    if "metadata" in result and isinstance(result["metadata"], dict):
        tables["metadata"] = _scalar_table("metadata", result["metadata"])
    if "groups" in result and isinstance(result["groups"], dict):
        tables["groups"] = _group_table(result["groups"])
    return tables


def save_json_report(result: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_serializable(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_paper_tables(
    result: dict[str, Any],
    output_dir: str | Path,
    stem: str = "fairness_report",
    formats: tuple[str, ...] = ("csv", "md"),
) -> dict[str, list[Path]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tables = build_paper_tables(result)
    outputs: dict[str, list[Path]] = {}
    for table_name, df in tables.items():
        outputs[table_name] = []
        for fmt in formats:
            path = output_root / f"{stem}_{table_name}.{fmt}"
            if fmt == "csv":
                df.to_csv(path, index=False)
            elif fmt == "md":
                path.write_text(df.to_markdown(index=False), encoding="utf-8")
            elif fmt == "json":
                path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
            else:
                raise ValueError(f"Unsupported table format: {fmt}")
            outputs[table_name].append(path)
    return outputs


@dataclass
class FairnessEvaluator:
    """Top-level evaluator dispatching to the four task-specific routines.

    Example:
        evaluator = FairnessEvaluator("binary")
        result = evaluator.evaluate(y_true=..., y_score=..., groups=...)
        evaluator.export_json(result, "report.json")
        evaluator.export_tables(result, "report_dir/")

    `task` must be one of "binary", "multilabel", "pairwise", "segmentation".
    For CSV / DataFrame / NPZ inputs, use the `evaluate_from_*` helpers with a
    `schema` dict (see module docstring for schema keys).
    """

    task: str

    def __post_init__(self) -> None:
        if self.task not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown task: {self.task}")

    def metric_catalog(self) -> list[str]:
        return list(SUPPORTED_METRICS[self.task])

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        if self.task == "binary":
            result = evaluate_binary_classification(**kwargs)
        elif self.task == "multilabel":
            result = evaluate_multilabel_fairness(**kwargs)
        elif self.task == "pairwise":
            result = evaluate_pairwise_fairness(**kwargs)
        else:
            result = evaluate_segmentation(**kwargs)
        result["supported_metrics"] = self.metric_catalog()
        return result

    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        schema: dict[str, Any] | None = None,
        **evaluate_kwargs: Any,
    ) -> dict[str, Any]:
        inputs = inputs_from_dataframe(task=self.task, df=df, schema=schema)
        inputs.update(evaluate_kwargs)
        return self.evaluate(**inputs)

    def evaluate_from_csv(
        self,
        path: str | Path,
        schema: dict[str, Any] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        array_columns: list[str] | None = None,
        **evaluate_kwargs: Any,
    ) -> dict[str, Any]:
        inputs = inputs_from_csv(
            task=self.task,
            path=path,
            schema=schema,
            read_csv_kwargs=read_csv_kwargs,
            array_columns=array_columns,
        )
        inputs.update(evaluate_kwargs)
        return self.evaluate(**inputs)

    def evaluate_from_npz(
        self,
        path: str | Path,
        key_map: dict[str, Any] | None = None,
        **evaluate_kwargs: Any,
    ) -> dict[str, Any]:
        inputs = inputs_from_npz(task=self.task, path=path, key_map=key_map)
        inputs.update(evaluate_kwargs)
        return self.evaluate(**inputs)

    def export_json(self, result: dict[str, Any], path: str | Path) -> Path:
        return save_json_report(result=result, path=path)

    def export_tables(
        self,
        result: dict[str, Any],
        output_dir: str | Path,
        stem: str = "fairness_report",
        formats: tuple[str, ...] = ("csv", "md"),
    ) -> dict[str, list[Path]]:
        return save_paper_tables(result=result, output_dir=output_dir, stem=stem, formats=formats)


def create_evaluator(task: str) -> FairnessEvaluator:
    return FairnessEvaluator(task=task)


__all__ = [
    "FairnessEvaluator",
    "SUPPORTED_METRICS",
    "build_paper_tables",
    "create_evaluator",
    "evaluate_binary_classification",
    "evaluate_multilabel_fairness",
    "evaluate_pairwise_fairness",
    "evaluate_segmentation",
    "inputs_from_csv",
    "inputs_from_dataframe",
    "inputs_from_npz",
    "save_json_report",
    "save_paper_tables",
]
