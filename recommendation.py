from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from analysis_engine import compute_efficiency_proxy, compute_weighted_scores, ensure_numeric_columns


def _as_row_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    # Avoid numpy scalars leaking into UI formatting.
    for k, v in list(d.items()):
        if isinstance(v, (np.floating, np.integer)):
            d[k] = v.item()
    return d


def _empty(msg: str) -> Tuple[Optional[dict], str]:
    return None, msg


def recommend_best_accuracy(df_feasible: pd.DataFrame) -> Tuple[Optional[dict], str]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty or "accuracy" not in d.columns or d["accuracy"].dropna().empty:
        return _empty("No feasible candidate with available `accuracy`.")

    best = d.dropna(subset=["accuracy"]).sort_values(["accuracy"], ascending=[False]).iloc[0]
    return (
        _as_row_dict(best),
        f"Highest feasible accuracy: {best.get('accuracy'):.2f}.",
    )


def recommend_best_efficiency(df_feasible: pd.DataFrame) -> Tuple[Optional[dict], str]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        return _empty("No feasible candidates.")

    eff = compute_efficiency_proxy(d)
    if eff.dropna().empty:
        return _empty("No feasible candidate with available `pin_reduction + slice_reduction`.")

    d = d.copy()
    d["_eff_proxy"] = eff
    sort_cols = ["_eff_proxy"]
    ascending = [False]
    if "accuracy" in d.columns:
        sort_cols.append("accuracy")
        ascending.append(False)
    best = d.dropna(subset=["_eff_proxy"]).sort_values(sort_cols, ascending=ascending).iloc[0]
    return (
        _as_row_dict(best.drop(labels=["_eff_proxy"])),
        f"Highest feasible efficiency proxy (pin_reduction + slice_reduction): {best.get('_eff_proxy'):.2f}.",
    )


def recommend_best_latency(df_feasible: pd.DataFrame) -> Tuple[Optional[dict], str]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty or "latency_cycles" not in d.columns or d["latency_cycles"].dropna().empty:
        return _empty("No feasible candidate with available `latency_cycles`.")

    # Tie-break: higher accuracy, then lower slices if present.
    sort_cols = ["latency_cycles"]
    ascending = [True]
    if "accuracy" in d.columns:
        sort_cols.append("accuracy")
        ascending.append(False)
    if "slices" in d.columns:
        sort_cols.append("slices")
        ascending.append(True)

    best = d.dropna(subset=["latency_cycles"]).sort_values(sort_cols, ascending=ascending).iloc[0]
    return (
        _as_row_dict(best),
        f"Lowest feasible latency: {best.get('latency_cycles'):.2f} cycles.",
    )


def recommend_best_balanced(
    df_feasible: pd.DataFrame,
    w_accuracy: float = 1.0,
    w_latency: float = 1.0,
    w_efficiency: float = 1.0,
) -> Tuple[Optional[dict], str]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        return _empty("No feasible candidates.")

    scored, meta = compute_weighted_scores(d, w_accuracy, w_latency, w_efficiency, score_column="weighted_score")
    if "weighted_score" not in scored.columns or scored["weighted_score"].dropna().empty:
        return _empty("No feasible candidate can be scored with the currently available metrics.")

    best = scored.dropna(subset=["weighted_score"]).sort_values(["weighted_score"], ascending=[False]).iloc[0]
    expl = (
        "Highest balanced score across feasible candidates "
        f"({meta.get('weights_used', 'weights adjusted for available metrics')})."
    )
    return _as_row_dict(best), expl
