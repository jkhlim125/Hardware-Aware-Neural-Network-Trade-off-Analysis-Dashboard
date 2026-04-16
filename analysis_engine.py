from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Direction = Literal["maximize", "minimize"]


CANONICAL_NUMERIC_COLS = [
    "accuracy",
    "accuracy_drop",
    "latency_cycles",
    "pins",
    "slices",
    "pin_reduction",
    "slice_reduction",
    "packing_efficiency",
    "lut_usage",
]


@dataclass(frozen=True)
class Constraints:
    min_accuracy: Optional[float] = None
    max_latency_cycles: Optional[float] = None
    max_pins: Optional[float] = None
    max_slices: Optional[float] = None
    min_pin_reduction: Optional[float] = None
    min_slice_reduction: Optional[float] = None

    treat_nan_as_violation: bool = True


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CANONICAL_NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_constraints(
    df: pd.DataFrame,
    constraints: Constraints,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply hard feasibility constraints and produce explicit rejection reasons.

    Returns (feasible_df, rejected_df). Both include:
      - `rejection_reasons`: list[str]
      - `rejection_reason_str`: joined reasons for easy display
      - `is_feasible`: boolean
    """
    d = ensure_numeric_columns(df)
    out = d.copy()
    out["rejection_reasons"] = [[] for _ in range(len(out))]

    def add_reason(mask: pd.Series, reason: str) -> None:
        if out.empty:
            return
        mask = mask.reindex(out.index, fill_value=False).astype(bool)
        if not mask.any():
            return
        idxs = out.index[mask].tolist()
        for i in idxs:
            out.at[i, "rejection_reasons"] = list(out.at[i, "rejection_reasons"]) + [reason]

    def violates_min(col: str, thr: float, label: str, reason_label: str) -> None:
        if col not in out.columns:
            if constraints.treat_nan_as_violation:
                add_reason(pd.Series(True, index=out.index), f"missing metric: {label}")
            return
        s = pd.to_numeric(out[col], errors="coerce")
        if constraints.treat_nan_as_violation:
            add_reason(s.isna(), f"missing metric: {label}")
        add_reason(s.notna() & (s < thr), f"{label} < {reason_label} ({thr:g})")

    def violates_max(col: str, thr: float, label: str, reason_label: str) -> None:
        if col not in out.columns:
            if constraints.treat_nan_as_violation:
                add_reason(pd.Series(True, index=out.index), f"missing metric: {label}")
            return
        s = pd.to_numeric(out[col], errors="coerce")
        if constraints.treat_nan_as_violation:
            add_reason(s.isna(), f"missing metric: {label}")
        add_reason(s.notna() & (s > thr), f"{label} > {reason_label} ({thr:g})")

    if constraints.min_accuracy is not None:
        violates_min("accuracy", float(constraints.min_accuracy), "accuracy", "min_accuracy")
    if constraints.max_latency_cycles is not None:
        violates_max("latency_cycles", float(constraints.max_latency_cycles), "latency_cycles", "max_latency_cycles")
    if constraints.max_pins is not None:
        violates_max("pins", float(constraints.max_pins), "pins", "max_pins")
    if constraints.max_slices is not None:
        violates_max("slices", float(constraints.max_slices), "slices", "max_slices")
    if constraints.min_pin_reduction is not None:
        violates_min("pin_reduction", float(constraints.min_pin_reduction), "pin_reduction", "min_pin_reduction")
    if constraints.min_slice_reduction is not None:
        violates_min("slice_reduction", float(constraints.min_slice_reduction), "slice_reduction", "min_slice_reduction")

    out["is_feasible"] = out["rejection_reasons"].apply(lambda r: len(r) == 0)
    out["rejection_reason_str"] = out["rejection_reasons"].apply(lambda r: "; ".join(r))

    feasible = out[out["is_feasible"]].copy().reset_index(drop=True)
    rejected = out[~out["is_feasible"]].copy().reset_index(drop=True)
    return feasible, rejected


def summarize_rejections(rejected_df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Return a dataframe with columns: reason, count
    """
    if rejected_df is None or rejected_df.empty or "rejection_reasons" not in rejected_df.columns:
        return pd.DataFrame({"reason": [], "count": []})

    counts: Dict[str, int] = {}
    for reasons in rejected_df["rejection_reasons"].tolist():
        if not isinstance(reasons, list):
            continue
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return pd.DataFrame({"reason": [k for k, _ in items], "count": [v for _, v in items]})


def _resolve_objectives(
    df: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[Direction],
) -> Tuple[pd.DataFrame, list[str], list[Direction], pd.DataFrame]:
    out = ensure_numeric_columns(df).copy()
    if out.empty or not objectives or len(objectives) != len(directions):
        return out, [], [], pd.DataFrame(index=out.index)

    active_objectives: list[str] = []
    active_directions: list[Direction] = []
    for objective, direction in zip(objectives, directions):
        if objective in out.columns:
            active_objectives.append(objective)
            active_directions.append(direction)

    pts = out[active_objectives].apply(pd.to_numeric, errors="coerce") if active_objectives else pd.DataFrame(index=out.index)
    return out, active_objectives, active_directions, pts


def _dominates(
    row_a: pd.Series,
    row_b: pd.Series,
    objectives: Sequence[str],
    directions: Sequence[Direction],
) -> bool:
    comparable = 0
    strictly_better = False

    for objective, direction in zip(objectives, directions):
        a_val = row_a.get(objective, np.nan)
        b_val = row_b.get(objective, np.nan)

        if pd.isna(a_val) and pd.isna(b_val):
            continue
        if pd.isna(a_val) or pd.isna(b_val):
            return False

        comparable += 1
        if direction == "maximize":
            if a_val < b_val:
                return False
            if a_val > b_val:
                strictly_better = True
        else:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better = True

    return comparable > 0 and strictly_better


def compute_dominance_strength(
    df: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[Direction],
) -> pd.DataFrame:
    """
    Annotate candidates with pairwise Pareto dominance counts.

    Adds:
      - `dominated_count`: number of other rows this candidate dominates
      - `dominated_by_count`: number of other rows that dominate this candidate
    """
    out, active_objectives, active_directions, pts = _resolve_objectives(df, objectives, directions)
    out["dominated_count"] = 0
    out["dominated_by_count"] = 0

    if out.empty or not active_objectives or pts.empty:
        return out

    candidate_idxs = pts.index[pts.notna().any(axis=1)].tolist()
    for i in candidate_idxs:
        dominates_count = 0
        dominated_by_count = 0
        row_i = pts.loc[i]
        for j in candidate_idxs:
            if i == j:
                continue
            row_j = pts.loc[j]
            if _dominates(row_i, row_j, active_objectives, active_directions):
                dominates_count += 1
            if _dominates(row_j, row_i, active_objectives, active_directions):
                dominated_by_count += 1
        out.at[i, "dominated_count"] = int(dominates_count)
        out.at[i, "dominated_by_count"] = int(dominated_by_count)

    return out


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[Direction],
    flag_column: str = "is_pareto",
) -> pd.DataFrame:
    """
    Mark Pareto-optimal candidates across selected objectives.

    Pareto-optimal means: no other candidate is strictly better in all objectives
    (with at least one strict improvement) under the given directions.
    """
    out, active_objectives, active_directions, pts = _resolve_objectives(df, objectives, directions)
    out[flag_column] = False

    if out.empty or not active_objectives or pts.empty:
        return out

    comparable_rows = pts.notna().any(axis=1)
    if not comparable_rows.any():
        return out

    candidate_idxs = out.index[comparable_rows].tolist()
    for i in candidate_idxs:
        row_i = pts.loc[i]
        dominated = False
        for j in candidate_idxs:
            if i == j:
                continue
            if _dominates(pts.loc[j], row_i, active_objectives, active_directions):
                dominated = True
                break
        out.at[i, flag_column] = not dominated

    return out


def _minmax_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=s.index)
    if hi <= lo:
        out = pd.Series(np.nan, index=s.index)
        out.loc[x.notna()] = 0.0
        return out
    return (x - lo) / (hi - lo)


def compute_efficiency_proxy(df: pd.DataFrame) -> pd.Series:
    d = ensure_numeric_columns(df)
    pin_red = pd.to_numeric(d["pin_reduction"], errors="coerce") if "pin_reduction" in d.columns else pd.Series(np.nan, index=d.index)
    slice_red = pd.to_numeric(d["slice_reduction"], errors="coerce") if "slice_reduction" in d.columns else pd.Series(np.nan, index=d.index)

    proxy = pin_red.fillna(0.0) + slice_red.fillna(0.0)
    has_component = pin_red.notna() | slice_red.notna()
    proxy = proxy.where(has_component, np.nan)
    return proxy


def compute_weighted_scores(
    df_feasible: pd.DataFrame,
    w_accuracy: float,
    w_latency: float,
    w_efficiency: float,
    score_column: str = "weighted_score",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Compute a simple, explainable weighted score among feasible candidates only.

    - accuracy term: higher is better (normalized)
    - latency term: lower is better (normalized, subtracted)
    - efficiency term: higher (pin_reduction + slice_reduction) is better
    """
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        d[score_column] = np.nan
        return d, {"note": "No feasible candidates to score."}

    weights = np.array([w_accuracy, w_latency, w_efficiency], dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if weights.sum() == 0:
        weights = np.array([1.0, 1.0, 1.0])
    acc_norm = _minmax_norm(d["accuracy"]) if "accuracy" in d.columns else pd.Series(np.nan, index=d.index)
    lat_norm = _minmax_norm(d["latency_cycles"]) if "latency_cycles" in d.columns else pd.Series(np.nan, index=d.index)
    eff_norm = _minmax_norm(compute_efficiency_proxy(d))

    d["score_accuracy_norm"] = acc_norm
    d["score_latency_norm"] = lat_norm
    d["score_efficiency_norm"] = eff_norm

    terms = pd.DataFrame(
        {
            "accuracy": acc_norm,
            "latency": -lat_norm,
            "efficiency": eff_norm,
        },
        index=d.index,
    )

    raw_weights = pd.Series(
        {
            "accuracy": float(weights[0]),
            "latency": float(weights[1]),
            "efficiency": float(weights[2]),
        }
    )
    dataset_available = terms.notna().any(axis=0)
    effective_weights = raw_weights.where(dataset_available, 0.0)
    if effective_weights.sum() == 0:
        d[score_column] = np.nan
        return d, {"note": "No scorable metrics available among feasible candidates."}

    row_weight_sum = terms.notna().mul(effective_weights, axis=1).sum(axis=1)
    weighted_sum = terms.fillna(0.0).mul(effective_weights, axis=1).sum(axis=1)
    d[score_column] = weighted_sum.divide(row_weight_sum).where(row_weight_sum > 0, np.nan)

    meta = {
        "weights_used": (
            f"accuracy={effective_weights['accuracy']:.2f}, "
            f"latency={effective_weights['latency']:.2f}, "
            f"efficiency={effective_weights['efficiency']:.2f}"
        ),
        "note": "Score is computed on the feasible set using min-max normalization and ignores missing terms row by row.",
    }
    return d, meta


def sweep_single_weight(
    df_feasible: pd.DataFrame,
    sweep: Literal["accuracy", "latency", "efficiency"],
    values: Iterable[float],
    base_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Simple sensitivity helper: sweep one weight while keeping the other two fixed.
    Returns a table of (sweep_value, top_config_id, top_score).
    """
    if base_weights is None:
        base_weights = {"accuracy": 1.0, "latency": 1.0, "efficiency": 1.0}

    rows = []
    previous_top = None
    for v in values:
        w_acc = float(base_weights.get("accuracy", 1.0))
        w_lat = float(base_weights.get("latency", 1.0))
        w_eff = float(base_weights.get("efficiency", 1.0))

        if sweep == "accuracy":
            w_acc = float(v)
        elif sweep == "latency":
            w_lat = float(v)
        else:
            w_eff = float(v)

        scored, _ = compute_weighted_scores(df_feasible, w_acc, w_lat, w_eff)
        scored = scored.dropna(subset=["weighted_score"]) if "weighted_score" in scored.columns else scored
        if scored.empty:
            rows.append({"sweep": v, "top_config_id": None, "top_score": np.nan, "selection_changed": False})
            continue
        best = scored.sort_values("weighted_score", ascending=False).iloc[0]
        top_config_id = best.get("config_id")
        rows.append(
            {
                "sweep": v,
                "top_config_id": top_config_id,
                "top_score": best.get("weighted_score"),
                "selection_changed": bool(previous_top is not None and top_config_id != previous_top),
            }
        )
        previous_top = top_config_id

    return pd.DataFrame(rows)
