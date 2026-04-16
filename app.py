from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_engine import (
    Constraints,
    apply_constraints,
    compute_pareto_frontier,
    compute_weighted_scores,
    summarize_rejections,
    sweep_single_weight,
)
from parsers import load_and_normalize_data
from recommendation import (
    recommend_best_accuracy,
    recommend_best_balanced,
    recommend_best_efficiency,
    recommend_best_latency,
)


APP_TITLE = "Constraint-Aware Hardware Design Decision Support Tool"
SAMPLE_DIR = Path("sample_data")


def _metric_str(v: object, unit: str = "", digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        x = float(v)
        if np.isnan(x):
            return "N/A"
        return f"{x:.{digits}f}{unit}"
    except Exception:
        s = str(v)
        return s if unit == "" else f"{s}{unit}"


def _maybe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    pareto_only: bool,
) -> Tuple[bool, Optional[px.scatter]]:
    if x not in df.columns or y not in df.columns:
        return False, None
    d = df.copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if pareto_only and "is_pareto" in d.columns:
        d = d[d["is_pareto"]]
    if d.empty:
        return False, None

    hover = [c for c in ["config_id", "design_type", "source", "accuracy", "latency_cycles", "pins", "slices"] if c in d.columns]
    fig = px.scatter(
        d,
        x=x,
        y=y,
        color="is_pareto" if "is_pareto" in d.columns else None,
        symbol="source" if "source" in d.columns else None,
        hover_data=hover,
        title=title,
        color_discrete_map={True: "#1f77b4", False: "#7f7f7f"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return True, fig


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.write(
    "A hardware-aware design-space exploration helper that turns experiment logs and RTL-derived metrics "
    "into constraint-aware feasibility filtering, Pareto analysis, and engineering recommendations."
)

st.divider()

st.subheader("A. Overview")
st.caption(
    "Decision flow: load candidates → apply hard constraints (feasible vs rejected) → Pareto-optimal set → "
    "recommendations under priorities → sensitivity to weights. The tool prefers clear engineering logic over dashboard flash."
)

st.subheader("B. Data input")
left, right = st.columns([1.2, 1.8], vertical_alignment="top")
with left:
    mode = st.radio("Input mode", ["Use sample dataset", "Upload file"], horizontal=False)
    sample_choice = None
    uploaded = None

    if mode == "Use sample dataset":
        sample_choice = st.selectbox(
            "Sample dataset",
            [
                "sample_tradeoff.csv (accuracy/latency/resources)",
                "sample_experiments.json (experiment log style)",
                "sample_rtl_results.csv (RTL latency LUT vs MAC)",
            ],
        )
    else:
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

with right:
    df: pd.DataFrame
    detected_detail = ""
    warnings: list[str] = []
    try:
        if mode == "Use sample dataset":
            if sample_choice.startswith("sample_tradeoff.csv"):
                df, detected, warnings = load_and_normalize_data(SAMPLE_DIR / "sample_tradeoff.csv")
            elif sample_choice.startswith("sample_experiments.json"):
                df, detected, warnings = load_and_normalize_data(SAMPLE_DIR / "sample_experiments.json")
            else:
                df, detected, warnings = load_and_normalize_data(SAMPLE_DIR / "sample_rtl_results.csv")
        else:
            if uploaded is None:
                st.info("Upload a CSV/JSON file to begin.")
                st.stop()
            df, detected, warnings = load_and_normalize_data(uploaded)

        detected_detail = f"{detected.kind}: {detected.detail}"
    except Exception as e:
        st.error(f"Failed to load input: {e}")
        st.stop()

    if warnings:
        with st.expander("Input warnings", expanded=False):
            for w in warnings:
                st.warning(w)

    k1, k2, k3 = st.columns(3)
    k1.metric("Candidates", len(df))
    k2.metric("Detected input", detected.kind)
    k3.metric("Schema columns", len(df.columns))
    st.caption(detected_detail)

    with st.expander("Detected schema (columns)", expanded=False):
        st.code(", ".join(df.columns), language="text")

st.divider()

st.subheader("C. Hard constraint panel")
st.caption(
    "Hard constraints are applied before any scoring/ranking. By default, a constraint requires the metric to be present; "
    "missing values are treated as infeasible for that constraint."
)

c1, c2, c3 = st.columns(3)
with c1:
    use_min_acc = st.checkbox("Enable min accuracy", value=False)
    min_acc = st.number_input("Minimum accuracy", value=90.0, step=0.1) if use_min_acc else None

    use_min_pin_red = st.checkbox("Enable min pin_reduction", value=False)
    min_pin_red = st.number_input("Minimum pin_reduction (%)", value=0.0, step=0.5) if use_min_pin_red else None

with c2:
    use_max_lat = st.checkbox("Enable max latency_cycles", value=False)
    max_lat = st.number_input("Maximum latency_cycles", value=1500.0, step=10.0) if use_max_lat else None

    use_min_slice_red = st.checkbox("Enable min slice_reduction", value=False)
    min_slice_red = st.number_input("Minimum slice_reduction (%)", value=0.0, step=0.5) if use_min_slice_red else None

with c3:
    use_max_pins = st.checkbox("Enable max pins", value=False)
    max_pins = st.number_input("Maximum pins", value=320.0, step=1.0) if use_max_pins else None

    use_max_slices = st.checkbox("Enable max slices", value=False)
    max_slices = st.number_input("Maximum slices", value=1800.0, step=10.0) if use_max_slices else None

treat_nan_as_violation = st.checkbox("Treat missing metric as constraint violation", value=True)

constraints = Constraints(
    min_accuracy=min_acc,
    max_latency_cycles=max_lat,
    max_pins=max_pins,
    max_slices=max_slices,
    min_pin_reduction=min_pin_red,
    min_slice_reduction=min_slice_red,
    treat_nan_as_violation=bool(treat_nan_as_violation),
)

df_feasible, df_rejected = apply_constraints(df, constraints)
rej_summary = summarize_rejections(df_rejected)

st.divider()

st.subheader("D. Feasibility summary")
k1, k2, k3 = st.columns(3)
k1.metric("Total candidates", len(df))
k2.metric("Feasible", len(df_feasible))
k3.metric("Rejected", len(df_rejected))

left, right = st.columns([1.25, 1.75], vertical_alignment="top")
with left:
    st.markdown("**Most common rejection reasons**")
    if rej_summary.empty:
        st.caption("No rejection reasons (either no constraints enabled, or no rejected candidates).")
    else:
        st.dataframe(rej_summary, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Feasible preview**")
    show_cols = [c for c in ["config_id", "design_type", "accuracy", "latency_cycles", "pins", "slices", "pin_reduction", "slice_reduction", "packing_efficiency", "source"] if c in df_feasible.columns]
    st.dataframe(df_feasible[show_cols].head(25), use_container_width=True, hide_index=True)

st.divider()

st.subheader("E. Pareto analysis")
pareto_group = st.selectbox(
    "Objective group",
    [
        "maximize accuracy, minimize latency_cycles, minimize slices",
        "maximize accuracy, minimize pins",
        "maximize accuracy, minimize resource usage (pins+slices)",
    ],
)
pareto_only = st.checkbox("Show Pareto candidates only", value=False)

pareto_df = df_feasible.copy()
if pareto_group == "maximize accuracy, minimize latency_cycles, minimize slices":
    objectives = ["accuracy", "latency_cycles", "slices"]
    directions = ["maximize", "minimize", "minimize"]
elif pareto_group == "maximize accuracy, minimize pins":
    objectives = ["accuracy", "pins"]
    directions = ["maximize", "minimize"]
else:
    pareto_df = pareto_df.copy()
    if "pins" in pareto_df.columns and "slices" in pareto_df.columns:
        pareto_df["resource_usage"] = pd.to_numeric(pareto_df["pins"], errors="coerce") + pd.to_numeric(pareto_df["slices"], errors="coerce")
    objectives = ["accuracy", "resource_usage"]
    directions = ["maximize", "minimize"]

pareto_df = compute_pareto_frontier(pareto_df, objectives=objectives, directions=directions, flag_column="is_pareto")

st.caption(
    "Pareto-optimal means no other feasible candidate is strictly better in all selected objectives "
    "(with at least one strict improvement)."
)

pareto_candidates = pareto_df[pareto_df.get("is_pareto", False)].copy()
st.markdown("**Pareto candidate table**")
pareto_cols = [c for c in ["config_id", "design_type", "accuracy", "latency_cycles", "pins", "slices", "pin_reduction", "slice_reduction", "source"] if c in pareto_df.columns]
if pareto_only:
    st.dataframe(pareto_candidates[pareto_cols], use_container_width=True, hide_index=True)
else:
    st.dataframe(pareto_df[pareto_cols].sort_values("is_pareto", ascending=False), use_container_width=True, hide_index=True)

plot_cols = st.columns(2)
ok, fig = _maybe_scatter(pareto_df, "accuracy", "latency_cycles", "Accuracy vs Latency (cycles)", pareto_only)
with plot_cols[0]:
    if ok:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Skipped: `accuracy` and `latency_cycles` are required for this plot.")

ok2, fig2 = _maybe_scatter(pareto_df, "accuracy", "slices", "Accuracy vs Slices", pareto_only)
with plot_cols[1]:
    if ok2:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Skipped: `accuracy` and `slices` are required for this plot.")

ok3, fig3 = _maybe_scatter(pareto_df, "pin_reduction", "accuracy", "Pin reduction vs Accuracy", pareto_only)
if ok3:
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Skipped: `pin_reduction` and `accuracy` are required for this plot.")

st.divider()

st.subheader("F. Recommendation engine (feasible set only)")

rec_cols = st.columns(4)
rec_acc, expl_acc = recommend_best_accuracy(df_feasible)
rec_eff, expl_eff = recommend_best_efficiency(df_feasible)
rec_lat, expl_lat = recommend_best_latency(df_feasible)
rec_bal, expl_bal = recommend_best_balanced(df_feasible, 1.0, 1.0, 1.0)

def _render_rec(container, title: str, rec: Optional[dict], expl: str) -> None:
    with container:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(expl)
            if rec is None:
                st.write("No recommendation available.")
                return
            st.write(f"- **config_id**: `{rec.get('config_id')}`")
            st.write(f"- **design_type**: `{rec.get('design_type')}`")
            st.write(f"- **source**: `{rec.get('source')}`")
            st.write(f"- **accuracy**: {_metric_str(rec.get('accuracy'))}")
            st.write(f"- **latency_cycles**: {_metric_str(rec.get('latency_cycles'))}")
            st.write(f"- **pins / slices**: {_metric_str(rec.get('pins'))} / {_metric_str(rec.get('slices'))}")
            st.write(f"- **pin_reduction / slice_reduction**: {_metric_str(rec.get('pin_reduction'), unit='%')} / {_metric_str(rec.get('slice_reduction'), unit='%')}")


_render_rec(rec_cols[0], "Best accuracy", rec_acc, expl_acc)
_render_rec(rec_cols[1], "Best efficiency", rec_eff, expl_eff)
_render_rec(rec_cols[2], "Best latency-aware", rec_lat, expl_lat)
_render_rec(rec_cols[3], "Best balanced (default weights)", rec_bal, expl_bal)

st.divider()

st.subheader("G. Sensitivity / weight analysis (feasible set only)")
w1, w2, w3 = st.columns(3)
with w1:
    w_acc = st.slider("Accuracy importance", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
with w2:
    w_lat = st.slider("Latency importance", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
with w3:
    w_eff = st.slider("Hardware efficiency importance", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

scored, meta = compute_weighted_scores(df_feasible, w_acc, w_lat, w_eff, score_column="weighted_score")
st.caption(meta.get("note", ""))
st.caption(meta.get("weights_used", ""))

rank_cols = [c for c in ["config_id", "design_type", "weighted_score", "accuracy", "latency_cycles", "pin_reduction", "slice_reduction", "packing_efficiency", "source"] if c in scored.columns]
st.dataframe(scored.sort_values("weighted_score", ascending=False)[rank_cols].head(30), use_container_width=True, hide_index=True)

st.markdown("**Simple sweep: varying latency weight**")
sweep_df = sweep_single_weight(
    df_feasible,
    sweep="latency",
    values=np.linspace(0.0, 5.0, 11),
    base_weights={"accuracy": w_acc, "latency": w_lat, "efficiency": w_eff},
)
if not sweep_df.empty and sweep_df["top_score"].notna().any():
    fig_sweep = px.line(
        sweep_df,
        x="sweep",
        y="top_score",
        markers=True,
        hover_data=["top_config_id"],
        title="Top recommendation score vs latency weight",
    )
    fig_sweep.update_xaxes(title="Latency weight")
    fig_sweep.update_yaxes(title="Top weighted_score")
    fig_sweep.update_layout(height=420, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_sweep, use_container_width=True)
else:
    st.info("Sweep skipped: no scorable feasible candidates under the current data/constraints.")

st.divider()

st.subheader("H. Raw data / debug")
with st.expander("Raw normalized candidates", expanded=False):
    st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Feasible candidates (with flags/scores)", expanded=False):
    st.dataframe(pareto_df, use_container_width=True, hide_index=True)

with st.expander("Rejected candidates (with reasons)", expanded=False):
    if df_rejected.empty:
        st.caption("No rejected candidates.")
    else:
        rej_cols = [c for c in ["config_id", "design_type", "source", "rejection_reason_str"] if c in df_rejected.columns]
        st.dataframe(df_rejected[rej_cols], use_container_width=True, hide_index=True)
