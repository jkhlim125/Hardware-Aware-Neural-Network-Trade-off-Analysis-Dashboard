import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_engine import (
    parse_json_to_df,
    pareto_frontier_maximize_x_y,
    pareto_frontier_maximize_x_minimize_y,
    get_top_configs,
    build_summary_text,
)

def format_config_text(row):
    pr = row["pack_ratio"]
    gs = row["global_sparsity"]
    return f"PR={pr:.2f}, GS={gs:.2f}"


def get_best_accuracy_config(df):
    d = df[~df["is_baseline"]].copy().dropna(subset=["max_acc"])
    if d.empty:
        return None
    return d.sort_values("max_acc", ascending=False).iloc[0]


def get_max_pin_config(df):
    d = df[~df["is_baseline"]].copy().dropna(subset=["pin_reduction"])
    if d.empty:
        return None
    return d.sort_values("pin_reduction", ascending=False).iloc[0]


def get_min_acc_drop_config(df):
    d = df[~df["is_baseline"]].copy().dropna(subset=["acc_drop_positive"])
    if d.empty:
        return None
    return d.sort_values("acc_drop_positive", ascending=True).iloc[0]

def apply_plot_style(fig, title, x_title, y_title, height=600):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        font=dict(size=13),
        title_font=dict(size=22),
        legend=dict(font=dict(size=12)),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=12))
    return fig


st.set_page_config(page_title="Hardware-Aware Experiment Dashboard", layout="wide")

st.title("Hardware-Aware Experiment Analysis Dashboard")
st.write(
    "This tool is for comparing experiment configurations in LUT-based neural networks "
    "from both accuracy and hardware-efficiency perspectives."
)

st.sidebar.header("Settings")

st.sidebar.markdown("### Input")
uploaded_file = st.sidebar.file_uploader("Upload parsed experiment JSON", type=["json"])
use_sample = st.sidebar.checkbox("Use sample file", value=True if uploaded_file is None else False)

st.sidebar.markdown("### Scoring")
lambda_score = st.sidebar.slider(
    "Trade-off weight (lambda)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1,
)
st.sidebar.caption("Higher lambda penalizes accuracy drop more strongly.")

json_path = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        json_path = tmp.name
elif use_sample:
    json_path = "sample_experiments.json"

if json_path is None:
    st.info("Please upload a parsed JSON file or use the sample file.")
    st.stop()

df = parse_json_to_df(json_path, lambda_score=lambda_score)

if df.empty:
    st.warning("No valid experiments were parsed.")
    st.stop()

non_baseline = df[~df["is_baseline"]].copy()
baseline_df = df[df["is_baseline"]].copy()

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", len(df))
col2.metric("Non-baseline runs", len(non_baseline))
col3.metric("Baseline runs", len(baseline_df))
col4.metric("Lambda", f"{lambda_score:.1f}")

best_tradeoff = get_top_configs(df, top_k=1)
best_acc = get_best_accuracy_config(df)
best_pin = get_max_pin_config(df)

st.markdown("### Quick Recommendations")

c1, c2, c3 = st.columns(3)

with c1:
    if not best_tradeoff.empty:
        row = best_tradeoff.iloc[0]
        st.info(
            "Best trade-off\n\n"
            f"{format_config_text(row)}\n\n"
            f"Pin Reduction: {row['pin_reduction']:.2f}%\n\n"
            f"Accuracy Drop: {row['acc_drop_positive']:.2f}%"
        )

with c2:
    if best_acc is not None:
        st.info(
            "Best accuracy retention\n\n"
            f"{format_config_text(best_acc)}\n\n"
            f"Max Accuracy: {best_acc['max_acc']:.2f}\n\n"
            f"Trade-off Score: {best_acc['tradeoff_score']:.2f}"
        )

with c3:
    if best_pin is not None:
        st.info(
            "Highest hardware gain\n\n"
            f"{format_config_text(best_pin)}\n\n"
            f"Pin Reduction: {best_pin['pin_reduction']:.2f}%\n\n"
            f"Slice Reduction: {best_pin['slice_reduction']:.2f}%"
        )

st.text(build_summary_text(df, lambda_score))

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Trade-off View",
        "Pareto View",
        "Config Explorer",
        "Epoch Curves",
        "Top Configs",
        "Raw Data",
    ]
)

with tab1:
    st.subheader("Accuracy vs Pin Reduction")
    view_mode = st.radio(
        "Select y-axis",
        ["Maximum Accuracy", "Accuracy Drop"],
        horizontal=True,
    )

    fig = go.Figure()

    if not non_baseline.empty:
        if view_mode == "Maximum Accuracy":
            y_col = "max_acc"
            y_title = "Maximum Accuracy (%)"
            frontier = pareto_frontier_maximize_x_y(non_baseline, "pin_reduction", y_col)
        else:
            y_col = "acc_drop_positive"
            y_title = "Accuracy Drop from Baseline (%)"
            frontier = pareto_frontier_maximize_x_minimize_y(non_baseline, "pin_reduction", y_col)

        fig.add_trace(
            go.Scatter(
                x=non_baseline["pin_reduction"],
                y=non_baseline[y_col],
                mode="markers",
                marker=dict(
                    size=10,
                    color=non_baseline["pack_ratio"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Pack Ratio"),
                    line=dict(width=0.5, color="black"),
                ),
                text=[
                    f"PR={r.pack_ratio}, GS={r.global_sparsity}"
                    for _, r in non_baseline.iterrows()
                ],
                name="Configs",
            )
        )

        if not frontier.empty:
            fig.add_trace(
                go.Scatter(
                    x=frontier["pin_reduction"],
                    y=frontier[y_col],
                    mode="lines+markers",
                    line=dict(color="green", width=3),
                    marker=dict(size=8),
                    name="Pareto Frontier",
                )
            )

    if not baseline_df.empty and view_mode == "Maximum Accuracy":
        fig.add_trace(
            go.Scatter(
                x=baseline_df["pin_reduction"],
                y=baseline_df["max_acc"],
                mode="markers+lines",
                marker=dict(size=12, symbol="x", color="red"),
                line=dict(color="red", width=2),
                name="Baseline",
            )
        )

    plot_title = "Maximum Accuracy vs Pin Reduction" if view_mode == "Maximum Accuracy" else "Accuracy Drop vs Pin Reduction"
    
    fig = apply_plot_style(
        fig,
        title=plot_title,
        x_title="Pin Reduction Rate (%)",
        y_title=y_title,
        height=620,
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Slice Reduction vs Accuracy Drop")

    fig2 = go.Figure()

    d2 = non_baseline.dropna(subset=["slice_reduction", "acc_drop_positive"])
    frontier2 = pareto_frontier_maximize_x_minimize_y(d2, "slice_reduction", "acc_drop_positive")

    fig2.add_trace(
        go.Scatter(
            x=d2["slice_reduction"],
            y=d2["acc_drop_positive"],
            mode="markers",
            marker=dict(
                size=10,
                color=d2["pack_ratio"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Pack Ratio"),
                line=dict(width=0.5, color="black"),
            ),
            text=[
                f"PR={r.pack_ratio}, GS={r.global_sparsity}"
                for _, r in d2.iterrows()
            ],
            name="Configs",
        )
    )

    if not frontier2.empty:
        fig2.add_trace(
            go.Scatter(
                x=frontier2["slice_reduction"],
                y=frontier2["acc_drop_positive"],
                mode="lines+markers",
                line=dict(color="green", width=3),
                marker=dict(size=8),
                name="Pareto Frontier",
            )
        )

    fig2 = apply_plot_style(
        fig2,
        title="Slice Reduction vs Accuracy Drop",
        x_title="Slice Reduction (%)",
        y_title="Accuracy Drop from Baseline (%)",
        height=620,
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Config Explorer")

    available_pack = sorted(non_baseline["pack_ratio"].dropna().unique().tolist())
    available_sparse = sorted(non_baseline["global_sparsity"].dropna().unique().tolist())

    if len(available_pack) == 0 or len(available_sparse) == 0:
        st.warning("No valid non-baseline configurations found.")
    else:
        selected_pack = st.selectbox("Pack Ratio", available_pack)
        selected_sparse = st.selectbox("Global Sparsity", available_sparse)

        selected_df = non_baseline[
            (non_baseline["pack_ratio"] == selected_pack)
            & (non_baseline["global_sparsity"] == selected_sparse)
        ].copy()

        if selected_df.empty:
            st.warning("No matching configuration.")
        else:
            row = selected_df.iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Max Accuracy", f"{row['max_acc']:.2f}" if pd.notna(row["max_acc"]) else "N/A")
            c2.metric("Pin Reduction", f"{row['pin_reduction']:.2f}%" if pd.notna(row["pin_reduction"]) else "N/A")
            c3.metric("Trade-off Score", f"{row['tradeoff_score']:.2f}" if pd.notna(row["tradeoff_score"]) else "N/A")

            detail_df = pd.DataFrame(
                {
                    "Metric": [
                        "Max Accuracy",
                        "Final Accuracy",
                        "Last5 Mean Accuracy",
                        "Final Test Loss",
                        "Pin Reduction",
                        "Slice Reduction",
                        "Pack Efficiency",
                        "Accuracy Drop",
                        "Trade-off Score",
                    ],
                    "Value": [
                        row["max_acc"],
                        row["final_acc"],
                        row["last5_mean_acc"],
                        row["final_test_loss"],
                        row["pin_reduction"],
                        row["slice_reduction"],
                        row["pack_efficiency"],
                        row["acc_drop_positive"],
                        row["tradeoff_score"],
                    ],
                }
            )

            st.dataframe(detail_df, use_container_width=True)

            if not baseline_df.empty:
            base_row = baseline_df.iloc[0]
        
            st.markdown("#### Baseline Comparison")
        
            b1, b2, b3 = st.columns(3)
        
            acc_diff = row["max_acc"] - base_row["max_acc"] if pd.notna(row["max_acc"]) and pd.notna(base_row["max_acc"]) else None
            loss_diff = row["final_test_loss"] - base_row["final_test_loss"] if pd.notna(row["final_test_loss"]) and pd.notna(base_row["final_test_loss"]) else None
        
            b1.metric(
                "Accuracy vs Baseline",
                f"{row['max_acc']:.2f}",
                f"{acc_diff:.2f}" if acc_diff is not None else None
            )
        
            b2.metric(
                "Loss vs Baseline",
                f"{row['final_test_loss']:.4f}",
                f"{loss_diff:.4f}" if loss_diff is not None else None
            )
        
            b3.metric(
                "Pin Reduction Gain",
                f"{row['pin_reduction']:.2f}%",
                None
            )
            
with tab4:
    st.subheader("Epoch Curves")

    available_pack = sorted(non_baseline["pack_ratio"].dropna().unique().tolist())
    available_sparse = sorted(non_baseline["global_sparsity"].dropna().unique().tolist())

    if len(available_pack) == 0 or len(available_sparse) == 0:
        st.warning("No valid non-baseline configurations found.")
    else:
        selected_pack = st.selectbox("Pack Ratio for curve view", available_pack, key="curve_pack")
        selected_sparse = st.selectbox("Global Sparsity for curve view", available_sparse, key="curve_sparse")

        selected_df = non_baseline[
            (non_baseline["pack_ratio"] == selected_pack)
            & (non_baseline["global_sparsity"] == selected_sparse)
        ].copy()

        if selected_df.empty:
            st.warning("No matching configuration.")
        else:
            row = selected_df.iloc[0]

            epochs = row["epochs"]
            acc_curve = row["test_acc_curve"]
            loss_curve = row["test_loss_curve"]

            if len(epochs) == 0:
                epochs = list(range(1, len(acc_curve) + 1))

            fig_acc = go.Figure()
            fig_loss = go.Figure()

            fig_acc.add_trace(
                go.Scatter(
                    x=epochs[:len(acc_curve)],
                    y=acc_curve,
                    mode="lines",
                    name=f"PR={selected_pack}, GS={selected_sparse}",
                )
            )

            fig_loss.add_trace(
                go.Scatter(
                    x=epochs[:len(loss_curve)],
                    y=loss_curve,
                    mode="lines",
                    name=f"PR={selected_pack}, GS={selected_sparse}",
                )
            )

            if not baseline_df.empty:
                base_row = baseline_df.iloc[0]

                base_epochs = base_row["epochs"]
                base_acc_curve = base_row["test_acc_curve"]
                base_loss_curve = base_row["test_loss_curve"]

                if len(base_epochs) == 0:
                    base_epochs = list(range(1, len(base_acc_curve) + 1))

                fig_acc.add_trace(
                    go.Scatter(
                        x=base_epochs[:len(base_acc_curve)],
                        y=base_acc_curve,
                        mode="lines",
                        line=dict(dash="dash"),
                        name="Baseline",
                    )
                )

                fig_loss.add_trace(
                    go.Scatter(
                        x=base_epochs[:len(base_loss_curve)],
                        y=base_loss_curve,
                        mode="lines",
                        line=dict(dash="dash"),
                        name="Baseline",
                    )
                )

            fig_acc = apply_plot_style(
                fig_acc,
                title="Test Accuracy over Epochs",
                x_title="Epoch",
                y_title="Accuracy (%)",
                height=420,
            )

            fig_loss = apply_plot_style(
                fig_loss,
                title="Test Loss over Epochs",
                x_title="Epoch",
                y_title="Loss",
                height=420,
            )

            st.plotly_chart(fig_acc, use_container_width=True)
            st.plotly_chart(fig_loss, use_container_width=True)

with tab5:
    st.subheader("Top Configurations by Trade-off Score")

    top_df = get_top_configs(df, top_k=10)
    
    if top_df.empty:
        st.warning("No valid configurations.")
    else:
        show_cols = [
            "pack_ratio",
            "global_sparsity",
            "max_acc",
            "pin_reduction",
            "slice_reduction",
            "acc_drop_positive",
            "tradeoff_score",
        ]
        st.dataframe(top_df[show_cols], use_container_width=True)
    
        bar_df = top_df.copy()
        bar_df["config_label"] = bar_df.apply(
            lambda r: f"PR={r['pack_ratio']:.2f}, GS={r['global_sparsity']:.2f}", axis=1
        )
    
        fig3 = px.bar(
            bar_df,
            x="config_label",
            y="tradeoff_score",
            hover_data=["max_acc", "pin_reduction", "slice_reduction", "acc_drop_positive"],
        )
    
        fig3 = apply_plot_style(
            fig3,
            title="Trade-off Score Ranking",
            x_title="Configuration",
            y_title="Trade-off Score",
            height=500,
        )
    
        st.plotly_chart(fig3, use_container_width=True)

with tab6:
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download summary CSV",
        data=csv_data,
        file_name="experiment_summary.csv",
        mime="text/csv",
    )
