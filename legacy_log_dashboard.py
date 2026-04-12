import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

# -------------------------------------------------
# 페이지 설정
# -------------------------------------------------
st.set_page_config(
    page_title="Experiment Log Dashboard",
    page_icon="📊",
    layout="wide",
)

# -------------------------------------------------
# 간단한 스타일
# -------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 유틸 함수
# -------------------------------------------------
DEFAULT_CANDIDATE_ID_COLS = [
    "run_id",
    "run",
    "experiment_id",
    "exp_id",
    "id",
    "name",
]

DEFAULT_PRIORITY_METRICS = [
    "accuracy",
    "loss",
    "sparsity",
    "latency",
    "memory",
    "lut_utilization",
    "packing_efficiency",
    "resource_utilization",
]

EXAMPLE_CSV = """run_id,accuracy,loss,sparsity,latency,memory,lut_utilization,packing_efficiency
run_1,0.87,0.34,0.50,12.3,1024,0.61,0.72
run_2,0.82,0.51,0.70,13.8,1102,0.66,0.69
run_3,0.54,1.22,0.70,15.2,1188,0.71,0.48
run_4,0.89,0.29,0.40,11.8,998,0.58,0.75
run_5,0.88,0.31,0.40,11.9,1005,0.57,0.76
run_6,0.62,1.05,0.80,16.0,1210,0.74,0.42
run_7,0.86,0.35,0.50,12.6,1033,0.62,0.71
run_8,0.78,0.60,0.90,17.5,1260,0.79,0.39
"""


def infer_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in DEFAULT_CANDIDATE_ID_COLS:
        if col in df.columns:
            return col
    return None


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def robust_zscore(series: pd.Series) -> pd.Series:
    """Median Absolute Deviation 기반 robust z-score"""
    s = pd.to_numeric(series, errors="coerce")
    median = s.median()
    mad = np.median(np.abs(s - median))

    if pd.isna(mad) or mad == 0:
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    return 0.6745 * (s - median) / mad


def direction_for_metric(metric: str) -> int:
    """
    +1: 클수록 좋음
    -1: 작을수록 좋음
    """
    metric_lower = metric.lower()

    if any(k in metric_lower for k in ["acc", "efficiency", "throughput"]):
        return 1

    if any(k in metric_lower for k in ["loss", "latency", "memory", "error"]):
        return -1

    # sparsity / utilization은 상황 의존적이라 일단 +1
    return 1


def normalize_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    method: str = "minmax"
) -> pd.DataFrame:
    """
    Normalize metrics to 0-1 range
    method: "minmax" or "zscore"
    """
    result = df[metrics].copy()
    
    if method == "minmax":
        for metric in metrics:
            s = pd.to_numeric(result[metric], errors="coerce")
            min_val = s.min()
            max_val = s.max()
            if max_val > min_val:
                result[f"{metric}_norm"] = (s - min_val) / (max_val - min_val)
            else:
                result[f"{metric}_norm"] = 0.5
    else:  # zscore
        for metric in metrics:
            s = pd.to_numeric(result[metric], errors="coerce")
            mean_val = s.mean()
            std_val = s.std(ddof=0)
            if std_val > 0:
                result[f"{metric}_norm"] = (s - mean_val) / std_val
                # Clip to [-3, 3] for visualization
                result[f"{metric}_norm"] = result[f"{metric}_norm"].clip(-3, 3)
            else:
                result[f"{metric}_norm"] = 0.0
    
    return result


def detect_anomalies(
    df: pd.DataFrame,
    anomaly_metric: str,
    z_threshold: float,
) -> pd.DataFrame:
    result = df.copy()
    z = robust_zscore(result[anomaly_metric])
    result[f"{anomaly_metric}_rz"] = z
    result["is_anomaly"] = np.abs(z) >= z_threshold

    reason_list = []
    for _, row in result.iterrows():
        rz = row[f"{anomaly_metric}_rz"]
        if abs(rz) >= z_threshold:
            direction = "높음" if rz > 0 else "낮음"
            reason_list.append(f"{anomaly_metric}({direction}, z={rz:.2f})")
        else:
            reason_list.append("정상")
    result["anomaly_reason"] = reason_list
    return result


def rank_runs(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    ranked = df.copy()
    score_parts = []

    for metric in metrics:
        s = pd.to_numeric(ranked[metric], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            norm = pd.Series(np.zeros(len(s)), index=s.index)
        else:
            norm = (s - s.min()) / (s.max() - s.min())

        direction = direction_for_metric(metric)
        if direction < 0:
            norm = 1 - norm

        score_parts.append(norm)

    if score_parts:
        stacked = pd.concat(score_parts, axis=1)
        ranked["composite_score"] = stacked.mean(axis=1)
    else:
        ranked["composite_score"] = 0.0

    ranked = ranked.sort_values("composite_score", ascending=False)
    return ranked


def build_summary(
    df: pd.DataFrame,
    id_col: Optional[str],
    selected_metrics: List[str],
    anomaly_metric: str,
) -> str:
    total_runs = len(df)
    anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0

    ranked = rank_runs(df, selected_metrics)
    best_row = ranked.iloc[0] if len(ranked) > 0 else None
    worst_row = ranked.iloc[-1] if len(ranked) > 0 else None

    def run_name(row) -> str:
        if row is None:
            return "N/A"
        return str(row[id_col]) if id_col and id_col in row.index else f"index_{row.name}"

    lines = []
    lines.append(f"- 총 run 수: {total_runs}")
    lines.append(f"- 이상 탐지된 run 수: {anomaly_count}")

    if best_row is not None:
        lines.append(f"- 종합 기준 best run: {run_name(best_row)} (score={best_row['composite_score']:.3f})")
    if worst_row is not None:
        lines.append(f"- 종합 기준 worst run: {run_name(worst_row)} (score={worst_row['composite_score']:.3f})")

    if "accuracy" in df.columns:
        acc_mean = pd.to_numeric(df["accuracy"], errors="coerce").mean()
        acc_std = pd.to_numeric(df["accuracy"], errors="coerce").std(ddof=0)
        lines.append(f"- accuracy 평균/표준편차: {acc_mean:.4f} / {acc_std:.4f}")

    if "loss" in df.columns:
        loss_mean = pd.to_numeric(df["loss"], errors="coerce").mean()
        lines.append(f"- loss 평균: {loss_mean:.4f}")

    lines.append(f"- 현재 anomaly 기준 metric: {anomaly_metric}")

    if anomaly_count > 0:
        top_anomalies = df[df["is_anomaly"]].head(3)
        names = [
            str(row[id_col]) if id_col and id_col in df.columns else str(idx)
            for idx, row in top_anomalies.iterrows()
        ]
        lines.append(f"- 우선 확인 권장 run: {', '.join(names)}")

    return "\n".join(lines)


def downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# -------------------------------------------------
# 제목 with 배지
# -------------------------------------------------
st.markdown(
    '''
    <div class="main-title">
        📊 Experiment Log Anomaly Dashboard
        <span style="display: inline-block; background-color: #FF6B6B; color: white; 
                     padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.7rem; 
                     font-weight: bold; margin-left: 1rem;">v2.0</span>
    </div>
    ''',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">실험 로그를 업로드하면 run별 성능 비교, 이상 탐지, 상관관계 분석, 요약 리포트를 제공합니다.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color:#9ca3af; font-size:16px; line-height:1.6;">
    💡 <b>대시보드 설명:</b> 실험들은 종종 노이즈 많은 결과를 생성합니다. 
    이 대시보드는 이상 run을 식별하고, 성능을 비교하며, 재현성을 개선하는 데 도움을 줍니다.
    정규화된 비교, 상관관계 분석, 자동 요약을 통해 실험 결과를 빠르게 파악하세요.
    </p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 사이드바
# -------------------------------------------------
st.sidebar.header("⚙️ 설정")

# 섹션 1: 데이터 관리
st.sidebar.subheader("📥 데이터")
uploaded_file = st.sidebar.file_uploader(
    "CSV 업로드",
    type=["csv"],
    help="run별 실험 결과가 들어있는 CSV를 업로드하세요.",
)
use_example = st.sidebar.checkbox("예제 데이터 사용", value=not bool(uploaded_file))

# 구분선
st.sidebar.divider()

# 섹션 2: 이상 탐지 설정
st.sidebar.subheader("🔍 이상 탐지")
z_threshold = st.sidebar.slider(
    "이상 탐지 민감도 (z-threshold)",
    min_value=1.5,
    max_value=4.0,
    value=2.5,
    step=0.1,
)
show_only_anomalies = st.sidebar.checkbox("이상 run만 보기", value=False)

st.sidebar.markdown(
    """
    📌 **이상 탐지 로직**  
    - Run은 선택된 metric이 평균에서 z-threshold 이상 벗어나면 플래그됩니다.
    - Robust z-score (MAD 기반)를 사용하여 이상치에 강건합니다.
    - 낮은 threshold: 더 민감함 | 높은 threshold: 극단적인 경우만 탐지
    """
)

# 구분선
st.sidebar.divider()

# 섹션 3: 분석 설정
st.sidebar.subheader("📈 분석")

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
if use_example and not uploaded_file:
    df_raw = pd.read_csv(io.StringIO(EXAMPLE_CSV))
elif uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
else:
    st.info("왼쪽에서 CSV를 업로드하거나 예제 데이터를 사용하세요.")
    st.stop()

id_col = infer_id_column(df_raw)
numeric_cols = get_numeric_columns(df_raw)

if not numeric_cols:
    st.error("숫자 컬럼이 없습니다. CSV 형식을 확인하세요.")
    st.stop()

default_metrics = [m for m in DEFAULT_PRIORITY_METRICS if m in numeric_cols]
if not default_metrics:
    default_metrics = numeric_cols[: min(4, len(numeric_cols))]

selected_metrics = st.sidebar.multiselect(
    "분석 metric 선택",
    options=numeric_cols,
    default=default_metrics,
)

if not selected_metrics:
    st.warning("하나 이상의 metric을 선택하세요.")
    st.stop()

default_anomaly_metric = "accuracy" if "accuracy" in numeric_cols else numeric_cols[0]
anomaly_metric = st.sidebar.selectbox(
    "이상 탐지 기준 metric",
    options=numeric_cols,
    index=numeric_cols.index(default_anomaly_metric),
)

df = detect_anomalies(df_raw, anomaly_metric=anomaly_metric, z_threshold=z_threshold)
ranked_df = rank_runs(df, selected_metrics)

display_df = df[df["is_anomaly"]] if show_only_anomalies else df

# -------------------------------------------------
# KPI 카드
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("총 run 수", len(df))
k2.metric("이상 run 수", int(df["is_anomaly"].sum()))
k3.metric("선택 metric 수", len(selected_metrics))
k4.metric("ID 컬럼", id_col if id_col else "없음")

# -------------------------------------------------
# Best / Worst 카드
# -------------------------------------------------
if "composite_score" in ranked_df.columns and len(ranked_df) > 0:
    best_row = ranked_df.iloc[0]
    worst_row = ranked_df.iloc[-1]

    def get_run_name(row):
        return str(row[id_col]) if id_col and id_col in row.index else f"index_{row.name}"

    c1, c2 = st.columns(2)
    with c1:
        best_metrics = ", ".join([f"{m}={best_row[m]:.3f}" for m in selected_metrics[:2]])
        st.success(
            f"🏆 Best Run: **{get_run_name(best_row)}**\n\nscore={best_row['composite_score']:.3f}\n\n{best_metrics}"
        )
    with c2:
        worst_metrics = ", ".join([f"{m}={worst_row[m]:.3f}" for m in selected_metrics[:2]])
        st.error(
            f"⚠️ Worst Run: **{get_run_name(worst_row)}**\n\nscore={worst_row['composite_score']:.3f}\n\n{worst_metrics}"
        )

# -------------------------------------------------
# 자동 요약 - Info 카드
# -------------------------------------------------
st.markdown("### 📋 자동 요약")
summary_text = build_summary(df, id_col, selected_metrics, anomaly_metric)
with st.container(border=True):
    st.markdown("#### 실험 분석 요약")
    st.markdown(summary_text)

# -------------------------------------------------
# 탭
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["개요", "시각화", "이상 run 분석", "run 비교", "결과 다운로드"]
)

# -------------------------------------------------
# 개요
# -------------------------------------------------
with tab1:
    st.markdown("#### 📋 데이터 개요")
    
    st.subheader("원본 데이터 미리보기")
    with st.expander("💡 데이터 설명", expanded=False):
        st.markdown(f"""
        - **총 run 수:** {len(df)}
        - **선택된 metrics:** {', '.join(selected_metrics)}
        - **ID 컬럼:** {id_col if id_col else '없음'}
        - **이상 run:** {int(df['is_anomaly'].sum())} ({int(df['is_anomaly'].sum())/len(df)*100:.1f}%)
        """)
    st.dataframe(display_df.head(20), use_container_width=True)

    st.subheader("📊 기본 통계")
    st.markdown("선택된 metrics의 기본 통계량입니다. (min, 25%, median, 75%, max, mean, std)")
    st.dataframe(df[selected_metrics].describe().T, use_container_width=True)

    st.subheader("🏆 종합 랭킹")
    st.markdown("**Composite Score**: 선택된 모든 metrics를 정규화하여 종합 점수로 변환한 값입니다.")
    ranking_cols = [id_col] if id_col else []
    ranking_cols += selected_metrics + ["composite_score", "is_anomaly", "anomaly_reason"]
    ranking_cols = [c for c in ranking_cols if c in ranked_df.columns]
    st.dataframe(ranked_df[ranking_cols].head(20), use_container_width=True)

# -------------------------------------------------
# 시각화
# -------------------------------------------------
with tab2:
    st.markdown("#### 📈 시각화 및 분석")
    
    st.subheader("Scatter Plot")
    st.markdown("두 개의 metrics 간 관계를 시각화합니다. 빨강(anomaly), 파랑(정상)")

    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        plot_metric_x = st.selectbox("X축", options=selected_metrics, index=0)
    with scatter_col2:
        plot_metric_y = st.selectbox(
            "Y축",
            options=selected_metrics,
            index=min(1, len(selected_metrics) - 1)
        )

    hover_cols = [id_col] if id_col else []
    hover_cols += ["anomaly_reason"]
    
    scatter_fig = px.scatter(
        display_df,
        x=plot_metric_x,
        y=plot_metric_y,
        color="is_anomaly",
        hover_data=hover_cols,
        title=f"{plot_metric_x} vs {plot_metric_y}",
        color_discrete_map={True: "#FF6B6B", False: "#4A90E2"},
    )
    scatter_fig.update_traces(marker=dict(size=8))
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("단일 metric 분포")
    dist_metric = st.selectbox("분포 확인 metric", options=selected_metrics, key="dist_metric")
    hist_fig = px.histogram(
        display_df,
        x=dist_metric,
        color="is_anomaly",
        nbins=20,
        title=f"{dist_metric} distribution",
        color_discrete_map={True: "#FF6B6B", False: "#4A90E2"},
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    if id_col:
        st.subheader("Run별 metric 비교")
        bar_metric = st.selectbox("막대그래프 metric", options=selected_metrics, key="bar_metric")
        sorted_df = display_df.sort_values(bar_metric, ascending=False)
        bar_fig = px.bar(
            sorted_df,
            x=id_col,
            y=bar_metric,
            color="is_anomaly",
            title=f"{bar_metric} by run",
            color_discrete_map={True: "#FF6B6B", False: "#4A90E2"},
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = [c for c in selected_metrics if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr().round(2)
        heatmap_fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.values,
            colorscale="Blues",
            showscale=True,
        )
        heatmap_fig.update_layout(height=600)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # 상관관계 인사이트
        st.markdown("#### 💡 상관관계 인사이트")
        
        # 상위 상관관계 찾기
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'metric1': corr.columns[i],
                    'metric2': corr.columns[j],
                    'corr': corr.iloc[i, j]
                })
        
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['corr']), reverse=True)
        
        if corr_pairs:
            insight_text = ""
            for pair in corr_pairs[:3]:
                direction = "양의 상관" if pair['corr'] > 0 else "음의 상관"
                insight_text += f"- **{pair['metric1']}** ↔ **{pair['metric2']}**: {direction} ({pair['corr']:.2f})\n"
            
            st.markdown(insight_text)
            st.markdown("※ 강한 상관관계는 한 변수가 다른 변수에 영향을 미칠 수 있음을 시사합니다.")
    else:
        st.info("Heatmap을 보려면 최소 2개 이상의 metric이 필요합니다.")

# -------------------------------------------------
# 이상 run 분석
# -------------------------------------------------
with tab3:
    st.markdown("#### 🔍 이상 Run 분석")
    st.markdown("""
    이상 run은 선택된 metric에서 평균으로부터 통계적으로 유의한 편차를 보이는 run입니다.
    이러한 run들을 분석하여 실패 원인을 파악하고 실험 프로세스를 개선할 수 있습니다.
    """)
    
    st.subheader("이상 run 목록")
    anomalies = df[df["is_anomaly"]].copy()

    if anomalies.empty:
        st.success("✅ 현재 기준으로 탐지된 이상 run이 없습니다.")
    else:
        st.warning(f"⚠️ 총 {len(anomalies)}개의 이상 run이 탐지되었습니다.")
        show_cols = [id_col] if id_col else []
        show_cols += selected_metrics + ["anomaly_reason"]
        show_cols = [c for c in show_cols if c in anomalies.columns]
        st.dataframe(anomalies[show_cols], use_container_width=True)

        st.subheader("이상 run 상세")
        if id_col:
            anomaly_run_options = anomalies[id_col].astype(str).tolist()
            selected_anomaly_run = st.selectbox("상세 확인할 이상 run", options=anomaly_run_options)

            target = df[df[id_col].astype(str) == selected_anomaly_run].iloc[0]
            baseline = df[selected_metrics].mean()

            detail_df = pd.DataFrame({
                "metric": selected_metrics,
                "run_value": [target[m] for m in selected_metrics],
                "mean_value": [baseline[m] for m in selected_metrics],
            })

            fig_detail = go.Figure()
            fig_detail.add_trace(go.Bar(
                x=detail_df["metric"],
                y=detail_df["run_value"],
                name="selected run",
                marker_color="#FF6B6B"
            ))
            fig_detail.add_trace(go.Bar(
                x=detail_df["metric"],
                y=detail_df["mean_value"],
                name="mean",
                marker_color="#4A90E2"
            ))
            fig_detail.update_layout(
                barmode="group",
                title=f"Run {selected_anomaly_run} vs Mean"
            )
            st.plotly_chart(fig_detail, use_container_width=True)

            st.markdown("#### 📊 자동 해석")
            with st.container(border=True):
                st.write(f"**이상 사유:** {target['anomaly_reason']}")
                st.write("**Metric별 편차:**")
                for metric in selected_metrics:
                    value = pd.to_numeric(target[metric], errors="coerce")
                    mean_v = pd.to_numeric(df[metric], errors="coerce").mean()
                    std_v = pd.to_numeric(df[metric], errors="coerce").std(ddof=0)
                    if std_v and not np.isnan(std_v):
                        diff_std = (value - mean_v) / std_v
                        direction = "▲" if diff_std > 0 else "▼"
                        st.write(f"- {metric}: {direction} 평균 대비 {abs(diff_std):.2f}σ 차이")

# -------------------------------------------------
# run 비교
# -------------------------------------------------
with tab4:
    st.subheader("선택한 run vs 평균 비교")

    if id_col:
        run_options = df[id_col].astype(str).tolist()
        selected_run = st.selectbox("비교할 run 선택", options=run_options)

        target = df[df[id_col].astype(str) == selected_run].iloc[0]
        baseline = df[selected_metrics].mean()

        compare_df = pd.DataFrame({
            "metric": selected_metrics,
            "selected_run": [target[m] for m in selected_metrics],
            "mean": [baseline[m] for m in selected_metrics]
        })

        fig_compare = px.bar(
            compare_df,
            x="metric",
            y=["selected_run", "mean"],
            barmode="group",
            title=f"{selected_run} vs Mean (원본값)",
            color_discrete_map={"selected_run": "#FF6B6B", "mean": "#4A90E2"}
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # 정규화 버전
        st.subheader("정규화된 비교 (0-1 범위)")
        
        norm_method = st.radio(
            "정규화 방식 선택",
            options=["Min-Max (0-1)", "Z-Score"],
            horizontal=True
        )
        norm_type = "minmax" if norm_method == "Min-Max (0-1)" else "zscore"
        
        # 정규화
        norm_df = normalize_metrics(df[selected_metrics].reset_index(drop=True), selected_metrics, method=norm_type)
        target_idx = df[df[id_col].astype(str) == selected_run].index[0]
        
        # 선택한 run과 평균 값을 정규화
        target_normalized = {}
        baseline_normalized = {}
        
        for i, metric in enumerate(selected_metrics):
            target_norm_value = norm_df.loc[target_idx, f"{metric}_norm"]
            baseline_norm_value = norm_df[f"{metric}_norm"].mean()
            target_normalized[metric] = target_norm_value
            baseline_normalized[metric] = baseline_norm_value
        
        compare_norm_df = pd.DataFrame({
            "metric": selected_metrics,
            "selected_run": [target_normalized[m] for m in selected_metrics],
            "mean": [baseline_normalized[m] for m in selected_metrics]
        })
        
        fig_compare_norm = px.bar(
            compare_norm_df,
            x="metric",
            y=["selected_run", "mean"],
            barmode="group",
            title=f"{selected_run} vs Mean (정규화됨)",
            color_discrete_map={"selected_run": "#FF6B6B", "mean": "#4A90E2"}
        )
        st.plotly_chart(fig_compare_norm, use_container_width=True)

        st.markdown("**선택 run 정보**")
        if "is_anomaly" in target.index:
            status = "⚠️ 이상 run" if bool(target["is_anomaly"]) else "✅ 정상 run"
            st.write(f"- 상태: {status}")
        if "anomaly_reason" in target.index:
            st.write(f"- anomaly reason: {target['anomaly_reason']}")

        if "composite_score" in ranked_df.columns:
            target_score = ranked_df[ranked_df[id_col].astype(str) == selected_run]["composite_score"].values
            if len(target_score) > 0:
                st.write(f"- composite score: {target_score[0]:.3f}")
    else:
        st.info("run 비교를 위해서는 ID 컬럼이 필요합니다.")

# -------------------------------------------------
# 다운로드
# -------------------------------------------------
with tab5:
    st.subheader("📥 결과 다운로드")
    
    st.markdown("""
    분석 결과를 CSV 형식으로 다운로드하여 추가 분석에 활용할 수 있습니다.
    - **전체 분석 결과**: 모든 metrics, anomaly 정보, 종합 점수 포함
    - **이상 run만**: 탐지된 이상 run만 별도로 다운로드
    - **랭킹**: 성능 기준 정렬된 run 목록
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 전체 분석 결과",
            data=downloadable_csv(df),
            file_name="analysis_results_full.csv",
            mime="text/csv",
            key="download_full"
        )
    
    with col2:
        anomalies = df[df["is_anomaly"]].copy()
        if not anomalies.empty:
            st.download_button(
                label="⚠️ 이상 run만",
                data=downloadable_csv(anomalies),
                file_name="analysis_results_anomalies.csv",
                mime="text/csv",
                key="download_anomalies"
            )
        else:
            st.info("이상 run이 없습니다.")
    
    with col3:
        st.download_button(
            label="🏆 성능 랭킹",
            data=downloadable_csv(ranked_df),
            file_name="analysis_results_ranking.csv",
            mime="text/csv",
            key="download_ranking"
        )
