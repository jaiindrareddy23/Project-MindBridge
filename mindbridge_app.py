"""
MindBridge — Module 6: Full Unified Dashboard
==============================================
The complete, production-ready Streamlit application that ties together
all 5 previous modules into one polished interface.

Pages:
  🏠 Home          → Project overview, key stats, system status
  📊 Risk Overview → Campus-wide risk heatmap and distributions
  🎯 Student Profiles → Individual student deep-dive with SHAP explanation
  📈 Trajectories  → LSTM mood trajectory visualizations
  💚 Student Chat  → Student wellness check-in chatbot (Module 5)
  🏥 Counselor     → Counselor triage assistant (Module 5)
  📋 Reports       → Downloadable counselor reports

Run with:
  streamlit run app.py
"""

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import requests # type: ignore
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBridge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1b2d 0%, #1a2e4a 100%);
}
[data-testid="stSidebar"] * { color: #c8d8ea !important; }

.metric-card {
    background: linear-gradient(135deg, #0f1b2d, #1a2e4a);
    border-radius: 14px; padding: 20px; text-align: center; color: white;
    border: 1px solid #2a4a6a;
}
.metric-card h2 { color: #64b5f6 !important; font-size: 2rem; margin: 0; }
.metric-card p  { color: #90caf9; margin: 4px 0 0; font-size: 0.85rem; }

.risk-high   { background:#fdecea; border-left:4px solid #e74c3c; padding:12px 16px; border-radius:8px; margin:6px 0; }
.risk-medium { background:#fff8e1; border-left:4px solid #f39c12; padding:12px 16px; border-radius:8px; margin:6px 0; }
.risk-low    { background:#e8f5e9; border-left:4px solid #2ecc71; padding:12px 16px; border-radius:8px; margin:6px 0; }

.chat-user { background:#1a2e4a; color:white; padding:12px 16px; border-radius:18px 18px 4px 18px; margin:8px 0 8px 60px; font-size:0.95rem; }
.chat-ai   { background:#f0f4f8; border:1px solid #c8d8ea; color:#1a2e3a; padding:12px 16px; border-radius:18px 18px 18px 4px; margin:8px 60px 8px 0; font-size:0.95rem; line-height:1.6; }

.shap-bar { height:20px; background:linear-gradient(90deg,#1a2e4a,#64b5f6); border-radius:4px; }
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_predictions():
    p = "data/predictions.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_features():
    p = "data/features.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_shap():
    p = "data/shap_values.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_students():
    p = "data/students.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_lstm_predictions():
    p = "data/lstm_predictions.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def check_model_exists():
    return os.path.exists("models/xgboost_model.pkl")

def check_lstm_exists():
    return os.path.exists("models/lstm_model.pt")


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE API
# ══════════════════════════════════════════════════════════════════════════════

def call_claude(system_prompt, messages):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Set ANTHROPIC_API_KEY environment variable to enable AI features."
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json",
                     "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 800,
                  "system": system_prompt, "messages": messages},
            timeout=30
        )
        data = response.json()
        return data["content"][0]["text"] if "content" in data else "No response."
    except Exception as e:
        return f"Connection error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 MindBridge")
    st.markdown("*AI Mental Health Early Warning System*")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 Risk Overview",
        "🎯 Student Profiles",
        "📈 Trajectories",
        "💚 Student Check-In",
        "🏥 Counselor Assistant",
        "📋 Reports",
    ])

    st.markdown("---")

    # System status
    st.markdown("**System Status**")
    data_ok  = os.path.exists("data/features.csv")
    model_ok = check_model_exists()
    lstm_ok  = check_lstm_exists()
    api_ok   = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    st.markdown(f"{'✅' if data_ok  else '❌'} Data Pipeline")
    st.markdown(f"{'✅' if model_ok else '❌'} XGBoost Model")
    st.markdown(f"{'✅' if lstm_ok  else '❌'} LSTM Model")
    st.markdown(f"{'✅' if api_ok   else '❌'} Claude API")

    st.markdown("---")
    st.markdown("**Disclaimer**")
    st.caption(
        "MindBridge is a decision-support tool. "
        "All predictions must be reviewed by qualified "
        "mental health professionals."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("🧠 MindBridge")
    st.markdown("### AI-Powered Student Mental Health Early Warning System")
    st.markdown(
        "MindBridge detects early signs of mental health deterioration in college "
        "students by analyzing behavioral patterns — sleep, academic engagement, "
        "social activity — and alerts counselors **before** crisis occurs."
    )
    st.markdown("---")

    # Load data for stats
    predictions_df = load_predictions()
    features_df    = load_features()
    students_df    = load_students()

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    total_students  = students_df["student_id"].nunique() if not students_df.empty else 0
    high_risk_count = 0
    med_risk_count  = 0
    avg_drift       = 0.0

    if not predictions_df.empty:
        high_risk_count = predictions_df[predictions_df["predicted_risk"]==2]["student_id"].nunique()
        med_risk_count  = predictions_df[predictions_df["predicted_risk"]==1]["student_id"].nunique()
    if not features_df.empty:
        avg_drift = features_df["drift_score"].mean()

    for col, val, label in [
        (c1, total_students,  "Students Monitored"),
        (c2, high_risk_count, "High Risk Flagged"),
        (c3, med_risk_count,  "Medium Risk Flagged"),
        (c4, f"{avg_drift:.1f}", "Avg Drift Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{val}</h2><p>{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### How It Works")
        steps = [
            ("📥", "Data Collection", "Behavioral signals collected passively — sleep patterns, LMS logins, assignment timing, social engagement"),
            ("⚙️", "Feature Engineering", "45+ features computed including behavioral drift, rolling trends, and mood trajectories"),
            ("🤖", "ML Risk Classification", "XGBoost classifier predicts Low / Medium / High risk with SHAP explainability"),
            ("📡", "LSTM Forecasting",  "Time-series model predicts next week's risk from the past 4-week trajectory"),
            ("💬", "AI Counselor Chat", "Claude-powered chatbot for students and counselors with privacy-first design"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin:10px 0;padding:12px;
                        background:#f0f4f8;border-radius:10px">
                <span style="font-size:1.5rem">{icon}</span>
                <div><b>{title}</b><br><small style="color:#555">{desc}</small></div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("### The Problem We Solve")
        st.markdown("""
        <div style="background:#fdecea;border-radius:12px;padding:20px;margin-bottom:12px">
            <b>❌ Without MindBridge</b><br><br>
            Students silently deteriorate for weeks.<br>
            Counselors only see them after crisis hits.<br>
            Reactive care is expensive and often too late.
        </div>
        <div style="background:#e8f5e9;border-radius:12px;padding:20px">
            <b>✅ With MindBridge</b><br><br>
            Behavioral drift detected 1–2 weeks early.<br>
            Counselors get prioritized outreach lists daily.<br>
            Proactive support replaces crisis intervention.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Key Stats")
        st.markdown("""
        - 📊 **1 in 3** college students faces a mental health crisis
        - ⏱️ Average delay to treatment: **11 years** globally
        - 🏫 Average counselor ratio: **1 per 1,500** students
        - 💰 Cost per student dropout: **$40,000–$60,000**
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Risk Overview":
    st.title("📊 Campus Risk Overview")
    st.markdown("*Population-level risk distribution and trends.*")
    st.markdown("---")

    predictions_df = load_predictions()
    features_df    = load_features()
    students_df    = load_students()

    if predictions_df.empty:
        st.warning("No prediction data found. Run Modules 1–3 first.")
    else:
        # Top KPIs
        total   = predictions_df["student_id"].nunique()
        high    = predictions_df[predictions_df["predicted_risk"]==2]["student_id"].nunique()
        medium  = predictions_df[predictions_df["predicted_risk"]==1]["student_id"].nunique()
        low     = predictions_df[predictions_df["predicted_risk"]==0]["student_id"].nunique()
        acc     = predictions_df["correct"].mean() * 100 if "correct" in predictions_df.columns else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, val, label, color in [
            (k1, total,            "Total Students",    "#64b5f6"),
            (k2, high,             "High Risk",         "#e74c3c"),
            (k3, medium,           "Medium Risk",       "#f39c12"),
            (k4, low,              "Low Risk",          "#2ecc71"),
            (k5, f"{acc:.1f}%",    "Model Accuracy",    "#9c27b0"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color:{color} !important">{val}</h2>
                    <p>{label}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        # Donut chart
        with col_left:
            risk_counts = predictions_df.groupby("predicted_name")["student_id"].nunique()
            fig_donut = go.Figure(go.Pie(
                labels=list(risk_counts.index),
                values=list(risk_counts.values),
                hole=0.55,
                marker_colors=["#e74c3c","#2ecc71","#f39c12"],
            ))
            fig_donut.update_layout(
                title="Risk Distribution",
                title_font=dict(size=16, family="DM Serif Display"),
                paper_bgcolor="rgba(0,0,0,0)",
                height=340, margin=dict(t=50,b=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Risk by week trend
        with col_right:
            if "week" in predictions_df.columns:
                weekly_risk = (
                    predictions_df.groupby(["week","predicted_name"])
                    ["student_id"].nunique().reset_index()
                )
                fig_trend = px.line(
                    weekly_risk, x="week", y="student_id",
                    color="predicted_name",
                    color_discrete_map={
                        "High": "#e74c3c",
                        "Medium": "#f39c12",
                        "Low": "#2ecc71"
                    },
                    markers=True,
                    labels={"student_id": "Students", "week": "Week"},
                    title="Risk Levels by Week"
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=340, margin=dict(t=50,b=20),
                    legend_title="Risk Level"
                )
                fig_trend.update_yaxes(gridcolor="#eee")
                st.plotly_chart(fig_trend, use_container_width=True)

        # Drift score distribution
        if not features_df.empty:
            st.markdown("### Behavioral Drift Score Distribution")
            traj_colors = {
                "healthy":   "#2ecc71",
                "declining": "#f39c12",
                "crisis":    "#e74c3c"
            }
            fig_hist = go.Figure()
            for traj, color in traj_colors.items():
                data = features_df[features_df["trajectory"]==traj]["drift_score"]
                if not data.empty:
                    fig_hist.add_trace(go.Histogram(
                        x=data, name=traj.capitalize(),
                        marker_color=color, opacity=0.7,
                        nbinsx=30,
                    ))
            fig_hist.update_layout(
                barmode="overlay",
                title="Drift Score by Student Trajectory",
                title_font=dict(size=16, family="DM Serif Display"),
                xaxis_title="Drift Score (0-100)",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340, margin=dict(t=50,b=20),
            )
            fig_hist.update_xaxes(gridcolor="#eee")
            fig_hist.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Major breakdown
        if not students_df.empty and not predictions_df.empty:
            st.markdown("### Risk by Major")
            merged = predictions_df.merge(
                students_df[["student_id","major"]], on="student_id", how="left"
            )
            major_risk = (
                merged[merged["predicted_risk"]==2]
                .groupby("major")["student_id"].nunique()
                .reset_index()
                .sort_values("student_id", ascending=True)
            )
            major_risk.columns = ["Major", "High Risk Students"]
            fig_major = go.Figure(go.Bar(
                x=major_risk["High Risk Students"],
                y=major_risk["Major"],
                orientation="h",
                marker_color="#e74c3c",
                text=major_risk["High Risk Students"],
                textposition="outside",
            ))
            fig_major.update_layout(
                title="High-Risk Students by Major",
                title_font=dict(size=16, family="DM Serif Display"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(t=50,l=160,b=20),
            )
            st.plotly_chart(fig_major, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STUDENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Student Profiles":
    st.title("🎯 Student Risk Profiles")
    st.markdown("*Individual student deep-dive with SHAP explainability.*")
    st.markdown("---")

    features_df    = load_features()
    predictions_df = load_predictions()
    shap_df        = load_shap()
    students_df    = load_students()

    if features_df.empty:
        st.warning("No data found. Run Modules 1–3 first.")
    else:
        col_filter, col_main = st.columns([1, 3])

        with col_filter:
            st.markdown("### Filters")
            risk_filter = st.multiselect(
                "Risk Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium"]
            )
            risk_map = {"High": 2, "Medium": 1, "Low": 0}
            risk_vals = [risk_map[r] for r in risk_filter]

            filtered = predictions_df[
                predictions_df["predicted_risk"].isin(risk_vals)
            ] if not predictions_df.empty else pd.DataFrame()

            student_list = (
                filtered["student_id"].unique().tolist()
                if not filtered.empty else
                features_df["student_id"].unique().tolist()
            )

            selected_id = st.selectbox("Select Student", student_list[:50])

        with col_main:
            if selected_id:
                # Student metadata
                student_meta = students_df[
                    students_df["student_id"] == selected_id
                ].iloc[0] if not students_df.empty else None

                if student_meta is not None:
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Major",    student_meta.get("major", "N/A"))
                    with m2: st.metric("Year",     student_meta.get("year",  "N/A"))
                    with m3: st.metric("GPA (Start)", student_meta.get("gpa_start", "N/A"))
                    with m4: st.metric("Trajectory", student_meta.get("trajectory","N/A").capitalize())

                # Risk prediction
                student_pred = predictions_df[
                    predictions_df["student_id"] == selected_id
                ] if not predictions_df.empty else pd.DataFrame()

                if not student_pred.empty:
                    latest_pred = student_pred.sort_values("week").iloc[-1]
                    risk_name   = latest_pred["predicted_name"]
                    prob_high   = latest_pred.get("prob_high", 0)
                    prob_med    = latest_pred.get("prob_medium", 0)
                    prob_low    = latest_pred.get("prob_low", 0)

                    color_map = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
                    color = color_map.get(risk_name, "#888")

                    st.markdown(f"""
                    <div style="background:{color}22;border:2px solid {color};
                                border-radius:12px;padding:16px;margin:12px 0">
                        <b style="font-size:1.1rem;color:{color}">
                            {risk_name} Risk
                        </b> — Latest prediction
                        <br><small>
                            Confidence: High {prob_high*100:.0f}% |
                            Medium {prob_med*100:.0f}% |
                            Low {prob_low*100:.0f}%
                        </small>
                    </div>""", unsafe_allow_html=True)

                # Weekly behavioral signals
                student_features = features_df[
                    features_df["student_id"] == selected_id
                ].sort_values("week")

                if not student_features.empty:
                    st.markdown("#### Weekly Behavioral Signals")
                    sig_cols = st.columns(2)

                    signals = [
                        ("sleep_hours_mean",   "Sleep Hours",     "#3498db"),
                        ("social_score_mean",  "Social Score",    "#2ecc71"),
                        ("lms_logins_mean",    "LMS Logins/day",  "#9b59b6"),
                        ("drift_score",        "Drift Score",     "#e74c3c"),
                    ]

                    for i, (col_name, label, color) in enumerate(signals):
                        if col_name in student_features.columns:
                            with sig_cols[i % 2]:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=student_features["week"],
                                    y=student_features[col_name],
                                    mode="lines+markers",
                                    line=dict(color=color, width=2.5),
                                    marker=dict(size=6),
                                    name=label,
                                    fill="tozeroy",
                                    fillcolor=f"{color}22",
                                ))
                                fig.update_layout(
                                    title=label,
                                    title_font=dict(size=13),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    height=220,
                                    margin=dict(t=35,b=20,l=20,r=20),
                                    showlegend=False,
                                )
                                fig.update_xaxes(
                                    title="Week", gridcolor="#eee"
                                )
                                fig.update_yaxes(gridcolor="#eee")
                                st.plotly_chart(fig, use_container_width=True)

                # SHAP explanation
                if not shap_df.empty:
                    st.markdown("#### 🔍 Why Was This Student Flagged? (SHAP)")
                    st.caption(
                        "These features contributed most to this student's "
                        "High Risk prediction."
                    )
                    top10 = shap_df.head(10)
                    fig_shap = go.Figure(go.Bar(
                        x=top10["importance"][::-1],
                        y=top10["feature"][::-1],
                        orientation="h",
                        marker_color=[
                            f"rgba(231,76,60,{0.4 + 0.6*v/top10['importance'].max()})"
                            for v in top10["importance"][::-1]
                        ],
                        text=[f"{v:.4f}" for v in top10["importance"][::-1]],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        title="Top 10 Risk Factors",
                        xaxis_title="SHAP Importance",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=340,
                        margin=dict(t=50,l=200,b=20,r=60),
                    )
                    fig_shap.update_xaxes(gridcolor="#eee")
                    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Trajectories":
    st.title("📈 Student Mood Trajectories")
    st.markdown("*LSTM-powered trajectory forecasting — where is each student heading?*")
    st.markdown("---")

    features_df    = load_features()
    lstm_preds_df  = load_lstm_predictions()

    if features_df.empty:
        st.warning("No data found. Run all modules first.")
    else:
        # Trajectory comparison
        st.markdown("### Trajectory Comparison by Risk Group")
        traj_colors = {
            "healthy":   "#2ecc71",
            "declining": "#f39c12",
            "crisis":    "#e74c3c"
        }

        fig_comp = go.Figure()
        for traj, color in traj_colors.items():
            group = features_df[features_df["trajectory"]==traj]
            if group.empty: continue
            weekly_avg = group.groupby("week")["mood_score"].mean().reset_index()
            fig_comp.add_trace(go.Scatter(
                x=weekly_avg["week"],
                y=weekly_avg["mood_score"],
                name=traj.capitalize(),
                line=dict(color=color, width=3),
                mode="lines+markers",
                marker=dict(size=7),
            ))

        fig_comp.add_hline(y=5, line_dash="dash",
                           line_color="#888", opacity=0.5,
                           annotation_text="Concern threshold (5.0)")
        fig_comp.update_layout(
            title="Average Mood Score by Trajectory Group",
            title_font=dict(size=17, family="DM Serif Display"),
            xaxis_title="Week",
            yaxis_title="Average Mood Score (1-10)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(t=50,b=20),
            yaxis=dict(range=[0,11]),
        )
        fig_comp.update_xaxes(gridcolor="#eee")
        fig_comp.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Individual trajectory explorer
        st.markdown("### Individual Trajectory Explorer")
        student_ids = features_df["student_id"].unique()
        sel_student = st.selectbox("Select student", student_ids[:50],
                                    key="traj_student")

        student_data = features_df[
            features_df["student_id"]==sel_student
        ].sort_values("week")

        if not student_data.empty:
            fig_ind = go.Figure()

            # Mood score
            fig_ind.add_trace(go.Scatter(
                x=student_data["week"],
                y=student_data["mood_score"],
                name="Mood Score",
                line=dict(color="#3498db", width=3),
                mode="lines+markers",
                marker=dict(size=8),
            ))

            # Drift score (secondary axis)
            fig_ind.add_trace(go.Scatter(
                x=student_data["week"],
                y=student_data["drift_score"],
                name="Drift Score",
                line=dict(color="#e74c3c", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=6),
                yaxis="y2",
            ))

            # Sleep hours
            if "sleep_hours_mean" in student_data.columns:
                fig_ind.add_trace(go.Bar(
                    x=student_data["week"],
                    y=student_data["sleep_hours_mean"],
                    name="Sleep Hours",
                    marker_color="rgba(52,152,219,0.2)",
                    yaxis="y",
                ))

            fig_ind.update_layout(
                title=f"Full Trajectory — {sel_student}",
                title_font=dict(size=16, family="DM Serif Display"),
                xaxis_title="Week",
                yaxis=dict(title="Mood / Sleep", range=[0,12]),
                yaxis2=dict(title="Drift Score (0-100)",
                            overlaying="y", side="right", range=[0,110]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400, margin=dict(t=50,b=20),
                legend=dict(orientation="h", y=1.12),
            )
            fig_ind.update_xaxes(gridcolor="#eee")
            st.plotly_chart(fig_ind, use_container_width=True)

        # LSTM prediction accuracy
        if not lstm_preds_df.empty:
            st.markdown("### LSTM Prediction Performance")
            acc = (lstm_preds_df["actual_risk"] ==
                   lstm_preds_df["predicted_risk"]).mean()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("LSTM Accuracy", f"{acc*100:.1f}%")
            with c2: st.metric("Total Sequences", f"{len(lstm_preds_df):,}")
            with c3:
                high_acc = lstm_preds_df[lstm_preds_df["actual_risk"]==2]
                h_acc = (high_acc["actual_risk"]==high_acc["predicted_risk"]).mean()
                st.metric("High-Risk Detection Rate", f"{h_acc*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STUDENT CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💚 Student Check-In":
    st.title("💚 Wellness Check-In")
    st.markdown("*A safe space to talk. Everything here stays private.*")
    st.markdown("---")

    features_df    = load_features()
    predictions_df = load_predictions()

    # Student selector
    if not features_df.empty:
        students   = sorted(features_df["student_id"].unique())
        student_id = st.selectbox("Your Student ID (demo):", students[:20])

        # Build context
        student_feat = features_df[features_df["student_id"]==student_id]
        student_pred = predictions_df[predictions_df["student_id"]==student_id] \
            if not predictions_df.empty else pd.DataFrame()

        context = {}
        if not student_feat.empty:
            latest = student_feat.sort_values("week").iloc[-1]
            pred   = student_pred.sort_values("week").iloc[-1] \
                if not student_pred.empty else None
            risk_label = int(pred["predicted_risk"]) if pred is not None else 0

            concerns = []
            if latest.get("dev_sleep_hours", 0)  < -1.5: concerns.append("sleep reduction")
            if latest.get("dev_social_score", 0) < -2.0: concerns.append("social withdrawal")
            if latest.get("mood_score", 10)       < 5.0:  concerns.append("low mood")
            if latest.get("late_night_sum", 0)    > 4:    concerns.append("late-night activity")

            context = {
                "risk_label":  risk_label,
                "risk_level":  {0:"low",1:"moderate",2:"elevated"}[risk_label],
                "mood_score":  round(float(latest.get("mood_score", 7)), 1),
                "concerns":    concerns,
                "week":        int(latest.get("week", 1)),
            }
    else:
        student_id = "DEMO"
        context    = {"risk_label":0,"risk_level":"low",
                      "mood_score":7,"week":1,"concerns":[]}

    # Chat
    if "student_chat" not in st.session_state:
        st.session_state.student_chat = []

    if not st.session_state.student_chat:
        hour = datetime.now().hour
        greeting = (
            f"{'Good morning' if hour<12 else 'Good afternoon' if hour<17 else 'Good evening'}! "
            f"👋 I'm MindBridge, your wellness companion. "
            f"I'm here to listen and help you thrive this semester. "
            f"How are you feeling today?"
        )
        st.session_state.student_chat.append({"role":"assistant","content":greeting})

    # Quick prompts
    if len(st.session_state.student_chat) <= 1:
        cols = st.columns(2)
        for i, prompt in enumerate([
            "I've been really stressed lately",
            "I'm struggling to sleep",
            "I feel isolated from everyone",
            "I need help with study habits",
        ]):
            with cols[i%2]:
                if st.button(prompt, key=f"sp_{i}", use_container_width=True):
                    st.session_state.student_chat.append({"role":"user","content":prompt})
                    st.rerun()

    # Display messages
    for msg in st.session_state.student_chat:
        css = "chat-user" if msg["role"]=="user" else "chat-ai"
        icon = "👤" if msg["role"]=="user" else "💚"
        st.markdown(
            f'<div class="{css}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    user_input = st.chat_input("How are you feeling today?")
    if user_input:
        st.session_state.student_chat.append({"role":"user","content":user_input})
        concerns_text = "\n".join(f"- {c}" for c in context.get("concerns",[]))
        escalation = (
            "At an appropriate moment, gently mention campus counseling services."
            if context.get("risk_label",0) >= 2 else ""
        )
        system = f"""You are MindBridge, a warm empathetic wellness companion for college students.
Be casual, supportive, non-clinical. Ask open questions. Offer practical tips.
NEVER reveal you have behavioral data or risk scores.
If serious distress is expressed, provide Crisis Text Line: text HOME to 741741.
Keep responses to 3-5 sentences.
Internal context (DO NOT SHARE): mood={context.get('mood_score')}, risk={context.get('risk_level')}
Concerns to gently explore: {concerns_text}
{escalation}"""
        with st.spinner("Thinking..."):
            reply = call_claude(system, [
                {"role":m["role"],"content":m["content"]}
                for m in st.session_state.student_chat
            ])
        st.session_state.student_chat.append({"role":"assistant","content":reply})
        st.rerun()

    with st.expander("🆘 Crisis Resources"):
        st.markdown("""
        - 📱 **Crisis Text Line:** Text HOME to **741741**
        - 📞 **988 Suicide & Crisis Lifeline:** Call/text **988**
        - 🚨 **Emergency:** **911**
        """)

    if st.button("🗑️ Clear", key="clr_s"):
        st.session_state.student_chat = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COUNSELOR ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏥 Counselor Assistant":
    st.title("🏥 Counselor Triage Assistant")
    st.markdown("*AI-powered professional support for mental health counselors.*")
    st.warning("⚠️ AI predictions are decision-support only. Clinical judgment always takes precedence.")
    st.markdown("---")

    predictions_df = load_predictions()
    features_df    = load_features()
    shap_df        = load_shap()

    # Priority list
    if not predictions_df.empty:
        st.markdown("### 🎯 Priority Outreach List")
        high_risk = (
            predictions_df[predictions_df["predicted_risk"]==2]
            .groupby("student_id")["prob_high"].max()
            .reset_index()
            .sort_values("prob_high", ascending=False)
            .head(10)
        )
        if not high_risk.empty:
            high_risk["prob_high"] = high_risk["prob_high"].apply(lambda x: f"{x*100:.0f}%")
            high_risk.columns = ["Student ID", "Confidence"]
            st.dataframe(high_risk, use_container_width=True, hide_index=True)

        # Student briefing
        st.markdown("### 📋 Generate Student Briefing")
        high_ids = predictions_df[predictions_df["predicted_risk"]==2]["student_id"].unique()
        if len(high_ids) > 0:
            sel = st.selectbox("Select high-risk student:", high_ids[:15])
            if st.button("Generate Briefing"):
                student_feat = features_df[features_df["student_id"]==sel]
                student_pred = predictions_df[predictions_df["student_id"]==sel]
                if not student_feat.empty:
                    latest = student_feat.sort_values("week").iloc[-1]
                    concerns = []
                    if latest.get("dev_sleep_hours",0)  < -1.5: concerns.append("significant sleep reduction")
                    if latest.get("dev_social_score",0) < -2.0: concerns.append("notable social withdrawal")
                    if latest.get("dev_lms_logins",0)   < -2.0: concerns.append("academic disengagement")
                    if latest.get("mood_score",10)       < 5.0:  concerns.append("low mood score")
                    if latest.get("late_night_sum",0)    > 4:    concerns.append("frequent late-night activity")
                    concerns_str = "\n".join(f"- {c}" for c in concerns) or "No specific concerns"
                    prompt = f"""Generate a professional counselor briefing for student {sel}:
Behavioral concerns: {concerns_str}
Mood score: {latest.get('mood_score','N/A')}/10
Drift score: {latest.get('drift_score','N/A')}/100
Include: (1) key observations, (2) outreach approach, (3) conversation starters, (4) resources."""
                    system = """You are MindBridge Counselor Assistant.
Provide professional, actionable counselor briefings. Use clinical language.
Always note that AI is decision-support only, not diagnosis."""
                    with st.spinner("Generating briefing..."):
                        briefing = call_claude(system, [{"role":"user","content":prompt}])
                    st.markdown("#### Counselor Briefing")
                    st.markdown(briefing)
                    st.download_button(
                        "📥 Download Briefing",
                        data=briefing,
                        file_name=f"briefing_{sel}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

    st.markdown("---")
    st.markdown("### 💬 Ask the Counselor Assistant")

    if "counselor_chat" not in st.session_state:
        st.session_state.counselor_chat = []

    if not st.session_state.counselor_chat:
        st.session_state.counselor_chat.append({
            "role":"assistant",
            "content": ("Hello! I'm the MindBridge Counselor Assistant. I can help you "
                        "understand student risk patterns, plan outreach strategies, draft "
                        "sensitive messages, and navigate campus resources.")
        })

    # Quick prompts
    if len(st.session_state.counselor_chat) <= 1:
        cols = st.columns(2)
        for i, prompt in enumerate([
            "How should I approach a high-risk student?",
            "Draft an outreach email for a withdrawn student",
            "What behavioral signals matter most?",
            "What campus resources should I recommend?",
        ]):
            with cols[i%2]:
                if st.button(prompt, key=f"cp_{i}", use_container_width=True):
                    st.session_state.counselor_chat.append({"role":"user","content":prompt})
                    st.rerun()

    for msg in st.session_state.counselor_chat:
        role = "user" if msg["role"]=="user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    c_input = st.chat_input("Ask about outreach strategies, student patterns...")
    if c_input:
        st.session_state.counselor_chat.append({"role":"user","content":c_input})
        top5 = shap_df.head(5)["feature"].tolist() if not shap_df.empty else []
        system = f"""You are MindBridge Counselor Assistant.
Help mental health counselors with student triage, outreach strategies, and resources.
Use professional clinical language. Always emphasize AI is decision-support only.
Top model risk indicators: {', '.join(top5)}"""
        with st.spinner("Analyzing..."):
            reply = call_claude(system, [
                {"role":m["role"],"content":m["content"]}
                for m in st.session_state.counselor_chat
            ])
        st.session_state.counselor_chat.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("🗑️ Clear", key="clr_c"):
        st.session_state.counselor_chat = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Reports":
    st.title("📋 Downloadable Reports")
    st.markdown("*Export data and reports for clinical use.*")
    st.markdown("---")

    predictions_df = load_predictions()
    features_df    = load_features()
    shap_df        = load_shap()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 Data Exports")

        if not predictions_df.empty:
            st.download_button(
                "📊 Download All Predictions (CSV)",
                data=predictions_df.to_csv(index=False),
                file_name="mindbridge_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            high_risk = predictions_df[predictions_df["predicted_risk"]==2]
            if not high_risk.empty:
                st.download_button(
                    "🔴 Download High-Risk Students (CSV)",
                    data=high_risk.to_csv(index=False),
                    file_name="mindbridge_high_risk.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if not shap_df.empty:
            st.download_button(
                "🔍 Download SHAP Feature Importance (CSV)",
                data=shap_df.to_csv(index=False),
                file_name="mindbridge_shap.csv",
                mime="text/csv",
                use_container_width=True
            )

        if not features_df.empty:
            st.download_button(
                "📋 Download Full Feature Dataset (CSV)",
                data=features_df.to_csv(index=False),
                file_name="mindbridge_features.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        st.markdown("### 📈 Model Evaluation Report")
        report_path = "results/evaluation_report.txt"
        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read()
            st.text_area("Evaluation Report", report_text, height=300)
            st.download_button(
                "📥 Download Evaluation Report",
                data=report_text,
                file_name="mindbridge_evaluation_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Run Module 3 to generate the evaluation report.")

        st.markdown("### 📸 Saved Plots")
        plots = [
            ("plots/shap_importance.png",    "SHAP Feature Importance"),
            ("plots/confusion_matrix.png",   "Confusion Matrix"),
            ("plots/risk_distribution.png",  "Risk Distribution"),
            ("plots/lstm_training_curve.png","LSTM Training Curve"),
            ("plots/mood_trajectories.png",  "Mood Trajectories"),
        ]
        for path, label in plots:
            if os.path.exists(path):
                st.markdown(f"✅ {label}")
                with open(path, "rb") as f:
                    st.download_button(
                        f"📥 {label}",
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="image/png",
                        key=f"dl_{label}",
                        use_container_width=True
                    )
            else:
                st.markdown(f"❌ {label} (not generated yet)")
