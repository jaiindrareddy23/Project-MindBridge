"""
MindBridge — Module 5: AI Counselor Chatbot
============================================
An empathetic, privacy-preserving AI assistant powered by Claude API.

Two distinct interfaces:
  1. STUDENT VIEW  — Personal wellness check-in chatbot
     - Empathetic conversation about how the student is feeling
     - Personalized tips based on their behavioral patterns
     - Gentle escalation prompts if risk signals are detected
     - Never reveals the risk score to the student directly

  2. COUNSELOR VIEW — Professional triage assistant
     - Summarizes at-risk student profiles (anonymized)
     - Suggests outreach strategies per student
     - Answers clinical questions about behavioral patterns
     - Provides campus resource recommendations

Privacy principles:
  - Student chatbot never shares raw risk scores with the student
  - Counselor view uses anonymized IDs unless counselor has clearance
  - All conversations are session-only (not stored)
  - Escalation only triggers soft prompts, never automated alerts

Input:  data/predictions.csv, data/shap_values.csv, data/features.csv
Output: Streamlit chatbot interface (run standalone or integrated into app.py)
"""

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import requests # type: ignore
import json
import os
from datetime import datetime

# ── Page config (only when run standalone) ────────────────────────────────────
if __name__ == "__main__" or "chatbot" in __file__:
    pass  # config handled by app.py in Module 6

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-6"
MAX_TOKENS        = 800


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_predictions():
    path = "data/predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_shap():
    path = "data/shap_values.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_features():
    path = "data/features.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT PROFILE BUILDER
# Builds a context summary for a given student to feed into Claude
# ══════════════════════════════════════════════════════════════════════════════

def build_student_context(student_id: str, features_df: pd.DataFrame,
                           predictions_df: pd.DataFrame) -> dict:
    """
    Builds a structured context dict for a student.
    Used to personalize Claude's responses without revealing raw scores.
    """
    if features_df.empty or predictions_df.empty:
        return {}

    # Get latest week data
    student_features = features_df[
        features_df["student_id"] == student_id
    ].sort_values("week")

    student_preds = predictions_df[
        predictions_df["student_id"] == student_id
    ].sort_values("week")

    if student_features.empty:
        return {}

    latest = student_features.iloc[-1]
    pred   = student_preds.iloc[-1] if not student_preds.empty else None

    # Identify which signals are most concerning
    concerns = []
    if latest.get("dev_sleep_hours", 0) < -1.5:
        concerns.append("significant sleep reduction compared to their normal pattern")
    if latest.get("dev_social_score", 0) < -2.0:
        concerns.append("notable social withdrawal")
    if latest.get("dev_lms_logins", 0) < -2.0:
        concerns.append("decreased academic engagement")
    if latest.get("mood_score", 10) < 5.0:
        concerns.append("lower self-reported mood")
    if latest.get("slope_sleep_hours", 0) < -0.3:
        concerns.append("worsening sleep trend over past 4 weeks")
    if latest.get("late_night_sum", 0) > 4:
        concerns.append("frequent late-night activity")
    if latest.get("assignment_delta_mean", 2) < -1.0:
        concerns.append("assignment submissions becoming late")

    # Risk level (used internally, not shown to student)
    risk_label = pred["predicted_risk"] if pred is not None else 0
    risk_name  = {0: "low", 1: "moderate", 2: "elevated"}[risk_label]

    return {
        "student_id":   student_id,
        "risk_level":   risk_name,
        "risk_label":   int(risk_label),
        "drift_score":  round(float(latest.get("drift_score", 0)), 1),
        "mood_score":   round(float(latest.get("mood_score", 7)), 1),
        "concerns":     concerns,
        "week":         int(latest.get("week", 1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE API CALLER
# ══════════════════════════════════════════════════════════════════════════════

def call_claude(system_prompt: str, messages: list) -> str:
    """
    Calls the Anthropic Claude API.
    Returns the assistant's text response.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return (
            "⚠️ API key not found. Please set your ANTHROPIC_API_KEY "
            "environment variable:\n\n"
            "```bash\nexport ANTHROPIC_API_KEY=your_key_here\n```"
        )

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json",
                     "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            json={
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "system":     system_prompt,
                "messages":   messages,
            },
            timeout=30
        )
        data = response.json()

        if "content" in data and len(data["content"]) > 0:
            return data["content"][0]["text"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"
        else:
            return "Sorry, I couldn't generate a response. Please try again."

    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except Exception as e:
        return f"Connection error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def get_student_system_prompt(context: dict) -> str:
    """
    System prompt for student-facing chatbot.
    Warm, empathetic, never reveals risk score.
    Gently escalates if signals are concerning.
    """
    concerns_text = ""
    if context.get("concerns"):
        concerns_text = f"""
Based on the student's recent patterns, you may want to gently explore:
{chr(10).join(f'- {c}' for c in context['concerns'])}

Do NOT mention these directly. Instead, ask open questions like:
'How has your sleep been lately?' or 'Have you been staying connected with friends?'
"""

    escalation_note = ""
    if context.get("risk_label", 0) >= 2:
        escalation_note = """
IMPORTANT: This student's patterns suggest they may benefit from professional support.
At an appropriate moment in the conversation, gently mention that campus counseling
services are available and encourage them to reach out. Do this naturally, not abruptly.
Example: 'It sounds like things have been really tough lately. Have you ever thought
about chatting with someone at the campus counseling center? They're really helpful.'
"""

    return f"""You are MindBridge, a warm and empathetic wellness companion for college students.
Your role is to check in on how students are doing, offer a supportive ear, and provide
practical wellness tips — like a caring peer advisor, not a therapist.

Core principles:
- Be genuinely warm, casual, and non-clinical in your language
- Ask open-ended questions to understand how the student is feeling
- Offer practical, actionable wellness tips (sleep hygiene, study breaks, social connection)
- NEVER diagnose, never use clinical terms like 'depression' or 'anxiety disorder'
- NEVER reveal that you have access to their behavioral data or risk scores
- NEVER say things like 'I can see your sleep has declined' — instead ask naturally
- If a student expresses serious distress or mentions self-harm, immediately provide
  the Crisis Text Line (text HOME to 741741) and campus counseling contacts
- Keep responses concise (3-5 sentences max) unless the student clearly wants to talk more
- Use casual, friendly language — you're a peer, not a doctor

Student context (INTERNAL — never reveal this to student):
  Current week: {context.get('week', 'N/A')}
  Mood score:   {context.get('mood_score', 'N/A')}/10
  Risk level:   {context.get('risk_level', 'low')} (DO NOT SHARE THIS)
{concerns_text}
{escalation_note}"""


def get_counselor_system_prompt(shap_df: pd.DataFrame) -> str:
    """
    System prompt for counselor-facing assistant.
    Professional, data-informed, actionable.
    """
    top_features = ""
    if not shap_df.empty:
        top5 = shap_df.head(5)["feature"].tolist()
        top_features = f"Top risk indicators in current model: {', '.join(top5)}"

    return f"""You are MindBridge Counselor Assistant, a professional AI tool
designed to help campus mental health counselors prioritize and support students.

Your role:
- Help counselors understand student behavioral patterns
- Suggest evidence-based outreach strategies for different risk levels
- Answer questions about the MindBridge risk model and its signals
- Recommend campus and national resources
- Help counselors draft sensitive, appropriate outreach messages

Guidelines:
- Use professional, clinical language appropriate for mental health professionals
- Always emphasize that AI predictions are decision-support tools, not diagnoses
- Remind counselors that human judgment always supersedes model output
- Be specific and actionable in recommendations
- Cite behavioral signals (sleep, social withdrawal, academic disengagement)
  as observable patterns, not definitive indicators of any condition

Model context:
{top_features}
Risk levels: Low (0) = routine monitoring, Medium (1) = proactive outreach,
High (2) = priority intervention within 48 hours recommended.

Always remind counselors: 'This AI tool supports but never replaces clinical judgment.'"""


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT CHATBOT UI
# ══════════════════════════════════════════════════════════════════════════════

def render_student_chatbot():
    """Render the student-facing wellness check-in chatbot."""

    st.markdown("## 💚 MindBridge Wellness Check-In")
    st.markdown("*A safe space to talk about how you're doing. "
                "Everything here is just between us.*")
    st.markdown("---")

    # Load data
    features_df    = load_features()
    predictions_df = load_predictions()

    # Student selector (in real app this would be auth-based)
    if not features_df.empty:
        students = sorted(features_df["student_id"].unique())
        student_id = st.selectbox(
            "Select your student ID (demo mode):",
            students[:20],  # show first 20 for demo
            key="student_selector"
        )
        context = build_student_context(student_id, features_df, predictions_df)
    else:
        student_id = "DEMO"
        context    = {"risk_label": 0, "risk_level": "low",
                      "mood_score": 7, "week": 1, "concerns": []}

    # Session state for chat history
    if "student_chat" not in st.session_state:
        st.session_state.student_chat = []

    # Greeting on first load
    if not st.session_state.student_chat:
        hour = datetime.now().hour
        time_greeting = ("Good morning" if hour < 12
                         else "Good afternoon" if hour < 17
                         else "Good evening")
        greeting = (f"{time_greeting}! 👋 I'm MindBridge, your campus wellness "
                    f"companion. I'm here to chat, listen, and share some tips "
                    f"to help you thrive this semester. How are you feeling today?")
        st.session_state.student_chat.append({
            "role": "assistant", "content": greeting
        })

    # Quick prompts
    if len(st.session_state.student_chat) <= 1:
        st.markdown("**Quick starters:**")
        cols = st.columns(2)
        quick_prompts = [
            "I've been feeling really stressed lately",
            "I'm having trouble sleeping",
            "I feel disconnected from everyone",
            "I need some study tips",
        ]
        for i, prompt in enumerate(quick_prompts):
            with cols[i % 2]:
                if st.button(prompt, key=f"sq_{i}", use_container_width=True):
                    st.session_state.student_chat.append({
                        "role": "user", "content": prompt
                    })
                    st.rerun()

    # Display chat
    for msg in st.session_state.student_chat:
        if msg["role"] == "user":
            st.markdown(
                f'<div style="background:#1a4a2e;color:white;padding:12px 16px;'
                f'border-radius:18px 18px 4px 18px;margin:8px 0 8px 60px;'
                f'font-size:0.95rem">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background:#f0faf4;border:1px solid #c3e6cb;'
                f'color:#1a2e22;padding:12px 16px;border-radius:18px 18px 18px 4px;'
                f'margin:8px 60px 8px 0;font-size:0.95rem;line-height:1.6">'
                f'💚 {msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # Input
    user_input = st.chat_input("How are you feeling today?...")

    if user_input:
        st.session_state.student_chat.append({
            "role": "user", "content": user_input
        })

        system_prompt = get_student_system_prompt(context)
        api_messages  = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.student_chat
        ]

        with st.spinner("MindBridge is thinking..."):
            reply = call_claude(system_prompt, api_messages)

        st.session_state.student_chat.append({
            "role": "assistant", "content": reply
        })
        st.rerun()

    # Crisis resources (always visible)
    with st.expander("🆘 Crisis Resources (always available)"):
        st.markdown("""
        **If you're in crisis right now:**
        - 📱 **Crisis Text Line:** Text HOME to **741741**
        - 📞 **988 Suicide & Crisis Lifeline:** Call or text **988**
        - 🏥 **Campus Counseling Center:** [Your university's number here]
        - 🚨 **Emergency:** **911**

        *You are not alone. Help is always available.*
        """)

    # Clear button
    if st.session_state.student_chat:
        if st.button("🗑️ Clear Chat", key="clear_student"):
            st.session_state.student_chat = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# COUNSELOR CHATBOT UI
# ══════════════════════════════════════════════════════════════════════════════

def render_counselor_chatbot():
    """Render the counselor-facing triage assistant."""

    st.markdown("## 🏥 MindBridge Counselor Assistant")
    st.markdown("*Professional AI support for mental health triage and outreach planning.*")
    st.markdown(
        "⚠️ **Reminder:** AI predictions are decision-support tools only. "
        "Clinical judgment always takes precedence.",
        )
    st.markdown("---")

    # Load data
    predictions_df = load_predictions()
    shap_df        = load_shap()
    features_df    = load_features()

    # At-risk student summary
    if not predictions_df.empty:
        high_risk = predictions_df[predictions_df["predicted_risk"] == 2]
        med_risk  = predictions_df[predictions_df["predicted_risk"] == 1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🔴 High Risk Students",   high_risk["student_id"].nunique())
        with c2:
            st.metric("🟡 Medium Risk Students", med_risk["student_id"].nunique())
        with c3:
            st.metric("📊 Total Monitored",
                      predictions_df["student_id"].nunique())

        # Priority list
        st.markdown("### 🎯 Priority Outreach List")
        if not high_risk.empty:
            priority = (
                high_risk.groupby("student_id")["prob_high"]
                .max()
                .reset_index()
                .sort_values("prob_high", ascending=False)
                .head(10)
            )
            priority.columns = ["Student ID", "High Risk Confidence"]
            priority["High Risk Confidence"] = priority["High Risk Confidence"].apply(
                lambda x: f"{x*100:.0f}%"
            )
            st.dataframe(priority, use_container_width=True, hide_index=True)
        else:
            st.info("No high-risk students in current prediction window.")

    st.markdown("---")

    # Session state
    if "counselor_chat" not in st.session_state:
        st.session_state.counselor_chat = []

    # Greeting
    if not st.session_state.counselor_chat:
        greeting = ("Hello! I'm the MindBridge Counselor Assistant. I can help you "
                    "understand student risk patterns, plan outreach strategies, draft "
                    "sensitive messages, and navigate campus resources. What would you "
                    "like to explore today?")
        st.session_state.counselor_chat.append({
            "role": "assistant", "content": greeting
        })

    # Quick counselor prompts
    if len(st.session_state.counselor_chat) <= 1:
        st.markdown("**Quick actions:**")
        cols = st.columns(2)
        quick_prompts = [
            "How should I approach a high-risk student?",
            "What behavioral signals matter most?",
            "Draft an outreach email for a withdrawn student",
            "What campus resources should I recommend?",
        ]
        for i, prompt in enumerate(quick_prompts):
            with cols[i % 2]:
                if st.button(prompt, key=f"cq_{i}", use_container_width=True):
                    st.session_state.counselor_chat.append({
                        "role": "user", "content": prompt
                    })
                    st.rerun()

    # Student-specific analysis
    if not features_df.empty and not predictions_df.empty:
        st.markdown("### 🔍 Analyze Specific Student")
        high_risk_ids = (
            predictions_df[predictions_df["predicted_risk"] == 2]
            ["student_id"].unique()
        )
        if len(high_risk_ids) > 0:
            selected_id = st.selectbox(
                "Select student to analyze:",
                high_risk_ids[:15],
                key="counselor_student_select"
            )
            if st.button("📋 Generate Student Summary", key="gen_summary"):
                context = build_student_context(
                    selected_id, features_df, predictions_df
                )
                concerns_str = (
                    "\n".join(f"- {c}" for c in context.get("concerns", []))
                    or "No specific concerns flagged"
                )
                analysis_prompt = f"""
Please provide a professional counselor briefing for student {selected_id}:
- Risk level: {context.get('risk_level', 'unknown')}
- Behavioral concerns: {concerns_str}
- Current mood score: {context.get('mood_score', 'N/A')}/10
- Behavioral drift score: {context.get('drift_score', 'N/A')}/100

Include: (1) key observations, (2) suggested outreach approach,
(3) specific conversation starters, (4) recommended resources.
Keep it professional and actionable.
"""
                st.session_state.counselor_chat.append({
                    "role": "user", "content": analysis_prompt
                })
                st.rerun()

    # Display chat
    for msg in st.session_state.counselor_chat:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(msg["content"])

    # Input
    counselor_input = st.chat_input("Ask about student patterns, outreach strategies...")

    if counselor_input:
        st.session_state.counselor_chat.append({
            "role": "user", "content": counselor_input
        })

        system_prompt = get_counselor_system_prompt(shap_df)
        api_messages  = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.counselor_chat
        ]

        with st.spinner("Analyzing..."):
            reply = call_claude(system_prompt, api_messages)

        st.session_state.counselor_chat.append({
            "role": "assistant", "content": reply
        })
        st.rerun()

    # Clear
    if st.session_state.counselor_chat:
        if st.button("🗑️ Clear Chat", key="clear_counselor"):
            st.session_state.counselor_chat = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# Run this file directly to test the chatbot: streamlit run chatbot.py
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MindBridge Chatbot",
        page_icon="💚",
        layout="wide"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 MindBridge — AI Counselor System")

    tab1, tab2 = st.tabs(["💚 Student Check-In", "🏥 Counselor Assistant"])

    with tab1:
        render_student_chatbot()

    with tab2:
        render_counselor_chatbot()


if __name__ == "__main__":
    main()
