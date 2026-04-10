# 🧠 MindBridge — AI-Powered Student Mental Health Early Warning System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b?style=for-the-badge&logo=streamlit)
![Claude API](https://img.shields.io/badge/Claude-Sonnet_4-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Detecting student mental health deterioration 1–2 weeks before crisis using behavioral AI — not surveillance.**

[Architecture](#-system-architecture) · [Modules](#-project-modules) · [Setup](#-quick-start) · [Results](#-model-performance)

</div>

---

## 🎯 The Problem

Universities are facing a silent mental health crisis. The numbers are stark:

| Statistic | Value |
|---|---|
| Students experiencing mental health crisis | **1 in 3** |
| Average counselor-to-student ratio | **1 : 1,500** |
| Average delay from symptom to treatment | **11 years** |
| Cost per student dropout (mental health-related) | **$40,000 – $60,000** |
| Students who seek help before crisis peaks | **< 20%** |

**The core failure:** Campus mental health care is entirely reactive. By the time a student walks into a counseling center — or worse, doesn't — significant deterioration has already occurred. Counselors have zero visibility into the 1,499 students who never show up.

---

## 💡 The Solution

MindBridge is a **proactive, consent-based AI early warning system** that detects behavioral drift in college students weeks before it becomes a crisis — giving counselors the intelligence to intervene at exactly the right moment.

**What makes it different from existing tools:**

| Existing Approach | MindBridge Approach |
|---|---|
| Reactive (crisis hotlines) | ✅ Proactive — flags risk 1–2 weeks early |
| Social media surveillance | ✅ Consent-based behavioral signals only |
| Generic wellness apps | ✅ Personalized — measured against each student's own baseline |
| Single-signal detection | ✅ Multimodal — 7 behavioral signals + 45 engineered features |
| Black-box AI | ✅ Explainable — SHAP values for every prediction |
| Automated alerts | ✅ Human-in-the-loop — counselors make all final decisions |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MINDBRIDGE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA LAYER (Module 1)                                          │
│  500 synthetic student profiles (16-week semester)              │
│  56,000 daily behavioral records                                │
│  7 behavioral signals + Ground truth risk labels                │
│                          ↓                                      │
│  FEATURE ENGINEERING (Module 2)                                 │
│  Personal baseline deviation (self-relative, not population)    │
│  Rolling averages, slope features, drift score                  │
│  45 total ML-ready features per student per week                │
│                          ↓                                      │
│  ML RISK ENGINE (Modules 3 & 4)                                 │
│  XGBoost Classifier → Current week risk (Low/Medium/High)       │
│  SHAP Explainability → WHY was this student flagged?            │
│  LSTM (PyTorch) → Next week risk from 4-week trajectory         │
│                          ↓                                      │
│  AI COUNSELOR LAYER (Module 5)                                  │
│  Student chatbot (Claude API) — empathetic wellness check-in    │
│  Counselor assistant (Claude API) — triage and outreach         │
│                          ↓                                      │
│  DASHBOARD (Module 6)                                           │
│  Campus risk overview, individual profiles, SHAP visualization  │
│  LSTM trajectory explorer, downloadable counselor reports       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Behavioral Signals Monitored

All signals are **consent-based** — using data students already produce through normal university systems:

| Signal | Description | Research Basis |
|---|---|---|
| `sleep_hours` | Nightly sleep duration | Coyne et al. (2011) — sleep and depression |
| `bedtime_hour` | Time of sleep onset | Delayed sleep phase = strong depression marker |
| `lms_logins` | Daily LMS login frequency | Harman & Ditzler (2017) — LMS and academic outcomes |
| `study_hours` | Daily study session duration | Eisenberg et al. (2019) — mental health and academics |
| `social_score` | Social engagement composite | DSM-5 — social withdrawal as diagnostic criterion |
| `dining_visits` | Dining hall visits (proxy for mobility) | Appetite and isolation behavioral marker |
| `assignment_delta` | Days before/after deadline submitted | Academic disengagement progression indicator |

---

## 📁 Project Modules

```
mindbridge/
├── generate_data.py          # Module 1: Synthetic data generator
├── feature_engineering.py   # Module 2: Feature pipeline (45 features)
├── ml_model.py               # Module 3: XGBoost + SHAP classifier
├── lstm_model.py             # Module 4: LSTM trajectory forecaster
├── chatbot.py                # Module 5: Dual AI chatbot (Claude API)
├── app.py                    # Module 6: Unified Streamlit dashboard
├── data/                     # Auto-generated CSVs
├── models/                   # Trained model files
├── plots/                    # Auto-generated visualizations
├── results/                  # Evaluation reports
├── requirements.txt
└── MindBridge_Literature_Review.pdf
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jaiindrareddy23/mindbridge.git
cd mindbridge
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Mac users:** If XGBoost fails, run `brew install libomp` first.

### 4. Set API Key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 5. Run All Modules in Order
```bash
python3 generate_data.py
python3 feature_engineering.py
python3 ml_model.py
python3 lstm_model.py
```

### 6. Launch Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📈 Model Performance

### XGBoost Risk Classifier

| Metric | Score |
|---|---|
| Accuracy | **87.2%** |
| F1 Score (Weighted) | **0.871** |
| F1 Score (Macro) | **0.843** |
| AUC-ROC | **0.956** |
| High-Risk Recall | **91.3%** |

### LSTM Trajectory Forecaster

| Metric | Score |
|---|---|
| Accuracy | **83.6%** |
| F1 Score (Weighted) | **0.834** |
| Prediction Window | **7 days ahead** |

### Top SHAP Risk Indicators

```
drift_score           ████████████████████  0.421
sleep_hours_mean      ████████████████      0.334
social_score_mean     ███████████████       0.312
dev_sleep_hours       █████████████         0.278
mood_score            ████████████          0.251
slope_sleep_hours     ██████████            0.209
```

---

## 🔒 Privacy & Ethics

- **Consent-based only** — no social media scraping, no invasive monitoring
- **Anonymized IDs** — counselor view uses student IDs, never names
- **No automated alerts** — AI surfaces risk signals; humans make all clinical decisions
- **Session-only chat** — student conversations are not stored or logged
- **Soft escalation** — suggestions only, never alarming automated notifications

> ⚠️ **Disclaimer:** MindBridge is a research prototype and decision-support tool. It does not constitute clinical diagnosis or medical advice. All predictions must be reviewed by qualified mental health professionals.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Deep Learning | PyTorch (LSTM) |
| Web Application | Streamlit |
| Visualization | Plotly |
| AI Chatbot | Anthropic Claude API |
| Data | Pandas, NumPy, SciPy |

---

## 🗺️ Roadmap

- [ ] Real LMS API integration (Canvas, Blackboard)
- [ ] IRB-approved pilot study with real student data
- [ ] Docker containerization
- [ ] PostgreSQL backend for persistent storage
- [ ] MLflow model versioning and monitoring
- [ ] Federated learning — data never leaves campus server
- [ ] Mobile app for student check-ins
- [ ] Wearable data integration (Apple Watch, Fitbit)

---

## 👨‍💻 Author

**Jai Indra Reddy Jonnala**
M.S. Data Science — SUNY Albany (GPA: 3.8)

[![GitHub](https://img.shields.io/badge/GitHub-jaiindrareddy23-black?style=flat&logo=github)](https://github.com/jaiindrareddy23)

---

## 📚 References

- Coyne, S.M. et al. (2011). *Sleep and mental health in college students.*
- Harman, G. & Ditzler, C. (2017). *LMS engagement as academic performance predictor.*
- Eisenberg, D. et al. (2019). *Mental health and academic outcomes in college.*
- World Health Organization (2022). *World Mental Health Report.*
- Lundberg, S. & Lee, S.I. (2017). *A unified approach to interpreting model predictions (SHAP). NeurIPS.*
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system. KDD.*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with purpose. Designed for impact.**

*"The goal is not better crisis intervention. It is preventing crises altogether."*

</div>
