"""
MindBridge — Module 1: Synthetic Student Data Generator
=========================================================
Generates 500 realistic student behavioral profiles across 16 weeks
(one semester) with three mental health trajectories:
  - HEALTHY   (~50%): stable patterns throughout
  - DECLINING (~30%): gradual behavioral drift mid-semester
  - CRISIS    (~20%): severe deterioration, late semester

Each student profile contains daily behavioral signals:
  - Sleep hours & bedtime
  - LMS (Learning Management System) login frequency
  - Assignment submission timing (days before/after deadline)
  - Study session duration
  - Mood check-in score (weekly self-report, 1-10)
  - Social engagement score
  - Dining hall visits (proxy for leaving room / social contact)

Outputs:
  - data/students.csv          → student metadata
  - data/daily_behavior.csv    → daily behavioral signals (500 x 112 days)
  - data/weekly_mood.csv       → weekly self-reported mood scores
  - data/labels.csv            → ground truth risk labels per student per week

Based on:
  - Coyne et al. (2011) — Sleep and depression in college students
  - Harman & Ditzler (2017) — LMS engagement as academic predictor
  - Eisenberg et al. (2019) — Mental health & academic performance
"""

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
N_STUDENTS       = 500
N_WEEKS          = 16       # one semester
N_DAYS           = N_WEEKS * 7
SEMESTER_START   = datetime(2024, 9, 2)  # Monday, fall semester

TRAJECTORY_SPLIT = {
    "healthy":   0.50,   # 250 students
    "declining": 0.30,   # 150 students
    "crisis":    0.20,   # 100 students
}

MAJORS = [
    "Computer Science", "Psychology", "Business", "Biology",
    "Engineering", "English", "Nursing", "Economics",
    "Political Science", "Mathematics"
]

DORMS = ["North Hall", "South Hall", "East Hall", "West Hall",
         "Graduate House", "Off-Campus"]

YEARS = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]

os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def add_noise(value, noise_std, min_val=None, max_val=None):
    noisy = value + np.random.normal(0, noise_std)
    if min_val is not None or max_val is not None:
        noisy = clamp(noisy, min_val or -np.inf, max_val or np.inf)
    return noisy

def sigmoid_decline(week, onset_week, steepness=0.5):
    """Returns a 0→1 decline factor that accelerates after onset_week."""
    x = (week - onset_week) * steepness
    return 1 / (1 + np.exp(-x))


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE GENERATORS (healthy student norms)
# ══════════════════════════════════════════════════════════════════════════════

def healthy_sleep(day_of_week):
    """Sleep hours: 7-8hrs weekdays, 8-9hrs weekends."""
    is_weekend = day_of_week >= 5
    base = 8.2 if is_weekend else 7.3
    return add_noise(base, 0.6, 5.0, 10.0)

def healthy_bedtime(day_of_week):
    """Bedtime hour (24h): ~23:00 weekdays, ~00:30 weekends."""
    is_weekend = day_of_week >= 5
    base = 23.5 if is_weekend else 23.0
    return add_noise(base, 0.5, 21.0, 25.0)  # >24 = past midnight

def healthy_lms_logins(day_of_week):
    """LMS logins per day: higher on weekdays."""
    is_weekend = day_of_week >= 5
    base = 1.5 if is_weekend else 4.2
    return max(0, int(add_noise(base, 1.2)))

def healthy_study_hours(day_of_week):
    """Study session hours per day."""
    is_weekend = day_of_week >= 5
    base = 2.5 if is_weekend else 3.8
    return add_noise(base, 1.0, 0.0, 8.0)

def healthy_social_score():
    """Social engagement score 0-10."""
    return add_noise(7.2, 1.2, 3.0, 10.0)

def healthy_dining_visits(day_of_week):
    """Dining hall visits per day (proxy for leaving room)."""
    is_weekend = day_of_week >= 5
    base = 2.0 if is_weekend else 2.5
    return max(0, int(add_noise(base, 0.7)))

def healthy_assignment_delta():
    """Days before deadline submission (positive = early, negative = late)."""
    return add_noise(1.8, 1.5, -3.0, 7.0)

def healthy_mood():
    """Weekly self-reported mood score 1-10."""
    return add_noise(7.5, 1.0, 5.0, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY MODIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_declining_trajectory(signals, week, onset_week=6):
    """
    Gradual deterioration starting at onset_week.
    Mirrors clinical research on semester-long burnout patterns.
    """
    severity = sigmoid_decline(week, onset_week, steepness=0.4)
    if severity < 0.05:
        return signals  # no change yet

    s = signals.copy()
    s["sleep_hours"]       = clamp(s["sleep_hours"]       - severity * 2.0, 3.0, 10.0)
    s["bedtime_hour"]      = clamp(s["bedtime_hour"]       + severity * 2.5, 21.0, 30.0)
    s["lms_logins"]        = max(0, int(s["lms_logins"]    - severity * 2.0))
    s["study_hours"]       = clamp(s["study_hours"]        - severity * 1.5, 0.0, 8.0)
    s["social_score"]      = clamp(s["social_score"]       - severity * 3.0, 0.0, 10.0)
    s["dining_visits"]     = max(0, int(s["dining_visits"] - severity * 1.2))
    s["assignment_delta"]  = clamp(s["assignment_delta"]   - severity * 3.0, -7.0, 7.0)
    return s

def apply_crisis_trajectory(signals, week, onset_week=10):
    """
    Severe deterioration with faster onset and steeper decline.
    Models acute mental health crisis patterns.
    """
    severity = sigmoid_decline(week, onset_week, steepness=0.8)
    if severity < 0.05:
        return signals

    s = signals.copy()
    s["sleep_hours"]       = clamp(s["sleep_hours"]       - severity * 3.5, 2.0, 10.0)
    s["bedtime_hour"]      = clamp(s["bedtime_hour"]       + severity * 5.0, 21.0, 32.0)
    s["lms_logins"]        = max(0, int(s["lms_logins"]    - severity * 3.5))
    s["study_hours"]       = clamp(s["study_hours"]        - severity * 3.0, 0.0, 8.0)
    s["social_score"]      = clamp(s["social_score"]       - severity * 6.0, 0.0, 10.0)
    s["dining_visits"]     = max(0, int(s["dining_visits"] - severity * 2.0))
    s["assignment_delta"]  = clamp(s["assignment_delta"]   - severity * 6.0, -14.0, 7.0)
    return s

def apply_mood_trajectory(base_mood, week, trajectory, onset_week):
    """Weekly mood decline aligned with behavioral trajectory."""
    if trajectory == "healthy":
        return add_noise(base_mood, 0.8, 5.0, 10.0)

    severity = sigmoid_decline(week, onset_week, steepness=0.4 if trajectory == "declining" else 0.8)
    drop = severity * (3.0 if trajectory == "declining" else 6.0)
    return clamp(add_noise(base_mood - drop, 0.8), 1.0, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
# RISK LABEL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_label(trajectory, week, onset_week):
    """
    Ground truth risk label: 0=Low, 1=Medium, 2=High
    Used as target variable for ML model training.
    """
    if trajectory == "healthy":
        return 0

    severity = sigmoid_decline(week, onset_week)

    if trajectory == "declining":
        if severity < 0.2:  return 0
        if severity < 0.6:  return 1
        return 2

    if trajectory == "crisis":
        if severity < 0.15: return 0
        if severity < 0.45: return 1
        return 2

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_students():
    """Generate student metadata table."""
    trajectories = (
        ["healthy"]   * int(N_STUDENTS * TRAJECTORY_SPLIT["healthy"]) +
        ["declining"] * int(N_STUDENTS * TRAJECTORY_SPLIT["declining"]) +
        ["crisis"]    * int(N_STUDENTS * TRAJECTORY_SPLIT["crisis"])
    )
    # Fill any rounding gap
    while len(trajectories) < N_STUDENTS:
        trajectories.append("healthy")
    random.shuffle(trajectories)

    students = []
    for i in range(N_STUDENTS):
        traj = trajectories[i]

        # Onset week varies per student (realistic variation)
        if traj == "healthy":
            onset = None
        elif traj == "declining":
            onset = random.randint(4, 9)   # starts mid-semester
        else:
            onset = random.randint(8, 13)  # starts late semester

        # Baseline mood — slightly lower for at-risk groups
        base_mood = {
            "healthy":   add_noise(7.8, 0.8, 5.5, 10.0),
            "declining": add_noise(7.2, 0.9, 4.5, 9.5),
            "crisis":    add_noise(6.8, 1.0, 4.0, 9.0),
        }[traj]

        students.append({
            "student_id":   f"STU{str(i+1).zfill(4)}",
            "age":          random.randint(18, 28),
            "year":         random.choice(YEARS),
            "major":        random.choice(MAJORS),
            "dorm":         random.choice(DORMS),
            "gpa_start":    round(add_noise(3.2, 0.4, 1.5, 4.0), 2),
            "trajectory":   traj,
            "onset_week":   onset,
            "base_mood":    round(base_mood, 2),
            "has_history":  random.random() < (0.15 if traj == "healthy" else 0.40),
            "scholarship":  random.random() < 0.35,
            "first_gen":    random.random() < 0.28,
        })

    return pd.DataFrame(students)


def generate_daily_behavior(students_df):
    """Generate day-by-day behavioral signals for each student."""
    records = []

    for _, student in students_df.iterrows():
        sid        = student["student_id"]
        traj       = student["trajectory"]
        onset_week = student["onset_week"] if student["onset_week"] else 999

        for day in range(N_DAYS):
            date        = SEMESTER_START + timedelta(days=day)
            week        = day // 7
            day_of_week = date.weekday()  # 0=Mon, 6=Sun

            # Build healthy baseline signals
            signals = {
                "sleep_hours":      healthy_sleep(day_of_week),
                "bedtime_hour":     healthy_bedtime(day_of_week),
                "lms_logins":       healthy_lms_logins(day_of_week),
                "study_hours":      healthy_study_hours(day_of_week),
                "social_score":     healthy_social_score(),
                "dining_visits":    healthy_dining_visits(day_of_week),
                "assignment_delta": healthy_assignment_delta(),
            }

            # Apply trajectory modifier
            if traj == "declining":
                signals = apply_declining_trajectory(signals, week, onset_week)
            elif traj == "crisis":
                signals = apply_crisis_trajectory(signals, week, onset_week)

            records.append({
                "student_id":       sid,
                "date":             date.strftime("%Y-%m-%d"),
                "week":             week + 1,
                "day_of_week":      day_of_week,
                "sleep_hours":      round(signals["sleep_hours"], 2),
                "bedtime_hour":     round(signals["bedtime_hour"], 2),
                "lms_logins":       int(signals["lms_logins"]),
                "study_hours":      round(signals["study_hours"], 2),
                "social_score":     round(signals["social_score"], 2),
                "dining_visits":    int(signals["dining_visits"]),
                "assignment_delta": round(signals["assignment_delta"], 2),
            })

    return pd.DataFrame(records)


def generate_weekly_mood(students_df):
    """Generate weekly self-reported mood check-ins."""
    records = []

    for _, student in students_df.iterrows():
        sid        = student["student_id"]
        traj       = student["trajectory"]
        onset_week = student["onset_week"] if student["onset_week"] else 999
        base_mood  = student["base_mood"]

        for week in range(N_WEEKS):
            # ~15% of students skip a check-in (realistic non-response)
            if random.random() < 0.15:
                mood = None
            else:
                mood = round(apply_mood_trajectory(base_mood, week, traj, onset_week), 2)

            records.append({
                "student_id":   sid,
                "week":         week + 1,
                "mood_score":   mood,
                "check_in_date": (SEMESTER_START + timedelta(weeks=week, days=6)).strftime("%Y-%m-%d"),
            })

    return pd.DataFrame(records)


def generate_labels(students_df):
    """Generate ground truth risk labels per student per week."""
    records = []

    for _, student in students_df.iterrows():
        sid        = student["student_id"]
        traj       = student["trajectory"]
        onset_week = student["onset_week"] if student["onset_week"] else 999

        for week in range(N_WEEKS):
            label = compute_risk_label(traj, week, onset_week)
            records.append({
                "student_id":   sid,
                "week":         week + 1,
                "risk_label":   label,
                "risk_name":    ["Low", "Medium", "High"][label],
                "trajectory":   traj,
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# RUN & SAVE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("   🧠 MindBridge — Synthetic Data Generator")
    print("=" * 55)

    print("\n[1/4] Generating student profiles...")
    students_df = generate_students()
    students_df.to_csv("data/students.csv", index=False)
    print(f"      ✅ {len(students_df)} students created")
    print(f"         Healthy:   {len(students_df[students_df.trajectory=='healthy'])}")
    print(f"         Declining: {len(students_df[students_df.trajectory=='declining'])}")
    print(f"         Crisis:    {len(students_df[students_df.trajectory=='crisis'])}")

    print("\n[2/4] Generating daily behavioral signals...")
    daily_df = generate_daily_behavior(students_df)
    daily_df.to_csv("data/daily_behavior.csv", index=False)
    print(f"      ✅ {len(daily_df):,} daily records ({N_STUDENTS} students × {N_DAYS} days)")

    print("\n[3/4] Generating weekly mood check-ins...")
    mood_df = generate_weekly_mood(students_df)
    mood_df.to_csv("data/weekly_mood.csv", index=False)
    missing = mood_df["mood_score"].isna().sum()
    print(f"      ✅ {len(mood_df):,} weekly records ({missing} skipped check-ins)")

    print("\n[4/4] Generating ground truth risk labels...")
    labels_df = generate_labels(students_df)
    labels_df.to_csv("data/labels.csv", index=False)
    dist = labels_df["risk_name"].value_counts()
    print(f"      ✅ {len(labels_df):,} weekly labels")
    print(f"         Low:    {dist.get('Low', 0):,}")
    print(f"         Medium: {dist.get('Medium', 0):,}")
    print(f"         High:   {dist.get('High', 0):,}")

    print("\n" + "=" * 55)
    print("   📁 Files saved to /data/")
    print("      students.csv       → student metadata")
    print("      daily_behavior.csv → behavioral signals")
    print("      weekly_mood.csv    → mood check-ins")
    print("      labels.csv         → ML target labels")
    print("=" * 55)
    print("\n✅ Module 1 complete. Ready for Module 2: Feature Engineering.")
