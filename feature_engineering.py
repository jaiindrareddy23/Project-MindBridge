"""
MindBridge — Module 2: Feature Engineering Pipeline
=====================================================
Transforms raw daily behavioral signals into meaningful ML features.

Raw signals alone are weak predictors. What matters is CHANGE over time.
This module computes:

  1. Rolling averages       → smooth out daily noise, reveal trends
  2. Baseline deviation     → how far has THIS student drifted from THEIR Week 1?
  3. Behavioral drift score → single composite drift metric per student per week
  4. Sleep irregularity     → variance in bedtime (not just average)
  5. Late-night flag        → logins between 1am–4am
  6. Assignment delay trend → is submission timing getting worse each week?
  7. Mood trajectory        → rate of mood change (slope), not just score
  8. Engagement drop rate   → LMS + social + dining combined withdrawal signal

Input:  data/daily_behavior.csv, data/weekly_mood.csv, data/students.csv
Output: data/features.csv  (one row per student per week, ready for ML)

Key principle: We always compute features RELATIVE to the student's own
baseline (Week 1–2 average). This makes the model robust to the fact
that people have different natural baselines — a naturally introverted
student with a social_score of 5 is not declining; a previously social
student dropping from 8 to 5 IS.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading raw data...")
    daily   = pd.read_csv("data/daily_behavior.csv", parse_dates=["date"])
    mood    = pd.read_csv("data/weekly_mood.csv")
    students = pd.read_csv("data/students.csv")
    labels  = pd.read_csv("data/labels.csv")

    print(f"  ✅ daily_behavior: {len(daily):,} rows")
    print(f"  ✅ weekly_mood:    {len(mood):,} rows")
    print(f"  ✅ students:       {len(students):,} rows")
    print(f"  ✅ labels:         {len(labels):,} rows")
    return daily, mood, students, labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — WEEKLY AGGREGATION
# Collapse 7 daily rows into 1 weekly summary per student
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_weekly(daily_df):
    """
    Aggregate daily signals into weekly stats.
    For each signal we compute: mean, std, min, max
    This captures both the average level AND the variability within the week.
    """
    print("\n[Step 1] Aggregating daily → weekly...")

    signals = [
        "sleep_hours", "bedtime_hour", "lms_logins",
        "study_hours", "social_score", "dining_visits", "assignment_delta"
    ]

    agg_funcs = {sig: ["mean", "std", "min", "max"] for sig in signals}

    # Also count late-night logins (bedtime > 25 = past 1am)
    daily_df["late_night"] = (daily_df["bedtime_hour"] > 25.0).astype(int)
    agg_funcs["late_night"] = "sum"

    # Count days with zero LMS logins (complete disengagement days)
    daily_df["zero_lms_day"] = (daily_df["lms_logins"] == 0).astype(int)
    agg_funcs["zero_lms_day"] = "sum"

    weekly = daily_df.groupby(["student_id", "week"]).agg(agg_funcs).reset_index()

    # Flatten multi-level column names
    weekly.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in weekly.columns
    ]

    print(f"  ✅ Weekly aggregation: {len(weekly):,} rows "
          f"({weekly['student_id'].nunique()} students × {weekly['week'].nunique()} weeks)")
    return weekly


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PERSONAL BASELINE COMPUTATION
# Each student's own Week 1-2 average becomes their personal reference point
# ══════════════════════════════════════════════════════════════════════════════

def compute_personal_baselines(weekly_df):
    """
    Compute each student's personal baseline from their first 2 weeks.

    Why weeks 1-2? This is before any academic pressure or mental health
    decline kicks in. It's the student's natural resting behavioral state.

    Every subsequent feature is computed as DEVIATION from this baseline.
    This is the key insight: decline is relative to self, not to population.
    """
    print("\n[Step 2] Computing personal baselines (Weeks 1-2)...")

    baseline_signals = [
        "sleep_hours_mean", "bedtime_hour_mean", "lms_logins_mean",
        "study_hours_mean", "social_score_mean", "dining_visits_mean",
        "assignment_delta_mean"
    ]

    # Use only weeks 1-2 for baseline
    baseline_data = weekly_df[weekly_df["week"] <= 2].groupby("student_id")[
        baseline_signals
    ].mean().reset_index()

    # Rename to make clear these are baselines
    baseline_data.columns = (
        ["student_id"] +
        [f"baseline_{col}" for col in baseline_signals]
    )

    print(f"  ✅ Baselines computed for {len(baseline_data)} students")
    return baseline_data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DEVIATION FEATURES
# How much has each student drifted from their own Week 1-2 baseline?
# ══════════════════════════════════════════════════════════════════════════════

def compute_deviation_features(weekly_df, baseline_df):
    """
    For each signal, compute:
      deviation = current_week_value - personal_baseline

    Negative deviation in sleep = sleeping less than your normal
    Negative deviation in lms_logins = logging in less than your normal
    Positive deviation in bedtime = going to bed later than your normal

    These relative features are far more predictive than raw values.
    """
    print("\n[Step 3] Computing deviation from personal baseline...")

    df = weekly_df.merge(baseline_df, on="student_id", how="left")

    signal_pairs = [
        ("sleep_hours_mean",      "baseline_sleep_hours_mean"),
        ("bedtime_hour_mean",     "baseline_bedtime_hour_mean"),
        ("lms_logins_mean",       "baseline_lms_logins_mean"),
        ("study_hours_mean",      "baseline_study_hours_mean"),
        ("social_score_mean",     "baseline_social_score_mean"),
        ("dining_visits_mean",    "baseline_dining_visits_mean"),
        ("assignment_delta_mean", "baseline_assignment_delta_mean"),
    ]

    for current_col, baseline_col in signal_pairs:
        signal_name = current_col.replace("_mean", "").replace("baseline_", "")
        df[f"dev_{signal_name}"] = df[current_col] - df[baseline_col]

    print(f"  ✅ Deviation features computed: {len([c for c in df.columns if c.startswith('dev_')])} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ROLLING WINDOW FEATURES
# 2-week and 4-week rolling averages to capture trends
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_features(df):
    """
    Rolling averages smooth out week-to-week noise and reveal sustained trends.

    A single bad week (exam stress) looks very different from
    3 consecutive bad weeks (mental health decline).

    We compute rolling means over 2-week and 4-week windows for key signals.
    """
    print("\n[Step 4] Computing rolling window features...")

    roll_signals = [
        "sleep_hours_mean", "lms_logins_mean",
        "social_score_mean", "assignment_delta_mean", "bedtime_hour_mean"
    ]

    df = df.sort_values(["student_id", "week"])

    for signal in roll_signals:
        short_name = signal.replace("_mean", "")
        # 2-week rolling mean
        df[f"roll2_{short_name}"] = (
            df.groupby("student_id")[signal]
            .transform(lambda x: x.rolling(2, min_periods=1).mean())
        )
        # 4-week rolling mean
        df[f"roll4_{short_name}"] = (
            df.groupby("student_id")[signal]
            .transform(lambda x: x.rolling(4, min_periods=1).mean())
        )
        # Rolling standard deviation (captures instability)
        df[f"rollstd4_{short_name}"] = (
            df.groupby("student_id")[signal]
            .transform(lambda x: x.rolling(4, min_periods=2).std())
        )

    n_roll = len([c for c in df.columns if c.startswith("roll")])
    print(f"  ✅ Rolling features computed: {n_roll} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SLOPE FEATURES
# Is the trend getting better or worse? Rate of change matters.
# ══════════════════════════════════════════════════════════════════════════════

def compute_slope_features(df):
    """
    Slope = rate of change over the past 4 weeks.

    A student with sleep_hours = 5.5 might be:
      - Recovering (slope = +0.3/week → getting better)
      - Deteriorating (slope = -0.4/week → getting worse)

    The absolute value alone doesn't tell the story. Direction does.
    We use linear regression slope over a 4-week rolling window.
    """
    print("\n[Step 5] Computing slope (rate of change) features...")

    slope_signals = [
        "sleep_hours_mean", "lms_logins_mean",
        "social_score_mean", "assignment_delta_mean"
    ]

    df = df.sort_values(["student_id", "week"])

    def rolling_slope(series, window=4):
        """Compute linear regression slope over rolling window."""
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i - window + 1 : i + 1].values
                x = np.arange(window)
                if len(y) >= 2 and not np.all(np.isnan(y)):
                    try:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        slopes.append(slope)
                    except Exception:
                        slopes.append(np.nan)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)

    for signal in slope_signals:
        short_name = signal.replace("_mean", "")
        df[f"slope_{short_name}"] = (
            df.groupby("student_id")[signal]
            .transform(lambda x: rolling_slope(x))
        )

    n_slopes = len([c for c in df.columns if c.startswith("slope_")])
    print(f"  ✅ Slope features computed: {n_slopes} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — COMPOSITE BEHAVIORAL DRIFT SCORE
# A single 0-100 score summarizing overall behavioral deterioration
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift_score(df):
    """
    Behavioral Drift Score: a single 0-100 metric summarizing how much
    a student has drifted from their healthy baseline across ALL signals.

    Weights reflect clinical importance:
      - Sleep & social withdrawal are strongest predictors → higher weight
      - LMS & study hours → medium weight
      - Dining & assignment timing → lower weight but still meaningful

    Score of 0   = no drift, student is at baseline
    Score of 50  = moderate drift, medium risk
    Score of 100 = severe drift, high risk
    """
    print("\n[Step 6] Computing composite behavioral drift score...")

    # Normalize each deviation to a 0-1 scale based on expected max deviation
    # (based on the max changes we built into the trajectory modifiers)
    components = {
        "dev_sleep_hours":      {"col": "dev_sleep_hours",      "max_dev": 4.0,  "direction": -1, "weight": 0.25},
        "dev_bedtime_hour":     {"col": "dev_bedtime_hour",     "max_dev": 5.0,  "direction":  1, "weight": 0.20},
        "dev_social_score":     {"col": "dev_social_score",     "max_dev": 7.0,  "direction": -1, "weight": 0.20},
        "dev_lms_logins":       {"col": "dev_lms_logins",       "max_dev": 4.0,  "direction": -1, "weight": 0.15},
        "dev_study_hours":      {"col": "dev_study_hours",      "max_dev": 3.5,  "direction": -1, "weight": 0.10},
        "dev_assignment_delta": {"col": "dev_assignment_delta", "max_dev": 8.0,  "direction": -1, "weight": 0.05},
        "dev_dining_visits":    {"col": "dev_dining_visits",    "max_dev": 2.5,  "direction": -1, "weight": 0.05},
    }

    drift_score = pd.Series(0.0, index=df.index)

    for name, config in components.items():
        col       = config["col"]
        max_dev   = config["max_dev"]
        direction = config["direction"]  # -1 = lower is bad, +1 = higher is bad
        weight    = config["weight"]

        # Compute normalized component (0 = no drift, 1 = max drift)
        raw_drift = df[col] * direction  # flip sign so "bad" direction = positive
        normalized = (raw_drift / max_dev).clip(0, 1)
        drift_score += normalized * weight

    # Scale to 0-100
    df["drift_score"] = (drift_score * 100).round(2)

    print(f"  ✅ Drift score computed")
    print(f"     Mean: {df['drift_score'].mean():.1f} | "
          f"Max: {df['drift_score'].max():.1f} | "
          f"Min: {df['drift_score'].min():.1f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MOOD FEATURES
# Integrate weekly mood check-ins with behavioral signals
# ══════════════════════════════════════════════════════════════════════════════

def compute_mood_features(df, mood_df):
    """
    Add mood-based features:
      - mood_score:       raw self-reported score (1-10)
      - mood_missing:     1 if student skipped check-in (itself a signal)
      - mood_slope:       is mood improving or declining?
      - mood_volatility:  large swings week-to-week suggest instability
    """
    print("\n[Step 7] Computing mood features...")

    mood_df = mood_df.copy()
    mood_df["mood_missing"] = mood_df["mood_score"].isna().astype(int)
    mood_df["mood_score"]   = mood_df["mood_score"].ffill()  # carry forward

    # Mood slope over 3 weeks
    mood_df = mood_df.sort_values(["student_id", "week"])
    mood_df["mood_slope"] = (
        mood_df.groupby("student_id")["mood_score"]
        .transform(lambda x: x.rolling(3, min_periods=2).apply(
            lambda y: stats.linregress(range(len(y)), y)[0] if len(y) >= 2 else np.nan
        ))
    )

    # Mood volatility (std over 3 weeks)
    mood_df["mood_volatility"] = (
        mood_df.groupby("student_id")["mood_score"]
        .transform(lambda x: x.rolling(3, min_periods=2).std())
    )

    # Merge into main feature df
    mood_features = mood_df[["student_id", "week", "mood_score",
                               "mood_missing", "mood_slope", "mood_volatility"]]
    df = df.merge(mood_features, on=["student_id", "week"], how="left")

    print(f"  ✅ Mood features added: mood_score, mood_missing, mood_slope, mood_volatility")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — STUDENT METADATA FEATURES
# Demographic risk factors from research literature
# ══════════════════════════════════════════════════════════════════════════════

def add_student_metadata(df, students_df):
    """
    Add student-level features that modulate risk:
      - has_history:  prior mental health history → higher baseline risk
      - first_gen:    first-generation students face additional stressors
      - scholarship:  financial pressure → risk factor
      - year_encoded: freshmen and seniors have higher risk periods
    """
    print("\n[Step 8] Adding student metadata features...")

    year_map = {
        "Freshman": 1, "Sophomore": 2, "Junior": 3,
        "Senior": 4, "Graduate": 5
    }

    meta = students_df[[
        "student_id", "age", "has_history",
        "first_gen", "scholarship", "year", "gpa_start"
    ]].copy()
    meta["year_encoded"] = meta["year"].map(year_map)
    meta["has_history"]  = meta["has_history"].astype(int)
    meta["first_gen"]    = meta["first_gen"].astype(int)
    meta["scholarship"]  = meta["scholarship"].astype(int)
    meta = meta.drop(columns=["year"])

    df = df.merge(meta, on="student_id", how="left")
    print(f"  ✅ Metadata features added: age, has_history, first_gen, scholarship, year_encoded, gpa_start")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — ATTACH LABELS & FINAL CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def attach_labels_and_clean(df, labels_df):
    """
    Attach ground truth risk labels and clean up the final feature set.
    Drop raw intermediate columns, handle NaN values, finalize column order.
    """
    print("\n[Step 9] Attaching labels and cleaning up...")

    df = df.merge(
        labels_df[["student_id", "week", "risk_label", "risk_name", "trajectory"]],
        on=["student_id", "week"],
        how="left"
    )

    # Drop baseline columns (used for computation, not for ML)
    drop_cols = [c for c in df.columns if c.startswith("baseline_")]
    df = df.drop(columns=drop_cols)

    # Fill NaN slope/rolling values with 0 (early weeks have no history)
    fill_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["slope_", "roll", "mood_slope", "mood_volatility"]
    )]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Final column ordering
    id_cols     = ["student_id", "week", "trajectory", "risk_label", "risk_name"]
    target_cols = ["drift_score", "mood_score", "mood_missing",
                   "mood_slope", "mood_volatility"]
    meta_cols   = ["age", "year_encoded", "has_history",
                   "first_gen", "scholarship", "gpa_start"]

    feature_cols = [c for c in df.columns
                    if c not in id_cols + target_cols + meta_cols
                    and not c.startswith("baseline_")]

    final_cols = id_cols + target_cols + meta_cols + sorted(feature_cols)
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Remove week 1-2 rows (insufficient history for meaningful features)
    df = df[df["week"] >= 3].reset_index(drop=True)

    print(f"  ✅ Final dataset: {len(df):,} rows × {len(df.columns)} features")
    print(f"     Students: {df['student_id'].nunique()}")
    print(f"     Weeks:    {df['week'].min()}–{df['week'].max()}")

    # Label distribution
    dist = df["risk_name"].value_counts()
    print(f"     Risk distribution → Low: {dist.get('Low',0):,} | "
          f"Medium: {dist.get('Medium',0):,} | "
          f"High: {dist.get('High',0):,}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_summary(df):
    """Print a human-readable summary of all features created."""
    print("\n" + "=" * 60)
    print("   📋 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    feature_groups = {
        "🎯 Target Variables":        ["risk_label", "risk_name"],
        "📊 Composite Score":          ["drift_score"],
        "😊 Mood Features":            [c for c in df.columns if "mood" in c],
        "📉 Deviation Features":       [c for c in df.columns if c.startswith("dev_")],
        "📈 Rolling Avg (2-week)":     [c for c in df.columns if c.startswith("roll2_")],
        "📈 Rolling Avg (4-week)":     [c for c in df.columns if c.startswith("roll4_")],
        "〰️  Rolling Std (4-week)":    [c for c in df.columns if c.startswith("rollstd")],
        "📐 Slope Features":           [c for c in df.columns if c.startswith("slope_")],
        "🌙 Late-Night Signals":       ["late_night_sum", "zero_lms_day_sum"],
        "👤 Student Metadata":         ["age", "year_encoded", "has_history",
                                        "first_gen", "scholarship", "gpa_start"],
    }

    total = 0
    for group, cols in feature_groups.items():
        existing = [c for c in cols if c in df.columns]
        if existing:
            print(f"\n  {group} ({len(existing)})")
            for c in existing:
                print(f"    • {c}")
            total += len(existing)

    print(f"\n  Total features: {total}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("   🧠 MindBridge — Module 2: Feature Engineering")
    print("=" * 60)

    # Load
    daily, mood, students, labels = load_data()

    # Pipeline
    weekly   = aggregate_weekly(daily)
    baseline = compute_personal_baselines(weekly)
    df       = compute_deviation_features(weekly, baseline)
    df       = compute_rolling_features(df)
    df       = compute_slope_features(df)
    df       = compute_drift_score(df)
    df       = compute_mood_features(df, mood)
    df       = add_student_metadata(df, students)
    df       = attach_labels_and_clean(df, labels)

    # Save
    df.to_csv("data/features.csv", index=False)
    print(f"\n✅ Saved → data/features.csv")

    # Summary
    print_feature_summary(df)

    print("\n✅ Module 2 complete. Ready for Module 3: ML Risk Classifier.")
