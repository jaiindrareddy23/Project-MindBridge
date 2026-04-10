"""
MindBridge — Module 3: ML Risk Classifier
==========================================
Trains an XGBoost classifier on engineered features to predict
student mental health risk levels: Low (0), Medium (1), High (2).

What this module does:
  1. Loads features.csv from Module 2
  2. Splits data into train/test sets (time-aware split)
  3. Handles class imbalance with SMOTE
  4. Trains XGBoost classifier with cross-validation
  5. Evaluates model performance (accuracy, F1, confusion matrix)
  6. Generates SHAP explainability values
  7. Saves model + results for use in the dashboard (Module 6)

Key design decisions:
  - Time-aware split: train on weeks 3-12, test on weeks 13-16
    (simulates real deployment — model must predict future weeks)
  - SHAP explainability: model explains WHY each student is flagged
    (critical for counselor trust and ethical AI)
  - Per-student prediction: outputs risk score for every student
    every week, not just a global accuracy number

Input:  data/features.csv
Output:
  - models/xgboost_model.pkl     → trained model
  - data/predictions.csv         → per-student weekly risk predictions
  - data/shap_values.csv         → SHAP feature importances
  - results/evaluation_report.txt → model performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(path="data/features.csv"):
    print("=" * 60)
    print("   🧠 MindBridge — Module 3: ML Risk Classifier")
    print("=" * 60)
    print("\n[Step 1] Loading feature data...")

    df = pd.read_csv(path)
    print(f"  ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Drop non-feature columns
    drop_cols = ["student_id", "trajectory", "risk_name"]
    meta_cols  = ["week"]  # keep week for time-aware split

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + ["risk_label"] + meta_cols
    ]

    X = df[feature_cols].copy()
    y = df["risk_label"].copy()
    weeks = df["week"].copy()
    student_ids = df["student_id"].copy() if "student_id" in df.columns else None

    # Reload with student_id for predictions output
    df_full = pd.read_csv(path)

    # Fill any remaining NaNs
    X = X.fillna(0)

    print(f"  ✅ Features: {len(feature_cols)}")
    print(f"  ✅ Target distribution:")
    for label, name in [(0,"Low"), (1,"Medium"), (2,"High")]:
        count = (y == label).sum()
        pct   = count / len(y) * 100
        print(f"     {name}: {count:,} ({pct:.1f}%)")

    return X, y, weeks, df_full, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TIME-AWARE TRAIN/TEST SPLIT
# Train on early weeks, test on later weeks — simulates real deployment
# ══════════════════════════════════════════════════════════════════════════════

def time_aware_split(X, y, weeks, df_full, split_week=12):
    """
    Split by week rather than randomly.

    Why? In real deployment, the model trains on historical data
    and predicts future weeks. A random split would leak future
    information into training — making results unrealistically good.

    Train: weeks 3-12  (early/mid semester)
    Test:  weeks 13-16 (late semester — when crisis typically peaks)
    """
    print(f"\n[Step 2] Time-aware train/test split (split at week {split_week})...")

    train_mask = weeks <= split_week
    test_mask  = weeks >  split_week

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    df_test = df_full[test_mask.values].copy()

    print(f"  ✅ Train: {len(X_train):,} rows (weeks 3–{split_week})")
    print(f"  ✅ Test:  {len(X_test):,} rows  (weeks {split_week+1}–16)")
    print(f"  ✅ Test risk distribution:")
    for label, name in [(0,"Low"), (1,"Medium"), (2,"High")]:
        count = (y_test == label).sum()
        print(f"     {name}: {count:,}")

    return X_train, X_test, y_train, y_test, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — HANDLE CLASS IMBALANCE
# Use sample weights so the model doesn't ignore minority classes
# ══════════════════════════════════════════════════════════════════════════════

def handle_imbalance(X_train, y_train):
    """
    Class imbalance problem: most students are Low risk (majority class).
    Without correction, the model learns to always predict Low and gets
    decent accuracy while missing all actual at-risk students — the worst
    possible outcome for a mental health system.

    Solution: compute_sample_weight gives higher weight to rare classes
    so the model pays more attention to Medium and High risk students.
    """
    print("\n[Step 3] Handling class imbalance with sample weights...")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    class_counts = pd.Series(y_train).value_counts().sort_index()
    for label, name in [(0,"Low"), (1,"Medium"), (2,"High")]:
        count  = class_counts.get(label, 0)
        weight = sample_weights[y_train == label].mean() if count > 0 else 0
        print(f"  {name}: {count:,} samples → weight {weight:.2f}")

    return sample_weights


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN XGBOOST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, sample_weights):
    """
    XGBoost is chosen because:
      1. Handles tabular data better than neural networks at this scale
      2. Naturally handles missing values
      3. Built-in feature importance
      4. Works well with SHAP explainability
      5. Industry standard for structured data in healthcare analytics

    Hyperparameters are tuned for this specific problem:
      - n_estimators=300: enough trees for stable predictions
      - max_depth=5: deep enough to capture interactions, not overfit
      - learning_rate=0.05: slow learning = better generalization
      - subsample=0.8: use 80% of data per tree (reduces overfitting)
      - colsample_bytree=0.8: use 80% of features per tree
    """
    print("\n[Step 4] Training XGBoost classifier...")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Cross-validation on training set (5-fold stratified)
    print("  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="f1_weighted",
        params={"sample_weight": sample_weights}
    )
    print(f"  ✅ CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final training on full training set
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        verbose=False
    )
    print(f"  ✅ Model trained on {len(X_train):,} samples")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test):
    """Evaluate on held-out test set (weeks 13-16)."""
    print("\n[Step 5] Evaluating on test set (weeks 13–16)...")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # AUC-ROC (one-vs-rest for multiclass)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        auc = roc_auc_score(y_bin, y_pred_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    print(f"  ✅ Accuracy:       {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  ✅ F1 (weighted):  {f1_w:.3f}")
    print(f"  ✅ F1 (macro):     {f1_macro:.3f}")
    print(f"  ✅ AUC-ROC:        {auc:.3f}")

    print("\n  Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=["Low", "Medium", "High"],
        digits=3
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print("             Low   Med   High")
    labels = ["Low  ", "Med  ", "High "]
    for i, row in enumerate(cm):
        print(f"  Actual {labels[i]}: {row}")

    metrics = {
        "accuracy": accuracy,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "auc_roc": auc,
        "report": report,
        "confusion_matrix": cm,
    }

    return y_pred, y_pred_prob, metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SHAP EXPLAINABILITY
# WHY did the model flag this student?
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_train, X_test, feature_cols):
    """
    SHAP (SHapley Additive exPlanations) answers the question:
    'Which features contributed most to THIS prediction?'

    For a counselor, knowing a student is 'High Risk' is not enough.
    They need to know: WHY? What changed?

    SHAP gives us statements like:
      'drift_score contributed +0.42 to the High risk prediction'
      'sleep_hours_mean contributed +0.31 to the High risk prediction'
      'social_score_mean contributed +0.28 to the High risk prediction'

    This makes the AI explainable, trustworthy, and actionable.
    """
    print("\n[Step 6] Computing SHAP explainability values...")
    print("  (This may take 30-60 seconds...)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values shape: [n_classes, n_samples, n_features]
    # We focus on class 2 (High risk) — most important for intervention
    if isinstance(shap_values, list):
        shap_high = shap_values[2]   # High risk class
    else:
        shap_high = shap_values[:, :, 2]

    # Mean absolute SHAP value per feature (global importance)
    mean_shap = np.abs(shap_high).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": mean_shap
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  🔍 Top 15 Most Important Features (for High Risk prediction):")
    print(f"  {'Feature':<35} {'SHAP Importance':>15}")
    print(f"  {'-'*35} {'-'*15}")
    for _, row in shap_df.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<35} {row['importance']:>8.4f}  {bar}")

    # Save SHAP values
    shap_df.to_csv("data/shap_values.csv", index=False)
    print(f"\n  ✅ SHAP values saved → data/shap_values.csv")

    return shap_df, shap_high, explainer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — GENERATE PREDICTIONS TABLE
# Per-student, per-week risk predictions with confidence scores
# ══════════════════════════════════════════════════════════════════════════════

def generate_predictions(model, X_test, y_test, y_pred, y_pred_prob, df_test):
    """
    Create a predictions table that the dashboard (Module 6) will use
    to show counselors which students are at risk this week.
    """
    print("\n[Step 7] Generating per-student predictions table...")

    risk_names = {0: "Low", 1: "Medium", 2: "High"}

    predictions = pd.DataFrame({
        "student_id":       df_test["student_id"].values if "student_id" in df_test.columns else [f"STU{i}" for i in range(len(y_test))],
        "week":             df_test["week"].values if "week" in df_test.columns else [0]*len(y_test),
        "actual_risk":      y_test.values,
        "predicted_risk":   y_pred,
        "actual_name":      [risk_names[r] for r in y_test.values],
        "predicted_name":   [risk_names[r] for r in y_pred],
        "prob_low":         y_pred_prob[:, 0].round(3),
        "prob_medium":      y_pred_prob[:, 1].round(3),
        "prob_high":        y_pred_prob[:, 2].round(3),
        "correct":          (y_test.values == y_pred),
        "drift_score":      df_test["drift_score"].values if "drift_score" in df_test.columns else [0]*len(y_test),
        "mood_score":       df_test["mood_score"].values if "mood_score" in df_test.columns else [0]*len(y_test),
    })

    predictions.to_csv("data/predictions.csv", index=False)

    # Summary
    high_risk   = predictions[predictions["predicted_risk"] == 2]
    medium_risk = predictions[predictions["predicted_risk"] == 1]

    print(f"  ✅ Predictions saved → data/predictions.csv")
    print(f"  📊 Students flagged HIGH risk:   {high_risk['student_id'].nunique()}")
    print(f"  📊 Students flagged MEDIUM risk: {medium_risk['student_id'].nunique()}")

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_plots(metrics, shap_df, predictions):
    """Save evaluation plots as PNG files for the dashboard."""
    print("\n[Step 8] Saving evaluation plots...")

    # ── Plot 1: Feature Importance (SHAP) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    top15 = shap_df.head(15)
    bars = ax.barh(
        top15["feature"][::-1],
        top15["importance"][::-1],
        color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, 15))
    )
    ax.set_xlabel("Mean |SHAP Value| — Contribution to High Risk Prediction", fontsize=11)
    ax.set_title("🔍 Top 15 Most Important Features\n(MindBridge Risk Classifier)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ plots/shap_importance.png")

    # ── Plot 2: Confusion Matrix ────────────────────────────────────────────
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Low","Medium","High"])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Low","Medium","High"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i,j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ plots/confusion_matrix.png")

    # ── Plot 3: Risk Distribution ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    names  = ["Low", "Medium", "High"]

    for i, (ax, col, title) in enumerate(zip(
        axes,
        ["actual_risk", "predicted_risk"],
        ["Actual Risk Distribution", "Predicted Risk Distribution"]
    )):
        counts = predictions[col].value_counts().sort_index()
        ax.bar(names, [counts.get(j, 0) for j in range(3)], color=colors)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Students")
        ax.spines[["top", "right"]].set_visible(False)
        for j, v in enumerate([counts.get(k, 0) for k in range(3)]):
            ax.text(j, v + 5, str(v), ha="center", fontsize=11)

    plt.suptitle("MindBridge Risk Classification Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/risk_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ plots/risk_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE MODEL & REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_model_and_report(model, feature_cols, metrics):
    """Save trained model and evaluation report."""
    print("\n[Step 9] Saving model and evaluation report...")

    # Save model
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print("  ✅ models/xgboost_model.pkl")

    # Save report
    report_text = f"""
MindBridge — ML Risk Classifier Evaluation Report
==================================================
Model:        XGBoost (multi:softprob, 3 classes)
Train period: Weeks 3–12
Test period:  Weeks 13–16

PERFORMANCE METRICS
-------------------
Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
F1 (weighted):  {metrics['f1_weighted']:.3f}
F1 (macro):     {metrics['f1_macro']:.3f}
AUC-ROC:        {metrics['auc_roc']:.3f}

CLASSIFICATION REPORT
---------------------
{metrics['report']}

CONFUSION MATRIX
----------------
             Low   Med   High
"""
    cm = metrics["confusion_matrix"]
    for i, name in enumerate(["Low  ", "Med  ", "High "]):
        report_text += f"Actual {name}: {cm[i].tolist()}\n"

    with open("results/evaluation_report.txt", "w") as f:
        f.write(report_text)
    print("  ✅ results/evaluation_report.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load
    X, y, weeks, df_full, feature_cols = load_and_prepare()

    # Split
    X_train, X_test, y_train, y_test, df_test = time_aware_split(
        X, y, weeks, df_full
    )

    # Imbalance
    sample_weights = handle_imbalance(X_train, y_train)

    # Train
    model = train_model(X_train, y_train, sample_weights)

    # Evaluate
    y_pred, y_pred_prob, metrics = evaluate_model(model, X_test, y_test)

    # SHAP
    shap_df, shap_high, explainer = compute_shap(
        model, X_train, X_test, feature_cols
    )

    # Predictions table
    predictions = generate_predictions(
        model, X_test, y_test, y_pred, y_pred_prob, df_test
    )

    # Plots
    save_plots(metrics, shap_df, predictions)

    # Save
    save_model_and_report(model, feature_cols, metrics)

    # Final summary
    print("\n" + "=" * 60)
    print("   ✅ Module 3 Complete!")
    print("=" * 60)
    print(f"   Accuracy:      {metrics['accuracy']*100:.1f}%")
    print(f"   F1 (weighted): {metrics['f1_weighted']:.3f}")
    print(f"   AUC-ROC:       {metrics['auc_roc']:.3f}")
    print()
    print("   📁 Outputs:")
    print("   models/xgboost_model.pkl     → trained model")
    print("   data/predictions.csv         → per-student predictions")
    print("   data/shap_values.csv         → feature importances")
    print("   plots/shap_importance.png    → SHAP bar chart")
    print("   plots/confusion_matrix.png   → confusion matrix")
    print("   plots/risk_distribution.png  → risk distribution")
    print("   results/evaluation_report.txt → full metrics")
    print()
    print("   ✅ Ready for Module 4: LSTM Time-Series Model")
    print("=" * 60)
