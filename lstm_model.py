"""
MindBridge — Module 4: LSTM Time-Series Mood Trajectory Model
==============================================================
While XGBoost (Module 3) classifies risk at a single point in time,
the LSTM model learns the TRAJECTORY — the pattern of change over time.

Why two models?
  - XGBoost asks: "What does this week's snapshot look like?"
  - LSTM asks:    "Where is this student HEADING based on their history?"

A student with mood scores [7, 6, 5, 4, 3] is more alarming than
a student with mood scores [3, 4, 5, 6, 7] — even if their current
week looks identical. LSTM captures this directional momentum.

Architecture:
  Input:  Last 4 weeks of behavioral signals (sequence of 4 timesteps)
  Model:  2-layer LSTM → Dropout → Dense → Softmax
  Output: Risk probability for next week (Low / Medium / High)

This makes MindBridge PREDICTIVE, not just reactive —
it forecasts NEXT WEEK's risk from THIS WEEK's trajectory.

Input:  data/features.csv
Output:
  - models/lstm_model.pt          → trained PyTorch model
  - models/lstm_scaler.pkl        → feature scaler
  - data/lstm_predictions.csv     → per-student trajectory predictions
  - plots/lstm_training_curve.png → loss curve
  - plots/mood_trajectories.png   → sample student trajectories
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"Using device: {device}")

SEQUENCE_LEN = 4    # use last 4 weeks to predict next week
BATCH_SIZE   = 64
EPOCHS       = 60
LR           = 0.001
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_CLASSES  = 3


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & SELECT SEQUENCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path="data/features.csv"):
    print("=" * 60)
    print("   🧠 MindBridge — Module 4: LSTM Trajectory Model")
    print("=" * 60)
    print("\n[Step 1] Loading feature data...")

    df = pd.read_csv(path)
    df = df.sort_values(["student_id", "week"]).reset_index(drop=True)

    # Features that capture temporal dynamics best
    # These change meaningfully week-to-week and carry trajectory signal
    sequence_features = [
        "sleep_hours_mean",
        "bedtime_hour_mean",
        "lms_logins_mean",
        "study_hours_mean",
        "social_score_mean",
        "dining_visits_mean",
        "assignment_delta_mean",
        "mood_score",
        "drift_score",
        "dev_sleep_hours",
        "dev_social_score",
        "dev_lms_logins",
        "slope_sleep_hours",
        "slope_social_score",
        "slope_lms_logins",
        "mood_slope",
        "mood_volatility",
        "late_night_sum",
    ]

    # Keep only columns that exist in the dataframe
    sequence_features = [f for f in sequence_features if f in df.columns]

    print(f"  ✅ Loaded: {len(df):,} rows")
    print(f"  ✅ Sequence features: {len(sequence_features)}")
    print(f"  ✅ Students: {df['student_id'].nunique()}")
    print(f"  ✅ Weeks: {df['week'].min()}–{df['week'].max()}")

    return df, sequence_features


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD SEQUENCES
# Convert flat weekly rows into (sequence, label) pairs for LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(df, sequence_features, seq_len=SEQUENCE_LEN):
    """
    For each student, slide a window of seq_len weeks across their data.
    Each sequence predicts the NEXT week's risk label.

    Example for seq_len=4:
      Input:  weeks [3, 4, 5, 6]  → Predict: week 7 risk
      Input:  weeks [4, 5, 6, 7]  → Predict: week 8 risk
      Input:  weeks [5, 6, 7, 8]  → Predict: week 9 risk

    This teaches the model to recognize dangerous trajectories
    before they fully materialize.
    """
    print(f"\n[Step 2] Building sequences (window={seq_len} weeks → predict next week)...")

    X_sequences = []
    y_labels    = []
    meta        = []   # student_id, prediction_week for output table

    students = df["student_id"].unique()

    for sid in students:
        student_df = df[df["student_id"] == sid].sort_values("week")
        values     = student_df[sequence_features].values
        labels     = student_df["risk_label"].values
        weeks      = student_df["week"].values

        # Slide window: need seq_len history + 1 future label
        for i in range(len(values) - seq_len):
            seq   = values[i : i + seq_len]           # shape: (seq_len, n_features)
            label = labels[i + seq_len]                # next week's risk
            pred_week = weeks[i + seq_len]

            # Skip if any NaN in sequence
            if not np.isnan(seq).any():
                X_sequences.append(seq)
                y_labels.append(label)
                meta.append({"student_id": sid, "pred_week": pred_week})

    X = np.array(X_sequences, dtype=np.float32)   # (N, seq_len, n_features)
    y = np.array(y_labels,    dtype=np.int64)      # (N,)

    print(f"  ✅ Total sequences: {len(X):,}")
    print(f"  ✅ Sequence shape: {X.shape}  (samples × weeks × features)")
    print(f"  ✅ Label distribution:")
    for label, name in [(0,"Low"), (1,"Medium"), (2,"High")]:
        count = (y == label).sum()
        print(f"     {name}: {count:,} ({count/len(y)*100:.1f}%)")

    return X, y, meta


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SCALE FEATURES & SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def scale_and_split(X, y, meta, split_ratio=0.80):
    """
    Scale features and split into train/test.
    We use an 80/20 sequential split (not random) to preserve
    the temporal nature of the data.
    """
    print(f"\n[Step 3] Scaling features and splitting data...")

    split_idx = int(len(X) * split_ratio)

    X_train, X_test = X[:split_idx],    X[split_idx:]
    y_train, y_test = y[:split_idx],    y[split_idx:]
    meta_test       = meta[split_idx:]

    # Fit scaler on training data only (prevent data leakage)
    n_samples, seq_len, n_features = X_train.shape
    scaler = StandardScaler()

    # Reshape to 2D for scaler, then back to 3D
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_samples, seq_len, n_features)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(len(X_test), seq_len, n_features)

    # Save scaler for dashboard use
    with open("models/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"  ✅ Train: {len(X_train):,} sequences")
    print(f"  ✅ Test:  {len(X_test):,} sequences")
    print(f"  ✅ Scaler saved → models/lstm_scaler.pkl")

    return (X_train_scaled, X_test_scaled,
            y_train, y_test, meta_test, scaler)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PYTORCH DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

class StudentSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping our sequence arrays.
    DataLoader uses this to feed mini-batches to the LSTM during training.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(X_train, X_test, y_train, y_test):
    print(f"\n[Step 4] Building PyTorch DataLoaders...")

    train_dataset = StudentSequenceDataset(X_train, y_train)
    test_dataset  = StudentSequenceDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  ✅ Train batches: {len(train_loader)}")
    print(f"  ✅ Test batches:  {len(test_loader)}")

    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — LSTM MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class MindBridgeLSTM(nn.Module):
    """
    2-layer LSTM for mental health trajectory prediction.

    Architecture:
      Input (seq_len=4, n_features=18)
        ↓
      LSTM Layer 1 (hidden=128, dropout between layers)
        ↓
      LSTM Layer 2 (hidden=128)
        ↓
      Take last timestep output
        ↓
      Dropout (0.3) — prevents overfitting
        ↓
      Fully Connected Layer (128 → 64)
        ↓
      ReLU activation
        ↓
      Dropout (0.2)
        ↓
      Output Layer (64 → 3) — Low / Medium / High
        ↓
      Softmax → probabilities

    Why LSTM over simple RNN?
      LSTM has memory gates (forget, input, output) that let it
      selectively remember important past events and forget noise.
      A student's sleep pattern from 3 weeks ago matters more than
      a random fluctuation from last week — LSTM learns this.
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=DROPOUT):
        super(MindBridgeLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take output from last timestep only
        # Shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Pass through classifier
        logits = self.classifier(last_output)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TRAIN LSTM
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(model, train_loader, test_loader, y_train):
    """
    Training loop with:
      - CrossEntropyLoss with class weights (handles imbalance)
      - Adam optimizer (adaptive learning rate)
      - Learning rate scheduler (reduces LR when plateau detected)
      - Early stopping (stops if val loss doesn't improve for 10 epochs)
    """
    print(f"\n[Step 6] Training LSTM ({EPOCHS} epochs)...")

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_losses, val_losses     = [], []
    train_accs,   val_accs       = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds      = outputs.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

        train_loss = epoch_loss / len(train_loader)
        train_acc  = correct / total

        # ── Validation ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)
                val_loss_sum += loss.item()
                preds         = outputs.argmax(dim=1)
                val_correct  += (preds == y_batch).sum().item()
                val_total    += len(y_batch)

        val_loss = val_loss_sum / len(test_loader)
        val_acc  = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        # Progress print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/lstm_model_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # Load best weights
    model.load_state_dict(torch.load("models/lstm_model_best.pt",
                                      map_location=device))
    print(f"\n  ✅ Best val loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATE LSTM
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_lstm(model, test_loader):
    print(f"\n[Step 7] Evaluating LSTM on test set...")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs)

    y_pred  = np.array(all_preds)
    y_true  = np.array(all_labels)
    y_probs = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")

    print(f"  ✅ Accuracy:      {acc:.3f} ({acc*100:.1f}%)")
    print(f"  ✅ F1 (weighted): {f1_w:.3f}")
    print(f"  ✅ F1 (macro):    {f1_m:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["Low","Medium","High"],
                                 digits=3))

    return y_pred, y_true, y_probs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE PREDICTIONS & PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(y_pred, y_true, y_probs, meta_test):
    print(f"\n[Step 8] Saving LSTM predictions...")

    risk_names = {0: "Low", 1: "Medium", 2: "High"}
    n = min(len(y_pred), len(meta_test))

    predictions = pd.DataFrame({
        "student_id":     [m["student_id"] for m in meta_test[:n]],
        "pred_week":      [m["pred_week"]  for m in meta_test[:n]],
        "actual_risk":    y_true[:n],
        "predicted_risk": y_pred[:n],
        "actual_name":    [risk_names[r] for r in y_true[:n]],
        "predicted_name": [risk_names[r] for r in y_pred[:n]],
        "prob_low":       y_probs[:n, 0].round(3),
        "prob_medium":    y_probs[:n, 1].round(3),
        "prob_high":      y_probs[:n, 2].round(3),
        "correct":        (y_true[:n] == y_pred[:n]),
    })

    predictions.to_csv("data/lstm_predictions.csv", index=False)
    print(f"  ✅ Saved → data/lstm_predictions.csv ({len(predictions):,} rows)")
    return predictions


def save_plots(train_losses, val_losses, train_accs, val_accs, df):
    print(f"\n[Step 8b] Saving plots...")

    # ── Plot 1: Training Curve ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    epochs_ran = range(1, len(train_losses) + 1)

    ax1.plot(epochs_ran, train_losses, label="Train Loss",
             color="#1a4a2e", linewidth=2)
    ax1.plot(epochs_ran, val_losses,   label="Val Loss",
             color="#e74c3c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("LSTM Training & Validation Loss", fontweight="bold")
    ax1.legend(); ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(epochs_ran, train_accs, label="Train Acc",
             color="#1a4a2e", linewidth=2)
    ax2.plot(epochs_ran, val_accs,   label="Val Acc",
             color="#e74c3c", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("LSTM Training & Validation Accuracy", fontweight="bold")
    ax2.legend(); ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/lstm_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ plots/lstm_training_curve.png")

    # ── Plot 2: Sample Mood Trajectories ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    trajectories = ["healthy", "declining", "crisis"]
    colors_map   = {"healthy": "#2ecc71", "declining": "#f39c12", "crisis": "#e74c3c"}
    titles       = ["✅ Healthy Student", "⚠️ Declining Student", "🔴 Crisis Student"]

    for ax, traj, title in zip(axes, trajectories, titles):
        sample = df[df["trajectory"] == traj]["student_id"].iloc[0]
        student_data = df[df["student_id"] == sample].sort_values("week")

        weeks      = student_data["week"].values
        mood       = student_data["mood_score"].values
        drift      = student_data["drift_score"].values

        ax2_twin = ax.twinx()
        ax.plot(weeks, mood,  color=colors_map[traj],
                linewidth=2.5, marker="o", markersize=4, label="Mood Score")
        ax2_twin.plot(weeks, drift, color="gray",
                      linewidth=1.5, linestyle="--", alpha=0.7, label="Drift Score")

        ax.set_xlabel("Week"); ax.set_ylabel("Mood Score (1-10)", color=colors_map[traj])
        ax2_twin.set_ylabel("Drift Score (0-100)", color="gray")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 11)
        ax2_twin.set_ylim(0, 105)
        ax.spines[["top"]].set_visible(False)
        ax.axvline(x=8, color="#888", linestyle=":", alpha=0.5)
        ax.text(8.2, 1.5, "Mid-sem", fontsize=8, color="#888")

    plt.suptitle("MindBridge — Student Trajectory Comparison\n(Mood Score + Behavioral Drift)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/mood_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ plots/mood_trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, sequence_features):
    torch.save({
        "model_state_dict": model.state_dict(),
        "sequence_features": sequence_features,
        "hidden_size": HIDDEN_SIZE,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "num_classes": NUM_CLASSES,
        "sequence_len": SEQUENCE_LEN,
    }, "models/lstm_model.pt")
    print(f"\n  ✅ models/lstm_model.pt saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load
    df, sequence_features = load_data()

    # Build sequences
    X, y, meta = build_sequences(df, sequence_features)

    # Scale & split
    (X_train, X_test,
     y_train, y_test,
     meta_test, scaler) = scale_and_split(X, y, meta)

    # DataLoaders
    train_loader, test_loader = build_dataloaders(
        X_train, X_test, y_train, y_test
    )

    # Build model
    print(f"\n[Step 5] Building LSTM model...")
    n_features = X_train.shape[2]
    model = MindBridgeLSTM(input_size=n_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Model architecture:")
    print(f"     Input:   (batch, {SEQUENCE_LEN} weeks, {n_features} features)")
    print(f"     LSTM:    {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden units")
    print(f"     Output:  3 classes (Low / Medium / High)")
    print(f"     Params:  {total_params:,}")

    # Train
    model, train_losses, val_losses, train_accs, val_accs = train_lstm(
        model, train_loader, test_loader, y_train
    )

    # Evaluate
    y_pred, y_true, y_probs = evaluate_lstm(model, test_loader)

    # Save predictions
    predictions = save_predictions(y_pred, y_true, y_probs, meta_test)

    # Save plots
    save_plots(train_losses, val_losses, train_accs, val_accs, df)

    # Save model
    save_model(model, sequence_features)

    # Final summary
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("   ✅ Module 4 Complete!")
    print("=" * 60)
    print(f"   Accuracy:      {acc*100:.1f}%")
    print(f"   F1 (weighted): {f1_w:.3f}")
    print()
    print("   📁 Outputs:")
    print("   models/lstm_model.pt           → trained LSTM")
    print("   models/lstm_scaler.pkl         → feature scaler")
    print("   data/lstm_predictions.csv      → trajectory predictions")
    print("   plots/lstm_training_curve.png  → loss & accuracy curves")
    print("   plots/mood_trajectories.png    → student trajectory chart")
    print()
    print("   ✅ Ready for Module 5: AI Counselor Chatbot")
    print("=" * 60)