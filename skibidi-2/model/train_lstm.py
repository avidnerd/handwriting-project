"""
Train a 2-LSTM letter classifier on IMU gesture data.
Architecture from paper: LSTM(20) -> LSTM(25) -> Dense(25) -> Dense(n_classes)

Differences from train.py:
  - Normalization: per-channel min-max to [-1, 1] instead of z-score
  - Augmentation:  sliding-window averaging + Gaussian noise (paper method)
  - Model:         stacked LSTM with hard sigmoid instead of 1D-CNN

Dataset: CSV files named <LETTER>_<N>.csv, each row = timestamp,ax,ay,az,gx,gy,gz
Output:  imu_classifier_lstm.keras
         imu_classifier_lstm.tflite
         model_data_lstm.h
         labels.txt
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "."
LETTERS    = ["G", "I", "N", "S"]
SEQ_LEN    = 100
N_FEATURES = 6
EPOCHS     = 200
BATCH_SIZE = 8
AUG_FACTOR = 20
N_FOLDS    = 5
SEED       = 42

# Gesture extraction thresholds — must match realtime.py
MOTION_ONSET_THRESH  = 1.5
MOTION_OFFSET_THRESH = 0.5
MOTION_STILL_FRAMES  = 45

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_gesture(seq: np.ndarray) -> np.ndarray:
    """Trim a full recording to just the motion segment (identical to train.py)."""
    accel_mag = np.sqrt((seq[:, :3] ** 2).sum(axis=1))
    gyro_mag  = np.sqrt((seq[:, 3:] ** 2).sum(axis=1))
    win      = min(50, len(accel_mag) // 4)
    baseline = float(accel_mag[:win].mean())
    motion   = np.abs(accel_mag - baseline) + 3.0 * gyro_mag

    onset = None
    for i in range(len(motion)):
        if motion[i] > MOTION_ONSET_THRESH:
            onset = max(0, i - 5)
            break
    if onset is None:
        return seq

    still_count = 0
    offset = len(seq)
    for i in range(onset + 20, len(motion)):
        if motion[i] < MOTION_OFFSET_THRESH:
            still_count += 1
            if still_count >= MOTION_STILL_FRAMES:
                offset = max(onset + 20, i - MOTION_STILL_FRAMES + 10)
                break
        else:
            still_count = 0

    extracted = seq[onset:offset]
    return extracted if len(extracted) >= 20 else seq


def load_csv(path: str):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            try:
                rows.append([float(x) for x in parts[1:]])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float32) if len(rows) > 10 else None


def pad_or_truncate(seq: np.ndarray, length: int) -> np.ndarray:
    if len(seq) >= length:
        return seq[:length]
    pad = np.zeros((length - len(seq), seq.shape[1]), dtype=np.float32)
    return np.vstack([seq, pad])


def normalize(seq: np.ndarray) -> np.ndarray:
    """Per-sample, per-channel min-max scaling to [-1, 1]."""
    min_val = seq.min(axis=0, keepdims=True)
    max_val = seq.max(axis=0, keepdims=True)
    return 2.0 * (seq - min_val) / ((max_val - min_val) + 1e-8) - 1.0


# ── Augmentation ──────────────────────────────────────────────────────────────

def window_average(seq: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slide a window of size N with stride M, replacing each step with the mean.

    Simulates slightly slower/smoother writing. Output length varies with
    window_size and stride, so the caller must pad/truncate afterwards.
    """
    out = [
        seq[i : i + window_size].mean(axis=0)
        for i in range(0, len(seq) - window_size + 1, stride)
    ]
    return np.array(out, dtype=np.float32) if out else seq.copy()


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """Paper augmentation: random window averaging followed by Gaussian noise."""
    win    = np.random.randint(3, 9)   # window size N  (3–8 frames)
    stride = np.random.randint(1, 3)   # stride M       (1–2 frames)
    s = window_average(seq, win, stride)
    s += np.random.normal(0, 0.05, s.shape).astype(np.float32)
    return pad_or_truncate(s, SEQ_LEN)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int) -> tf.keras.Model:
    """2-layer LSTM from the paper.

    Hard sigmoid used for both cell and recurrent activations —
    computationally cheaper and consistent with paper spec.
    """
    inp = tf.keras.Input(shape=(SEQ_LEN, N_FEATURES))

    x = tf.keras.layers.LSTM(
        20,
        activation="hard_sigmoid",
        recurrent_activation="hard_sigmoid",
        return_sequences=True,
    )(inp)
    x = tf.keras.layers.LSTM(
        25,
        activation="hard_sigmoid",
        recurrent_activation="hard_sigmoid",
        return_sequences=False,
    )(x)

    x   = tf.keras.layers.Dense(25, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Load raw data ─────────────────────────────────────────────────────────────

X_raw, y_raw = [], []
for letter in LETTERS:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{letter}_*.csv")))
    for path in files:
        seq = load_csv(path)
        if seq is None:
            print(f"  SKIP {path} (too few valid rows)")
            continue
        seq = extract_gesture(seq)
        seq = normalize(seq)
        seq = pad_or_truncate(seq, SEQ_LEN)
        X_raw.append(seq)
        y_raw.append(letter)

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw)

le    = LabelEncoder()
y_enc = le.fit_transform(y_raw).astype(np.int32)

counts = {l: int((y_raw == l).sum()) for l in LETTERS}
print(f"Loaded {len(X_raw)} samples — {counts}")


# ── Cross-validation (augment only train folds) ───────────────────────────────

print(f"\n=== {N_FOLDS}-fold CV ===")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_accs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y_enc)):
    Xtr = list(X_raw[tr_idx])
    ytr = list(y_enc[tr_idx])
    for seq, lbl in zip(X_raw[tr_idx], y_enc[tr_idx]):
        for _ in range(AUG_FACTOR):
            Xtr.append(augment_sequence(seq))
            ytr.append(lbl)
    Xtr, ytr = np.array(Xtr, dtype=np.float32), np.array(ytr, dtype=np.int32)
    Xval, yval = X_raw[val_idx], y_enc[val_idx]

    model = build_model(len(LETTERS))
    cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(Xval, yval), callbacks=[cb], verbose=0)

    _, acc = model.evaluate(Xval, yval, verbose=0)
    preds  = model.predict(Xval, verbose=0).argmax(axis=1)
    fold_accs.append(acc)
    print(f"\nFold {fold + 1}  val_acc={acc:.3f}")
    print(classification_report(yval, preds, target_names=le.classes_, zero_division=0))

print(f"\nMean CV accuracy: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")


# ── Train final model on ALL data (with augmentation) ────────────────────────

print("\n=== Training final model ===")
X_all = list(X_raw)
y_all = list(y_enc)
for seq, lbl in zip(X_raw, y_enc):
    for _ in range(AUG_FACTOR):
        X_all.append(augment_sequence(seq))
        y_all.append(lbl)
X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)

perm  = np.random.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]

final_model = build_model(len(LETTERS))
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=15, factor=0.5, verbose=1)
final_model.fit(
    X_all, y_all,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[cb_lr],
    verbose=1,
)

preds_all = final_model.predict(X_raw, verbose=0).argmax(axis=1)
print("\n=== Final model on original (unaugmented) samples ===")
print(classification_report(y_enc, preds_all, target_names=le.classes_, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_enc, preds_all))


# ── Save ──────────────────────────────────────────────────────────────────────

final_model.save("imu_classifier_lstm.keras")
print("\nSaved imu_classifier_lstm.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open("imu_classifier_lstm.tflite", "wb") as fh:
    fh.write(tflite_bytes)
print(f"Saved imu_classifier_lstm.tflite  ({len(tflite_bytes):,} bytes)")

hex_values = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
with open("model_data_lstm.h", "w") as fh:
    fh.write("// Auto-generated by train_lstm.py — do not edit manually.\n")
    fh.write("#pragma once\n\n")
    fh.write(f"const unsigned int MODEL_DATA_LEN = {len(tflite_bytes)};\n")
    fh.write("alignas(8) const unsigned char MODEL_DATA[] = {\n  ")
    fh.write(hex_values)
    fh.write("\n};\n")
print("Saved model_data_lstm.h")

with open("labels.txt", "w") as fh:
    for i, label in enumerate(le.classes_):
        fh.write(f"{i},{label}\n")
print(f"Saved labels.txt  (order: {list(le.classes_)})")
print("\nDone.")
