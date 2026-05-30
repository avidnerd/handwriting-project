"""
Train a 1D-CNN letter classifier on IMU gesture data.

Dataset: CSV files named <LETTER>_<N>.csv, each row = timestamp,ax,ay,az,gx,gy,gz
Output:  imu_classifier.keras  (full model)
         imu_classifier.tflite (quantized, for Python inference)
         model_data.h          (C byte array for Arduino TFLite)
         labels.txt            (index → letter mapping)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "."
LETTERS     = ["G", "I", "N", "S"]
SEQ_LEN     = 300        # pad / truncate all sequences to this
N_FEATURES  = 6          # ax, ay, az, gx, gy, gz (timestamp dropped)
EPOCHS      = 200
BATCH_SIZE  = 8
AUG_FACTOR  = 20         # extra augmented copies per original sample
N_FOLDS     = 5
SEED        = 42

# Gesture extraction thresholds — must match realtime.py
MOTION_ONSET_THRESH  = 1.5   # accel_dev + 3*gyro_mag to declare gesture start
MOTION_OFFSET_THRESH = 0.5   # combined motion below this = stillness
MOTION_STILL_FRAMES  = 45    # consecutive still frames to end gesture

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_gesture(seq: np.ndarray) -> np.ndarray:
    """Trim a full recording to just the motion segment.

    Uses the same combined accel+gyro motion metric as realtime.py so that
    training and inference see identically-structured inputs.
    """
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
        return seq  # no clear gesture found; keep as-is

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
                rows.append([float(x) for x in parts[1:]])  # drop timestamp
            except ValueError:
                continue
    return np.array(rows, dtype=np.float32) if len(rows) > 10 else None


def pad_or_truncate(seq: np.ndarray, length: int) -> np.ndarray:
    if len(seq) >= length:
        return seq[:length]
    pad = np.zeros((length - len(seq), seq.shape[1]), dtype=np.float32)
    return np.vstack([seq, pad])


def normalize(seq: np.ndarray) -> np.ndarray:
    """Per-sample, per-channel z-score normalization."""
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0,  keepdims=True) + 1e-8
    return (seq - mean) / std


# ── Augmentation ──────────────────────────────────────────────────────────────

def random_rotation(seq: np.ndarray) -> np.ndarray:
    """Apply a small random 3-D rotation to accel and gyro axes (pen tilt simulation)."""
    angle = np.random.uniform(-0.26, 0.26)  # ±15 degrees
    axis  = np.random.randn(3).astype(np.float32)
    axis /= np.linalg.norm(axis)
    c, s_  = float(np.cos(angle)), float(np.sin(angle))
    K = np.array([
        [        0, -axis[2],  axis[1]],
        [ axis[2],         0, -axis[0]],
        [-axis[1],   axis[0],        0],
    ], dtype=np.float32)
    R = c * np.eye(3, dtype=np.float32) + s_ * K + (1 - c) * np.outer(axis, axis)
    out = seq.copy()
    out[:, :3] = seq[:, :3] @ R.T
    out[:, 3:] = seq[:, 3:] @ R.T
    return out


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """Return one randomly-perturbed copy of seq (already normalized + padded)."""
    s = seq.copy()

    # Time shift (circular roll so padding zeros don't pile up at start)
    shift = np.random.randint(-40, 40)
    s = np.roll(s, shift, axis=0)

    # Additive Gaussian noise
    s += np.random.normal(0, 0.04, s.shape).astype(np.float32)

    # Per-channel amplitude scaling
    scale = np.random.uniform(0.88, 1.12, (1, s.shape[1])).astype(np.float32)
    s *= scale

    # Time stretch / compress via linear interpolation
    factor   = np.random.uniform(0.82, 1.18)
    new_len  = max(int(len(s) * factor), 20)
    indices  = np.linspace(0, len(s) - 1, new_len)
    s = np.stack(
        [np.interp(indices, np.arange(len(s)), s[:, i]) for i in range(s.shape[1])],
        axis=1
    ).astype(np.float32)

    # Pen tilt: small random 3-D rotation of sensor axes
    s = random_rotation(s)

    return pad_or_truncate(s, SEQ_LEN)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(1e-3)
    inp = tf.keras.Input(shape=(SEQ_LEN, N_FEATURES))

    x = tf.keras.layers.Conv1D(16, 7, padding="same", kernel_regularizer=reg)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv1D(32, 5, padding="same", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
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


# ── Cross-validation (on original samples, augment only train folds) ──────────

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

final_model.save("imu_classifier.keras")
print("\nSaved imu_classifier.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open("imu_classifier.tflite", "wb") as fh:
    fh.write(tflite_bytes)
print(f"Saved imu_classifier.tflite  ({len(tflite_bytes):,} bytes)")

hex_values = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
with open("model_data.h", "w") as fh:
    fh.write("// Auto-generated by train.py — do not edit manually.\n")
    fh.write("#pragma once\n\n")
    fh.write(f"const unsigned int MODEL_DATA_LEN = {len(tflite_bytes)};\n")
    fh.write("alignas(8) const unsigned char MODEL_DATA[] = {\n  ")
    fh.write(hex_values)
    fh.write("\n};\n")
print("Saved model_data.h")

with open("labels.txt", "w") as fh:
    for i, label in enumerate(le.classes_):
        fh.write(f"{i},{label}\n")
print(f"Saved labels.txt  (order: {list(le.classes_)})")
print("\nDone.")
