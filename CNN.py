# simple_cnn_asl_with_classical_classifiers.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ------------------ Config ------------------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

DATASET_PATH = r"C:\Users\vkous\Downloads\DL_PROECT\2\split\asl_dataset"  # <-- update
IMG_SIZE = (64, 64)
CHANNELS = 1
MAX_PER_CLASS = 500
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1               # taken from train portion
BATCH_SIZE = 32
EPOCHS = 30
OUT_DIR = "runs_simple_cnn"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Utils ------------------
def safe_imread(p):
    try:
        return cv2.imread(str(p), cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess(img_bgr, size=IMG_SIZE, channels=CHANNELS):
    if channels == 1:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if channels == 1:
        img = np.expand_dims(img, -1)
    return img


def load_dataset(dataset_path, img_size=IMG_SIZE, channels=CHANNELS, cap=MAX_PER_CLASS):
    X, y, class_names = [], [], []
    dpath = Path(dataset_path)
    if not dpath.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist!")

    class_dirs = [d for d in sorted(dpath.iterdir()) if d.is_dir()]
    for label, cdir in enumerate(class_dirs):
        count = 0
        for f in cdir.iterdir():
            if f.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tiff"} or f.name.startswith("."):
                continue
            if cap and count >= cap:
                break
            img = safe_imread(f)
            if img is None:
                continue
            X.append(preprocess(img, img_size, channels))
            y.append(label)
            count += 1
        print(f"Loaded {count} images for class '{cdir.name}'")
        if count > 0:
            class_names.append(cdir.name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    if len(X) == 0:
        raise ValueError("No images loaded.")
    print(f"\nDataset Summary: {len(X)} images, {len(class_names)} classes, shape={X[0].shape}")
    return X, y, class_names


def add_noise(X, noise=0.1):
    Xn = X + noise * np.random.normal(0, 1, size=X.shape).astype(np.float32)
    return np.clip(Xn, 0., 1.)

# ------------------ Simple CNN ------------------
def build_simple_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='feat_dense'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------ Plots ------------------
def plot_history(hist, save_to):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(['train','val'])
    plt.subplot(1,2,2); plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss'])
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(['train','val'])
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()


def plot_confusion_matrix(cm, class_names, save_to, title='Confusion Matrix'):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()


def save_sample_predictions(X, y_true, y_pred, class_names, save_to, n=12):
    n = min(n, len(X))
    idxs = np.random.choice(len(X), n, replace=False)
    cols = 6
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(2.5*cols, 2.5*rows))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        if CHANNELS == 1:
            plt.imshow(X[idx].squeeze(), cmap='gray')
        else:
            plt.imshow(np.clip(X[idx],0,1))
        t = class_names[y_true[idx]]; p = class_names[y_pred[idx]]
        plt.title(f"T:{t} | P:{p}", fontsize=9); plt.axis('off')
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()

# ------------------ Classical classifier helpers ------------------
def extract_features_from_model(model, X, input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    """
    Robust feature extraction that works even if the Sequential model has not been called.
    We build a new functional feature-extractor by feeding a fresh Input through the layers
    up to the chosen feature layer (named 'feat_dense' by default). This avoids relying on
    `model.input`/`model.inputs` which may be undefined in some contexts.
    """
    # 1) Find the feature layer object by name or fallback to the last Dense before output
    feat_layer = None
    if any(l.name == 'feat_dense' for l in model.layers):
        feat_layer = model.get_layer('feat_dense')
    else:
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Dense) and l.output_shape and l.output_shape[-1] > 1:
                feat_layer = l
                break
    if feat_layer is None:
        raise RuntimeError("Could not find a suitable feature layer (expected 'feat_dense' or a Dense layer).")

    # 2) Create a fresh Input and run it through the model's layers until feat_layer
    inp = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = inp
    for layer in model.layers:
        # If a layer is already built to a different input shape, calling it on x will wire it into the new graph
        x = layer(x)
        if layer is feat_layer:
            break

    feat_model = tf.keras.Model(inputs=inp, outputs=x)

    # 3) Predict features
    feats = feat_model.predict(X, verbose=0)
    return feats


def evaluate_classical(clf, X_test_f, y_test_int, class_names, prefix):
    y_pred = clf.predict(X_test_f)
    y_proba = None
    if hasattr(clf, 'predict_proba'):
        try:
            y_proba = clf.predict_proba(X_test_f)
        except Exception:
            y_proba = None

    acc  = accuracy_score(y_test_int, y_pred)
    prec_w = precision_score(y_test_int, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_test_int, y_pred, average='weighted', zero_division=0)
    f1_w   = f1_score(y_test_int, y_pred, average='weighted', zero_division=0)
    prec_m = precision_score(y_test_int, y_pred, average='macro', zero_division=0)
    rec_m  = recall_score(y_test_int, y_pred, average='macro', zero_division=0)
    f1_m   = f1_score(y_test_int, y_pred, average='macro', zero_division=0)
    kappa  = cohen_kappa_score(y_test_int, y_pred)

    print(f"\n--- {prefix} ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision (Macro): {prec_m:.4f}   |  Weighted: {prec_w:.4f}")
    print(f"Recall (Macro):    {rec_m:.4f}   |  Weighted: {rec_w:.4f}")
    print(f"F1-score (Macro):  {f1_m:.4f}   |  Weighted: {f1_w:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")

    # ROC-AUC if probabilities exist
    if y_proba is not None:
        try:
            y_test_bin = label_binarize(y_test_int, classes=np.arange(len(class_names)))
            roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
            print(f"ROC-AUC (OvR):     {roc_auc:.4f}")
        except Exception as e:
            print("ROC-AUC calc failed:", e)

    cm = confusion_matrix(y_test_int, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test_int, y_pred, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, os.path.join(OUT_DIR, f"cm_{prefix}.png"), title=f"CM - {prefix}")

    return {
        'acc': acc, 'prec_w': prec_w, 'rec_w': rec_w, 'f1_w': f1_w,
        'prec_m': prec_m, 'rec_m': rec_m, 'f1_m': f1_m, 'kappa': kappa
    }

# ------------------ Main ------------------
if __name__ == "__main__":
    # 1) Load & preprocess
    X, y, class_names = load_dataset(DATASET_PATH)
    num_classes = len(class_names)

    # 2) Split: train / test, then carve val from train
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    val_ratio_from_train = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train, y_train_int, test_size=val_ratio_from_train,
        random_state=SEED, stratify=y_train_int
    )

    # 3) One-hot for categorical crossentropy
    y_train = to_categorical(y_train_int, num_classes)
    y_val   = to_categorical(y_val_int, num_classes)
    y_test  = to_categorical(y_test_int, num_classes)

    # Optional: class weights if (slightly) imbalanced
    class_weights = None
    try:
        cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_int)
        class_weights = {i: float(cw[i]) for i in range(num_classes)}
        print("Class weights:", class_weights)
    except Exception as e:
        print("Could not compute class weights:", e)

    # 4) Build & train simple CNN
    model = build_simple_cnn((IMG_SIZE[0], IMG_SIZE[1], CHANNELS), num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(OUT_DIR, "best_simple_cnn.keras"), monitor='val_loss',
                        save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    plot_history(history, os.path.join(OUT_DIR, "training_curves.png"))

    # 5) Evaluate on TEST with CNN
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test_int

    acc  = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_m  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa  = cohen_kappa_score(y_true, y_pred)

    print("\n--- TEST METRICS (Simple CNN) ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision (Macro): {prec_m:.4f}   |  Weighted: {prec_w:.4f}")
    print(f"Recall (Macro):    {rec_m:.4f}   |  Weighted: {rec_w:.4f}")
    print(f"F1-score (Macro):  {f1_m:.4f}   |  Weighted: {f1_w:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")

    # ROC-AUC (OvR multiclass)
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        print(f"ROC-AUC (OvR):     {roc_auc:.4f}")
    except Exception as e:
        print("ROC-AUC calc failed:", e)

    # Confusion matrix & per-class analysis
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    plot_confusion_matrix(cm, class_names, os.path.join(OUT_DIR, "confusion_matrix.png"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\nPer-class Accuracy:")
    for i, cname in enumerate(class_names):
        mask = (y_true == i)
        cls_acc = accuracy_score(y_true[mask], y_pred[mask]) if np.any(mask) else float('nan')
        print(f"  {cname}: {cls_acc:.4f}")

    # 6) Classical classifiers: extract features from the trained CNN
    print("\nExtracting features from CNN for classical classifiers...")
    X_train_f = extract_features_from_model(model, X_train)
    X_val_f   = extract_features_from_model(model, X_val)
    X_test_f  = extract_features_from_model(model, X_test)

    print("Feature shapes:", X_train_f.shape, X_val_f.shape, X_test_f.shape)

    # scale features
    scaler = StandardScaler()
    X_train_f_s = scaler.fit_transform(X_train_f)
    X_val_f_s   = scaler.transform(X_val_f)
    X_test_f_s  = scaler.transform(X_test_f)

    # classifiers to try
    classical_clfs = {
        'LogisticRegression': LogisticRegression(multi_class='ovr', max_iter=2000, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=SEED)
    }

    clf_results = {}
    for name, clf in classical_clfs.items():
        print(f"\nTraining classical classifier: {name}")
        # wrap with scaler for safety where needed (we already scaled, but pipeline is fine)
        if name == 'KNN' or name == 'LogisticRegression':
            # these benefit from scaling
            model_pipe = make_pipeline(StandardScaler(), clf)
            model_pipe.fit(X_train_f, y_train_int)
            clf_final = model_pipe
        else:
            clf.fit(X_train_f_s, y_train_int)
            clf_final = clf

        # evaluate
        res = evaluate_classical(clf_final, X_test_f if name in ['LogisticRegression','KNN'] else X_test_f_s, y_test_int, class_names, prefix=name)
        clf_results[name] = res

        # save model
        joblib.dump(clf_final, os.path.join(OUT_DIR, f"clf_{name}.joblib"))

    # 7) Robustness under Gaussian noise for classical classifiers
    print("\nRobustness under Gaussian noise (classical classifiers):")
    for sigma in [0.05, 0.10, 0.15]:
        print(f"\nNoise sigma={sigma}")
        Xn = add_noise(X_test, sigma)
        Xn_f = extract_features_from_model(model, Xn)
        Xn_f_s = scaler.transform(Xn_f)
        for name, clf in classical_clfs.items():
            # load saved clf
            clf_loaded = joblib.load(os.path.join(OUT_DIR, f"clf_{name}.joblib"))
            if name in ['LogisticRegression','KNN']:
                y_pred_n = clf_loaded.predict(Xn_f)
            else:
                y_pred_n = clf_loaded.predict(Xn_f_s)
            acc_n = accuracy_score(y_test_int, y_pred_n)
            print(f"  {name:<12} acc={acc_n:.4f} drop={(clf_results[name]['acc']-acc_n):+.4f}")

    # 8) Save a small grid of CNN predictions
    save_sample_predictions(X_test, y_true, y_pred, class_names, os.path.join(OUT_DIR, "sample_predictions.png"))

    # 9) Save models
    model.save(os.path.join(OUT_DIR, "final_simple_cnn.keras"))
    print(f"\nArtifacts saved to: {OUT_DIR}")
