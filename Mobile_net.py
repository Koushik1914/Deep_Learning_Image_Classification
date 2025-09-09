# pretrained_cnn_asl_with_classical_classifiers.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef
)
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# ------------------ Config ------------------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

DATASET_PATH = r"C:\Users\vkous\Downloads\DL_PROECT\2\split\asl_dataset"  # <--- update this path
IMG_SIZE = (64, 64)
CHANNELS = 1  # 1 for grayscale (your dataset), 3 for RGB
MAX_PER_CLASS = 500
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1  # portion of the TRAIN split used for validation
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10  # Feature extraction (frozen backbone)
EPOCHS_STAGE2 = 10  # Fine-tuning (top layers unfrozen)
BACKBONE = "mobilenetv2"  # "mobilenetv2" or "efficientnetb0"

# For speed on CPU, MobileNetV2 works well at 160; EfficientNetB0 expects 224
BACKBONE_INPUT_SIZE = {"mobilenetv2": 160, "efficientnetb0": 224}

OUT_DIR = f"runs_pretrained_{BACKBONE}"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ IO & Preprocess ------------------
def safe_imread(p):
    try:
        return cv2.imread(str(p), cv2.IMREAD_COLOR)
    except Exception:
        return None

def preprocess(img_bgr, size=IMG_SIZE, channels=CHANNELS):
    # Grayscale/RGB -> resize -> normalize to [0,1]
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
            if cap and count >= cap: break
            img = safe_imread(f)
            if img is None: continue
            X.append(preprocess(img, img_size, channels))
            y.append(label); count += 1
        if count == 0:
            print(f"Warning: No valid images loaded for class '{cdir.name}'")
            continue
        class_names.append(cdir.name)
        print(f"Loaded {count} images for class '{cdir.name}'")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    if len(X) == 0: raise ValueError("No images loaded.")
    print(f"\nDataset Summary: {len(X)} images, {len(class_names)} classes, shape={X[0].shape}")
    return X, y, class_names

def add_noise(X, noise=0.1):
    Xn = X + noise * np.random.normal(0, 1, size=X.shape).astype(np.float32)
    return np.clip(Xn, 0., 1.)

# ------------------ Plots ------------------
def plot_and_save_history(hist1, hist2, save_to):
    # Concatenate stage-1 and stage-2 curves
    train_acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc   = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    train_loss = hist1.history['loss'] + hist2.history['loss']
    val_loss   = hist1.history['val_loss'] + hist2.history['val_loss']

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc); plt.plot(val_acc)
    plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch')
    plt.legend(['train','val'])

    plt.subplot(1,2,2)
    plt.plot(train_loss); plt.plot(val_loss)
    plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(['train','val'])
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()

def plot_confusion_matrix(cm, class_names, save_to, backbone):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({backbone})'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()

def save_sample_predictions(X, y_true, y_pred, class_names, save_to, n=12):
    n = min(n, len(X))
    idxs = np.random.choice(len(X), n, replace=False)
    cols = 6; rows = int(np.ceil(n/cols))
    plt.figure(figsize=(2.5*cols, 2.5*rows))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        if CHANNELS == 1: plt.imshow(X[idx].squeeze(), cmap='gray')
        else: plt.imshow(np.clip(X[idx],0,1))
        t = class_names[y_true[idx]]; p = class_names[y_pred[idx]]
        plt.title(f"T:{t}\nP:{p}", fontsize=8); plt.axis('off')
    plt.tight_layout(); plt.savefig(save_to, dpi=150); plt.close()

# ------------------ Pretrained model (with scaling fix) ------------------
def build_pretrained_model(num_classes, backbone="mobilenetv2",
                           input_shape=(64,64,1), dropout=0.3, label_smoothing=0.1):
    """
    Pipeline inside the model:
      1) Resize to backbone input (e.g., 160 or 224)
      2) 1x1 Conv to map grayscale->3 channels (learnable)
      3) Rescaling(255.0)   <-- convert 0–1 back to 0–255
      4) tf.keras.applications preprocess_input for the chosen backbone
      5) Frozen backbone (Stage 1) -> GAP -> Dropout -> Dense head
    """
    inp = layers.Input(shape=input_shape, name="input")
    input_size = BACKBONE_INPUT_SIZE.get(backbone.lower(), 224)
    x = layers.Resizing(input_size, input_size, interpolation="bilinear", name="resize")(inp)
    x = layers.Conv2D(3, 1, padding="same", use_bias=False, name="gray_to_rgb")(x)

    if backbone.lower() == "mobilenetv2":
        Base = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone.lower() == "efficientnetb0":
        Base = tf.keras.applications.EfficientNetB0
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError("Unsupported backbone. Use 'mobilenetv2' or 'efficientnetb0'.")

    # Scaling patch so preprocess_input sees 0–255 range
    x = layers.Rescaling(255.0, name="to_255")(x)
    x = layers.Lambda(preprocess, name="app_preprocess")(x)

    base = Base(include_top=False, input_shape=(input_size, input_size, 3), weights="imagenet")
    base.trainable = False  # Stage 1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(num_classes, activation="softmax", name="head")(x)

    model = models.Model(inp, out, name=f"pretrained_{backbone}")
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=loss, metrics=["accuracy"])
    return model, base

def train_pretrained_model(X_train, y_train, X_val, y_val, class_weights, num_classes,
                           backbone="mobilenetv2", out_dir="runs_pretrained"):
    os.makedirs(out_dir, exist_ok=True)
    model, base = build_pretrained_model(num_classes, backbone=backbone,
                                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS))

    # tf.data pipelines with light augmentation
    AUTOTUNE = tf.data.AUTOTUNE
    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(8192, seed=SEED)
    ds_train = ds_train.map(lambda i,l: (i,l), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Stage 1: frozen
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(out_dir, f"best_{backbone}_stage1.keras"),
                        monitor="val_loss", save_best_only=True, verbose=1),
        LearningRateScheduler(lambda epoch, lr: 3e-4 * (math.cos(epoch / EPOCHS_STAGE1 * math.pi) + 1) / 2 + 1e-6)
    ]
    print(f"\n[Stage 1] Training frozen {backbone} feature extractor...")
    hist1 = model.fit(
        ds_train, validation_data=ds_val,
        epochs=EPOCHS_STAGE1, callbacks=cb1, class_weight=class_weights, verbose=1
    )

    # Stage 2: fine-tune top 20% (skip BatchNorms)
    total = len(base.layers); unfreeze_from = int(total * 0.8)
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= unfreeze_from) and not isinstance(layer, layers.BatchNormalization)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])

    cb2 = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=5e-6, verbose=1),
        ModelCheckpoint(os.path.join(out_dir, f"best_{backbone}_stage2.keras"),
                        monitor="val_loss", save_best_only=True, verbose=1),
        LearningRateScheduler(lambda epoch, lr: 1e-4 * (math.cos(epoch / EPOCHS_STAGE2 * math.pi) + 1) / 2 + 5e-6)
    ]
    print(f"\n[Stage 2] Fine-tuning top layers of {backbone}...")
    hist2 = model.fit(
        ds_train, validation_data=ds_val,
        epochs=EPOCHS_STAGE2, callbacks=cb2, class_weight=class_weights, verbose=1
    )

    model.save(os.path.join(out_dir, f"final_{backbone}.keras"))
    model.save(os.path.join(out_dir, f"final_{backbone}.h5"))  # compatibility
    return model, hist1, hist2

# ------------------ Classical classifier helpers ------------------
def extract_features_from_pretrained(model, X, layer_name='gap'):
    """Extract features from the pretrained model's GAP output (or other named layer).
    Builds a small feature-extractor model mapping inputs->layer_output and returns numpy features."""
    if layer_name not in [l.name for l in model.layers]:
        # fallback to last GlobalAveragePooling2D or the dropout layer
        for l in reversed(model.layers):
            if isinstance(l, layers.GlobalAveragePooling2D) or l.name == 'drop':
                layer_name = l.name
                break
    feat_layer = model.get_layer(layer_name)
    feat_model = models.Model(inputs=model.input, outputs=feat_layer.output)
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
    plot_confusion_matrix(cm, class_names, os.path.join(OUT_DIR, f"cm_{prefix}.png"), BACKBONE)

    return {'acc': acc, 'prec_w': prec_w, 'rec_w': rec_w, 'f1_w': f1_w,
            'prec_m': prec_m, 'rec_m': rec_m, 'f1_m': f1_m, 'kappa': kappa}

# ------------------ Main ------------------
if __name__ == "__main__":
    try:
        # 1) Load & split
        print("Loading dataset...")
        X, y, class_names = load_dataset(DATASET_PATH)
        num_classes = len(class_names)
        print(f"Class distribution: {Counter(y)}")
        is_balanced = len(set(Counter(y).values())) == 1
        print(f"Dataset is {'balanced' if is_balanced else 'imbalanced'}")

        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
        )
        val_ratio_from_train = VAL_SPLIT / (1.0 - TEST_SPLIT)
        X_train, X_val, y_train_int, y_val_int = train_test_split(
            X_train, y_train_int, test_size=val_ratio_from_train,
            random_state=SEED, stratify=y_train_int
        )

        # 2) One-hot
        y_train = tf.keras.utils.to_categorical(y_train_int, num_classes)
        y_val   = tf.keras.utils.to_categorical(y_val_int, num_classes)
        y_test  = tf.keras.utils.to_categorical(y_test_int, num_classes)

        # 3) Class weights (handles slight imbalance)
        class_weights = None
        if not is_balanced:
            try:
                cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_int)
                class_weights = {i: float(cw[i]) for i in range(num_classes)}
                print("Class weights:", class_weights)
            except Exception as e:
                print("Could not compute class weights:", e)

        # 4) Train pretrained model
        model, hist1, hist2 = train_pretrained_model(
            X_train, y_train, X_val, y_val, class_weights,
            num_classes=num_classes, backbone=BACKBONE, out_dir=OUT_DIR
        )

        # 5) Save training curves
        plot_and_save_history(hist1, hist2, os.path.join(OUT_DIR, "training_curves.png"))

        # 6) Evaluate pretrained CNN on TEST
        print("\nEvaluating on TEST set...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_test_int

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)

        print("\n--- TEST METRICS (Pretrained) ---")
        print(f"Accuracy:              {acc:.4f}")
        print(f"Balanced Accuracy:     {bal_acc:.4f}")
        print(f"Matthews CorrCoef:     {mcc:.4f}")
        print(f"Precision (Macro):     {prec_m:.4f}   |  Weighted: {prec_w:.4f}")
        print(f"Recall   (Macro):      {rec_m:.4f}   |  Weighted: {rec_w:.4f}")
        print(f"F1-score (Macro):      {f1_m:.4f}   |  Weighted: {f1_w:.4f}")
        print(f"Cohen's Kappa:         {kappa:.4f}")

        try:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
            print(f"ROC-AUC (OvR):         {roc_auc:.4f}")
        except Exception as e:
            print("ROC-AUC calc failed:", e)

        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:\n", cm)
        plot_confusion_matrix(cm, class_names, os.path.join(OUT_DIR, "confusion_matrix.png"), BACKBONE)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        print("\nPer-class Accuracy:")
        for i, cname in enumerate(class_names):
            mask = (y_true == i)
            cls_acc = accuracy_score(y_true[mask], y_pred[mask]) if np.any(mask) else float('nan')
            print(f"  {cname}: {cls_acc:.4f}")

        # Robustness (Gaussian noise)
        base_acc = acc  # <-- define baseline accuracy
        print("\nRobustness under Gaussian noise:")
        print(f"{'Noise':<8} {'Accuracy':<10} {'Drop':<10}")
        print("-"*30)
        for sigma in [0.05, 0.10, 0.15]:
            Xn = add_noise(X_test, sigma)
            yn = np.argmax(model.predict(Xn, verbose=0), axis=1)
            acc_n = accuracy_score(y_true, yn)
            print(f"{sigma:<8.2f} {acc_n:<10.4f} {base_acc-acc_n:+.4f}")

        save_sample_predictions(X_test, y_true, y_pred, class_names,
                                os.path.join(OUT_DIR, "sample_predictions.png"))

        # ------------------ Classical classifiers on pretrained features ------------------
        print("\nExtracting features from pretrained model for classical classifiers...")
        # use the GAP output (name 'gap') as features
        X_train_f = extract_features_from_pretrained(model, X_train, layer_name='gap')
        X_val_f   = extract_features_from_pretrained(model, X_val, layer_name='gap')
        X_test_f  = extract_features_from_pretrained(model, X_test, layer_name='gap')

        print("Feature shapes:", X_train_f.shape, X_val_f.shape, X_test_f.shape)

        # scale features
        scaler = StandardScaler()
        X_train_f_s = scaler.fit_transform(X_train_f)
        X_val_f_s   = scaler.transform(X_val_f)
        X_test_f_s  = scaler.transform(X_test_f)

        classical_clfs = {
            'LogisticRegression': LogisticRegression(max_iter=2000, random_state=SEED),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(random_state=SEED),
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=SEED)
        }

        clf_results = {}
        for name, clf in classical_clfs.items():
            print(f"\nTraining classical classifier: {name}")
            if name in ['KNN']:
                model_pipe = make_pipeline(StandardScaler(), clf)
                model_pipe.fit(X_train_f, y_train_int)
                clf_final = model_pipe
                X_eval = X_test_f
            else:
                clf.fit(X_train_f_s, y_train_int)
                clf_final = clf
                X_eval = X_test_f_s

            res = evaluate_classical(clf_final, X_eval, y_test_int, class_names, prefix=name)
            clf_results[name] = res
            joblib.dump(clf_final, os.path.join(OUT_DIR, f"clf_{name}.joblib"))

        # Robustness under noise for classical classifiers
        print("\nRobustness under Gaussian noise (classical classifiers):")
        for sigma in [0.05, 0.10, 0.15]:
            print(f"\nNoise sigma={sigma}")
            Xn = add_noise(X_test, sigma)
            Xn_f = extract_features_from_pretrained(model, Xn, layer_name='gap')
            Xn_f_s = scaler.transform(Xn_f)
            for name, clf in classical_clfs.items():
                clf_loaded = joblib.load(os.path.join(OUT_DIR, f"clf_{name}.joblib"))
                if name in ['KNN']:
                    y_pred_n = clf_loaded.predict(Xn_f)
                else:
                    y_pred_n = clf_loaded.predict(Xn_f_s)
                acc_n = accuracy_score(y_test_int, y_pred_n)
                print(f"  {name:<12} acc={acc_n:.4f} drop={(clf_results[name]['acc']-acc_n):+.4f}")

        print(f"\nArtifacts saved to: {OUT_DIR}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
