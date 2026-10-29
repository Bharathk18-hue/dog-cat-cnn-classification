# %% [markdown]
# # Dog vs Cat Classification with a 3-Conv-Layer CNN
# Label convention used everywhere: **0 = Cat, 1 = Dog**. The sigmoid output is P(Dog).

# %% [markdown]
# ## 1. Imports and settings
# (Colab users: run `!pip install -q tensorflow scikit-learn pillow matplotlib` first if anything is missing.)

# %%
import hashlib
import json
import os
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 30                                     # EarlyStopping usually stops sooner
# Images used per class. By default ALL images in dataset/cats and dataset/dogs are used.
# For a quick test set the PER_CLASS environment variable (for example 500).
PER_CLASS = int(os.environ["PER_CLASS"]) if os.environ.get("PER_CLASS") else None
CLASS_NAMES = ["Cat", "Dog"]                        # index 0 = Cat, index 1 = Dog
EXTENSIONS = {".jpg", ".jpeg", ".png"}

try:
    ROOT = Path(__file__).resolve().parent.parent   # running as a script (src/train_model.py)
except NameError:                                   # running inside a notebook
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "dog_cat_cnn.keras"
OUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def save_fig(name):
    """Save the current figure into outputs/ and show it when running in a notebook."""
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150)
    if matplotlib.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


def load_and_resize(path):
    """Open an image, force RGB, resize to 128x128. Returns a uint8 array. Used for training AND prediction."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)

# %% [markdown]
# ## 2. Collect image paths, remove corrupted files and exact duplicates
# Duplicates are found by hashing the file bytes (MD5). Removing them *before* splitting
# guarantees the same image can never appear in two different splits (no data leakage).

# %%
def collect_files(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}\nSee README.md for how to set up the dataset.")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


paths, labels, seen_hashes = [], [], set()
skipped_corrupt = skipped_duplicate = 0

for label, sub in enumerate(["cats", "dogs"]):
    kept = 0
    for p in collect_files(DATA_DIR / sub):
        if PER_CLASS is not None and kept >= PER_CLASS:
            break
        try:
            with Image.open(p) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            skipped_corrupt += 1
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(digest)
        paths.append(str(p))
        labels.append(label)
        kept += 1

labels = np.array(labels)
print(f"Images kept: {len(paths)}  (cats={int((labels == 0).sum())}, dogs={int((labels == 1).sum())})")
print(f"Skipped corrupted: {skipped_corrupt} | Skipped duplicates: {skipped_duplicate}")
if len(paths) < 20 or len(set(labels)) < 2:
    raise RuntimeError("Not enough images found. Put images in dataset/cats and dataset/dogs (see README.md).")

# %% [markdown]
# ## 3. Split 70% train / 15% validation / 15% test
# We split the *file list* (not pixels). `stratify` keeps the cat/dog ratio equal in every split and
# `random_state=SEED` makes the split reproducible.

# %%
p_train, p_temp, y_train, y_temp = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
p_val, p_test, y_val, y_test = train_test_split(
    p_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

assert not (set(p_train) & set(p_val)) and not (set(p_train) & set(p_test)) and not (set(p_val) & set(p_test))
print(f"Train={len(p_train)}  Validation={len(p_val)}  Test={len(p_test)}  (no overlap between splits)")

# %% [markdown]
# ## 4. Load images (resize to 128x128) and normalize to 0-1
# Images are stored as small uint8 arrays to save memory and divided by 255 batch by batch.
# **Data augmentation (random flips, small rotations, zooms and shifts) is applied to the TRAINING set only.**
# Validation and test images are never changed, so the test score stays fair.

# %%
def load_array(path_list):
    return np.stack([load_and_resize(p) for p in path_list])


X_train, X_val, X_test = load_array(p_train), load_array(p_val), load_array(p_test)
print("Array shapes:", X_train.shape, X_val.shape, X_test.shape)


augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.15, seed=SEED),
    layers.RandomTranslation(0.1, 0.1, seed=SEED),
], name="augmentation")


def make_dataset(X, y, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype("float32")))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, t: (tf.cast(x, tf.float32) / 255.0, t))  # normalize to 0-1
    if augment:                                                     # training data only
        ds = ds.map(lambda x, t: (augmenter(x, training=True), t), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train, y_train, shuffle=True, augment=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)          # not shuffled, so predictions line up with y_test

# Safety check: validation and test images must NOT be augmented (only normalized).
for name, ds_, X_ in [("validation", val_ds, X_val), ("test", test_ds, X_test)]:
    batch = next(iter(ds_))[0].numpy()
    assert np.allclose(batch, X_[:len(batch)] / 255.0, atol=1e-6), f"{name} data was changed!"
print("Verified: validation and test images are not augmented.")

# %% [markdown]
# ## 5. Show sample images with labels

# %%
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_train), size=8, replace=False)
plt.figure(figsize=(12, 6))
for i, k in enumerate(idx):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_train[k])
    plt.title(f"{CLASS_NAMES[y_train[k]]} (label={y_train[k]})")
    plt.axis("off")
save_fig("sample_images.png")

# %% [markdown]
# ### Augmentation examples
# The first picture is the original; the others are random augmented versions the model sees during training.

# %%
original = X_train[idx[0]].astype("float32")[np.newaxis] / 255.0
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    shown = original if i == 0 else augmenter(original, training=True).numpy()
    plt.imshow(np.clip(shown[0], 0, 1))
    plt.title("Original" if i == 0 else f"Augmented {i}")
    plt.axis("off")
save_fig("augmentation_examples.png")

# %% [markdown]
# ## 6. Build the CNN (exactly three Conv2D layers)

# %%
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu", name="conv3"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),       # P(Dog); 0 = Cat, 1 = Dog
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

n_conv = sum(isinstance(l, layers.Conv2D) for l in model.layers)
assert n_conv == 3, f"Expected exactly 3 Conv2D layers, found {n_conv}"
print(f"Verified: the model has exactly {n_conv} Conv2D layers.")
with open(OUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# %% [markdown]
# ## 7. Train
# * **EarlyStopping** stops when validation loss stops improving for 6 epochs and restores the best weights.
# * **ReduceLROnPlateau** halves the learning rate when validation loss stops improving for 2 epochs.
# * **ModelCheckpoint** saves the best model (lowest validation loss) to `dog_cat_cnn.keras`.

# %%
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, callbacks=callbacks,
                    shuffle=False, verbose=2)  # data is already shuffled inside train_ds
print(f"Best model saved to: {MODEL_PATH}")

# %% [markdown]
# ## 8. Training graphs

# %%
for metric, title in [("accuracy", "Accuracy"), ("loss", "Loss")]:
    plt.figure(figsize=(6, 4))
    plt.plot(history.history[metric], label=f"Training {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"Training and Validation {title}")
    plt.legend()
    save_fig(f"{metric}_graph.png")

# %% [markdown]
# ## 9. Evaluate on the test set (all numbers come from the real model)

# %%
best_model = tf.keras.models.load_model(MODEL_PATH)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
probs = best_model.predict(test_ds, verbose=0).ravel()      # P(Dog)
y_pred = (probs > 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary", pos_label=1, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)

print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Precision (Dog as positive class): {precision:.4f}")
print(f"Recall    (Dog as positive class): {recall:.4f}")
print(f"F1-score  (Dog as positive class): {f1:.4f}")
print("\nConfusion matrix (rows = true, columns = predicted):\n", cm)
print("\nClassification report:\n", report)

ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
save_fig("confusion_matrix.png")

metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc), "precision": float(precision),
           "recall": float(recall), "f1_score": float(f1), "confusion_matrix": cm.tolist(),
           "train_images": len(p_train), "val_images": len(p_val), "test_images": len(p_test),
           "epochs_run": len(history.history["loss"])}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT_DIR / "classification_report.txt").write_text(report)
print("Saved metrics, report and graphs to:", OUT_DIR)

# %% [markdown]
# ## 10. Prediction on a new image

# %%
def predict_image(image_path, model_to_use=None, show=True):
    """Return (class_name, confidence) for one image, or None if the file cannot be used."""
    model_to_use = model_to_use or best_model
    try:
        arr = load_and_resize(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except (UnidentifiedImageError, OSError):
        print(f"Unsupported or corrupted image: {image_path}")
        return None
    p_dog = float(model_to_use.predict(arr[np.newaxis] / 255.0, verbose=0)[0][0])
    name = CLASS_NAMES[int(p_dog > 0.5)]
    confidence = p_dog if p_dog > 0.5 else 1 - p_dog
    if show:
        plt.figure(figsize=(3, 3))
        plt.imshow(arr)
        plt.title(f"{name} ({confidence * 100:.1f}%)")
        plt.axis("off")
        save_fig(f"prediction_{name.lower()}_{Path(image_path).stem}.png")
    print(f"{Path(image_path).name}: {name} with {confidence * 100:.1f}% confidence")
    return name, confidence


sample_cat = next(p for p, y in zip(p_test, y_test) if y == 0)
sample_dog = next(p for p, y in zip(p_test, y_test) if y == 1)
predict_image(sample_cat)
predict_image(sample_dog)
predict_image("this/path/does/not/exist.jpg")        # shows graceful error handling

# %% [markdown]
# ## 11. Reload the saved model and predict again

# %%
reloaded = tf.keras.models.load_model(MODEL_PATH)
print("Reloaded model from:", MODEL_PATH)
predict_image(sample_cat, reloaded)
predict_image(sample_dog, reloaded)
