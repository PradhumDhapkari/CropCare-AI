"""
Image preprocessing utilities for CropCare AI.
Handles loading, augmentation, normalization, and dataset splitting.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, DATASET_DIR, CLASS_LABELS_PATH


# ─── Augmentation pipeline ────────────────────────────────────────────────────

def build_augmentation_pipeline() -> tf.keras.Sequential:
    """Data augmentation layers applied only during training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name='augmentation')


# ─── Single image preprocessing ───────────────────────────────────────────────

def preprocess_image_for_prediction(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for model prediction.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3).

    Raises:
        ValueError: If image cannot be opened or is not RGB-compatible.
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # MobileNetV2 normalization
        return img_array
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file: {image_path}")
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


# ─── Dataset discovery ────────────────────────────────────────────────────────

def discover_classes(dataset_dir: str = None) -> Dict[int, str]:
    """
    Discover class labels from dataset folder structure.
    Expects: dataset_dir/ClassName/image.jpg

    Returns:
        Dict mapping {index: class_name}.
    """
    dataset_dir = dataset_dir or DATASET_DIR
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Please download the PlantVillage dataset and place it at:\n"
            f"  {dataset_dir}\n"
            "See README.md for download instructions."
        )

    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not classes:
        raise ValueError(f"No class subdirectories found in {dataset_dir}")

    class_map = {idx: name for idx, name in enumerate(classes)}
    return class_map


def save_class_labels(class_map: Dict[int, str], path: str = None) -> None:
    """Save class labels dict to JSON."""
    path = path or CLASS_LABELS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(class_map, f, indent=2)
    print(f"[INFO] Class labels saved to: {path}")


def load_class_labels(path: str = None) -> Dict[int, str]:
    """Load class labels from JSON."""
    path = path or CLASS_LABELS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class labels file not found: {path}")
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─── Dataset builder ──────────────────────────────────────────────────────────

def collect_image_paths(dataset_dir: str, class_map: Dict[int, str]) -> List[Tuple[str, int]]:
    """
    Walk dataset directory and collect (image_path, class_index) tuples.
    Skips corrupted or unreadable images.
    """
    name_to_idx = {v: k for k, v in class_map.items()}
    items = []
    skipped = 0

    for class_name, idx in name_to_idx.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                # Quick validity check
                with Image.open(fpath) as img:
                    img.verify()
                items.append((fpath, idx))
            except Exception:
                skipped += 1

    print(f"[INFO] Collected {len(items)} images, skipped {skipped} corrupted files.")
    return items


def split_dataset(
    items: List[Tuple[str, int]],
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = 42
) -> Tuple[list, list, list]:
    """
    Split dataset into train/val/test without data leakage.
    Uses stratified split by class.
    """
    random.seed(seed)
    from collections import defaultdict

    class_items = defaultdict(list)
    for item in items:
        class_items[item[1]].append(item)

    train, val, test = [], [], []
    for cls_items in class_items.values():
        random.shuffle(cls_items)
        n = len(cls_items)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        train.extend(cls_items[:n_train])
        val.extend(cls_items[n_train:n_train + n_val])
        test.extend(cls_items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"[INFO] Split: {len(train)} train | {len(val)} val | {len(test)} test")
    return train, val, test


# ─── tf.data pipeline ─────────────────────────────────────────────────────────

def load_and_preprocess_tf(path: str, label: int, num_classes: int) -> Tuple:
    """Load image and one-hot encode label for tf.data pipeline."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label_oh = tf.one_hot(label, num_classes)
    return img, label_oh


def build_tf_dataset(
    items: List[Tuple[str, int]],
    num_classes: int,
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from (path, label) pairs.

    Args:
        items: List of (image_path, class_index) tuples.
        num_classes: Total number of classes.
        batch_size: Batch size.
        augment: Whether to apply data augmentation.
        shuffle: Whether to shuffle data.

    Returns:
        Configured tf.data.Dataset.
    """
    paths = [item[0] for item in items]
    labels = [item[1] for item in items]

    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=min(len(items), 2000), seed=42)

    ds = path_ds.map(
        lambda p, l: load_and_preprocess_tf(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        aug = build_augmentation_pipeline()
        ds = ds.map(
            lambda img, lbl: (aug(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
