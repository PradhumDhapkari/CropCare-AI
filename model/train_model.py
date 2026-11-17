"""
Training script for CropCare AI MobileNetV2 model.

Usage:
    python model/train_model.py

Expects PlantVillage dataset at: dataset/PlantVillage/
Each subdirectory = one class (e.g., Tomato___Early_blight/)
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_DIR, MODEL_PATH, CLASS_LABELS_PATH, GRAPHS_FOLDER, REPORTS_DIR,
    EPOCHS, FINE_TUNE_EPOCHS, BATCH_SIZE, IMG_SIZE
)
from model.mobilenet_model import build_model
from model.preprocessing import (
    discover_classes, save_class_labels, collect_image_paths,
    split_dataset, build_tf_dataset
)

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_history(history, fine_history=None, save_dir=GRAPHS_FOLDER):
    """Plot and save training accuracy and loss graphs."""
    os.makedirs(save_dir, exist_ok=True)

    # Merge histories if fine-tuning occurred
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if fine_history:
        acc += fine_history.history['accuracy']
        val_acc += fine_history.history['val_accuracy']
        loss += fine_history.history['loss']
        val_loss += fine_history.history['val_loss']

    epochs_range = range(len(acc))

    # Style
    plt.rcParams.update({'figure.facecolor': '#f8faf8', 'axes.facecolor': '#f0f4f0'})

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, acc, label='Training Accuracy', color='#2d7a2d', linewidth=2)
    ax.plot(epochs_range, val_acc, label='Validation Accuracy', color='#5cb85c', linewidth=2, linestyle='--')
    if fine_history:
        ax.axvline(x=len(history.history['accuracy']), color='gray', linestyle=':', label='Fine-tune start')
    ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', color='#1a4a1a')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, 'accuracy.png')
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Accuracy plot saved: {acc_path}")

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, loss, label='Training Loss', color='#c0392b', linewidth=2)
    ax.plot(epochs_range, val_loss, label='Validation Loss', color='#e74c3c', linewidth=2, linestyle='--')
    if fine_history:
        ax.axvline(x=len(history.history['loss']), color='gray', linestyle=':', label='Fine-tune start')
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold', color='#4a1a1a')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'loss.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Loss plot saved: {loss_path}")

    return acc_path, loss_path


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir=GRAPHS_FOLDER):
    """Generate and save confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig_size = max(12, len(class_names) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=(len(class_names) <= 20), fmt='d',
        cmap='Greens', ax=ax,
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")
    return cm_path


def generate_training_report(metrics: dict, graph_paths: dict, save_dir=REPORTS_DIR):
    """Generate a PDF training report using ReportLab."""
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'training_report.pdf')

    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("CropCare AI – Model Training Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Metrics table
    story.append(Paragraph("Test Metrics", styles['Heading2']))
    data = [
        ['Metric', 'Value'],
        ['Test Accuracy', f"{metrics.get('accuracy', 0):.4f}"],
        ['Test Loss', f"{metrics.get('loss', 0):.4f}"],
        ['Num Classes', str(metrics.get('num_classes', 0))],
        ['Total Epochs', str(metrics.get('total_epochs', 0))],
    ]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d7a2d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Graphs
    for label, path_key in [('Accuracy Graph', 'acc'), ('Loss Graph', 'loss'), ('Confusion Matrix', 'cm')]:
        gpath = graph_paths.get(path_key)
        if gpath and os.path.exists(gpath):
            story.append(Paragraph(label, styles['Heading2']))
            story.append(RLImage(gpath, width=400, height=250))
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"[INFO] Training report saved: {report_path}")
    return report_path


# ─── Main training routine ────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  CropCare AI – MobileNetV2 Training Pipeline")
    print("=" * 60)

    # 1. Discover classes
    print("\n[STEP 1] Discovering dataset classes...")
    class_map = discover_classes(DATASET_DIR)
    num_classes = len(class_map)
    print(f"[INFO] Found {num_classes} classes:")
    for idx, name in sorted(class_map.items()):
        print(f"  [{idx:02d}] {name}")

    save_class_labels(class_map)

    # 2. Collect & split data
    print("\n[STEP 2] Collecting image paths...")
    items = collect_image_paths(DATASET_DIR, class_map)

    print("\n[STEP 3] Splitting dataset (70/15/15)...")
    train_items, val_items, test_items = split_dataset(items)

    # 3. Build tf.data pipelines
    print("\n[STEP 4] Building tf.data pipelines...")
    train_ds = build_tf_dataset(train_items, num_classes, BATCH_SIZE, augment=True, shuffle=True)
    val_ds   = build_tf_dataset(val_items, num_classes, BATCH_SIZE, augment=False)
    test_ds  = build_tf_dataset(test_items, num_classes, BATCH_SIZE, augment=False)

    # 4. Build model
    print("\n[STEP 5] Building MobileNetV2 model...")
    model = build_model(num_classes=num_classes, fine_tune=False)
    model.summary()

    # 5. Callbacks
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        ),
    ]

    # 6. Phase 1 Training (frozen base)
    print(f"\n[STEP 6] Phase 1 Training (base frozen, {EPOCHS} epochs max)...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1
    )

    # 7. Fine-tuning phase
    print(f"\n[STEP 7] Phase 2 Fine-tuning (unfreezing top 30 layers, {FINE_TUNE_EPOCHS} epochs)...")
    fine_model = build_model(num_classes=num_classes, fine_tune=True, fine_tune_at=100)

    # Load best weights from phase 1
    if os.path.exists(MODEL_PATH):
        fine_model.load_weights(MODEL_PATH)

    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    fine_history = fine_model.fit(
        train_ds, validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS, callbacks=fine_callbacks, verbose=1
    )

    # 8. Evaluate on test set
    print("\n[STEP 8] Evaluating on test set...")
    test_loss, test_acc = fine_model.evaluate(test_ds, verbose=1)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Generate predictions for classification report
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = fine_model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    class_names = [class_map[i] for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n[CLASSIFICATION REPORT]")
    print(report)

    # Save report to file
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 9. Save graphs
    print("\n[STEP 9] Generating plots...")
    acc_path, loss_path = plot_history(history, fine_history)
    cm_path = plot_confusion_matrix(y_true, y_pred, class_names)

    # 10. Generate PDF report
    print("\n[STEP 10] Generating PDF training report...")
    metrics = {
        'accuracy': test_acc, 'loss': test_loss,
        'num_classes': num_classes,
        'total_epochs': len(history.history['accuracy']) + len(fine_history.history['accuracy'])
    }
    generate_training_report(metrics, {'acc': acc_path, 'loss': loss_path, 'cm': cm_path})

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Class labels:   {CLASS_LABELS_PATH}")
    print(f"  Test Accuracy:  {test_acc:.2%}")
    print("=" * 60)


if __name__ == '__main__':
    # Ensure GPU memory growth (prevents OOM on some systems)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using {len(gpus)} GPU(s)")
    else:
        print("[INFO] No GPU found, using CPU")

    train()
