"""
MobileNetV2 model architecture for crop disease detection.
Transfer learning from ImageNet weights.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from config import IMG_SIZE, LEARNING_RATE


def build_model(num_classes: int, fine_tune: bool = False, fine_tune_at: int = 100) -> Model:
    """
    Build MobileNetV2-based transfer learning model.

    Args:
        num_classes: Number of disease classes to predict.
        fine_tune: Whether to unfreeze top layers for fine-tuning.
        fine_tune_at: Layer index from which to unfreeze (for fine-tuning).

    Returns:
        Compiled Keras Model.
    """
    # Load MobileNetV2 without the top classification head
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze all base model layers initially
    base_model.trainable = False

    if fine_tune:
        # Unfreeze layers from fine_tune_at onwards
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Build custom classification head
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name='input_image')
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='batch_norm')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='CropCare_MobileNetV2')

    # Compile
    lr = LEARNING_RATE if not fine_tune else 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary(model: Model) -> str:
    """Return model architecture summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test build
    model = build_model(num_classes=38)
    print(get_model_summary(model))
    print(f"\nTotal params: {model.count_params():,}")
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"Trainable params: {trainable:,}")
