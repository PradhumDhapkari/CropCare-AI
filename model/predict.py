"""
Prediction module for CropCare AI.
Loads trained MobileNetV2 model once and performs disease classification.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, CLASS_LABELS_PATH, IMG_SIZE
from model.preprocessing import preprocess_image_for_prediction, load_class_labels


# ─── Singleton Model Loader ───────────────────────────────────────────────────

_model = None
_class_labels: Optional[Dict[int, str]] = None


def load_model_once():
    """
    Load the trained model and class labels into memory.
    Uses a singleton pattern – loads only on first call.
    
    Raises:
        FileNotFoundError: If model or class labels files are missing.
    """
    global _model, _class_labels

    if _model is not None:
        return _model, _class_labels

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Please train the model first:\n"
            "  python model/train_model.py"
        )

    if not os.path.exists(CLASS_LABELS_PATH):
        raise FileNotFoundError(
            f"Class labels file not found at: {CLASS_LABELS_PATH}\n"
            "Please train the model first:\n"
            "  python model/train_model.py"
        )

    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)
    _class_labels = load_class_labels(CLASS_LABELS_PATH)

    print(f"[INFO] Model loaded from: {MODEL_PATH}")
    print(f"[INFO] {len(_class_labels)} classes loaded.")
    return _model, _class_labels


def is_model_available() -> bool:
    """Check if model and class label files exist."""
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASS_LABELS_PATH)


# ─── Prediction ───────────────────────────────────────────────────────────────

def parse_class_name(class_name: str) -> Tuple[str, str]:
    """
    Parse PlantVillage class name into (crop, disease).
    Format: "Crop___Disease" or "Crop___healthy"

    Examples:
        "Tomato___Early_blight"   -> ("Tomato", "Early Blight")
        "Apple___Apple_scab"      -> ("Apple", "Apple Scab")
        "Corn_(maize)___healthy"  -> ("Corn (Maize)", "Healthy")
    """
    if '___' in class_name:
        parts = class_name.split('___', 1)
        crop = parts[0].replace('_', ' ').replace('(', '(').strip()
        disease = parts[1].replace('_', ' ').strip().title()
    else:
        crop = class_name.replace('_', ' ').strip()
        disease = 'Unknown'

    return crop, disease


def predict(image_path: str) -> Dict:
    """
    Predict crop disease from an image file.

    Args:
        image_path: Absolute path to the uploaded image.

    Returns:
        dict with keys:
            - success (bool)
            - class_name (str): Raw class label (e.g. "Tomato___Early_blight")
            - crop (str): Human-readable crop name
            - disease (str): Human-readable disease name
            - is_healthy (bool)
            - confidence (float): 0–100 percentage
            - top_predictions (list): Top 3 predictions with confidence
            - error (str): Error message if success=False
    """
    if not is_model_available():
        return {
            'success': False,
            'error': (
                'Model not trained yet. '
                'Please download the PlantVillage dataset and run: '
                'python model/train_model.py'
            )
        }

    try:
        model, class_labels = load_model_once()
    except FileNotFoundError as e:
        return {'success': False, 'error': str(e)}

    try:
        img_array = preprocess_image_for_prediction(image_path)
    except ValueError as e:
        return {'success': False, 'error': f'Image preprocessing failed: {str(e)}'}

    # Inference
    try:
        predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)
    except Exception as e:
        return {'success': False, 'error': f'Model inference error: {str(e)}'}

    # Top prediction
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx]) * 100.0
    class_name = class_labels[top_idx]
    crop, disease = parse_class_name(class_name)
    is_healthy = 'healthy' in class_name.lower()

    # Top 3 predictions
    top3_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = []
    for idx in top3_indices:
        c_name = class_labels[int(idx)]
        c_crop, c_disease = parse_class_name(c_name)
        top_predictions.append({
            'class_name': c_name,
            'crop': c_crop,
            'disease': c_disease,
            'confidence': round(float(predictions[idx]) * 100, 2)
        })

    return {
        'success': True,
        'class_name': class_name,
        'crop': crop,
        'disease': disease,
        'is_healthy': is_healthy,
        'confidence': round(confidence, 2),
        'top_predictions': top_predictions
    }


if __name__ == '__main__':
    # Quick test
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))
