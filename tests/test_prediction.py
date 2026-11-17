"""
Unit tests for the prediction pipeline.
Run with: python -m pytest tests/test_prediction.py -v
"""

import os
import sys
import json
import io
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_LABELS_PATH, MODEL_PATH, IMG_SIZE
from model.preprocessing import preprocess_image_for_prediction, load_class_labels, save_class_labels
from model.predict import parse_class_name, is_model_available, predict


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    """Test image preprocessing utilities."""

    def _create_test_image(self, path: str, size=(300, 300), color=(100, 150, 80)):
        """Create a valid JPEG test image."""
        img = Image.new('RGB', size, color=color)
        img.save(path, 'JPEG')
        return path

    def test_preprocess_valid_rgb_image(self, tmp_path):
        """Should preprocess a valid image to correct shape."""
        img_path = str(tmp_path / 'leaf.jpg')
        self._create_test_image(img_path)

        result = preprocess_image_for_prediction(img_path)

        assert result is not None
        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32

    def test_preprocess_different_sizes(self, tmp_path):
        """Should handle images of various sizes."""
        for size in [(100, 100), (1024, 768), (224, 224), (50, 300)]:
            img_path = str(tmp_path / f'img_{size[0]}x{size[1]}.jpg')
            self._create_test_image(img_path, size=size)
            result = preprocess_image_for_prediction(img_path)
            assert result.shape == (1, 224, 224, 3)

    def test_preprocess_grayscale_converted(self, tmp_path):
        """Grayscale images should be converted to RGB."""
        img_path = str(tmp_path / 'gray.jpg')
        img = Image.new('L', (200, 200), color=128)
        img.save(img_path)
        result = preprocess_image_for_prediction(img_path)
        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_png_image(self, tmp_path):
        """Should handle PNG images."""
        img_path = str(tmp_path / 'leaf.png')
        img = Image.new('RGB', (256, 256), color=(80, 150, 60))
        img.save(img_path, 'PNG')
        result = preprocess_image_for_prediction(img_path)
        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_nonexistent_file(self):
        """Should raise ValueError for missing file."""
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            preprocess_image_for_prediction('/nonexistent/path/image.jpg')

    def test_preprocess_invalid_file_content(self, tmp_path):
        """Should raise ValueError for non-image content."""
        bad_path = str(tmp_path / 'fake.jpg')
        with open(bad_path, 'wb') as f:
            f.write(b'this is not image data at all')
        with pytest.raises((ValueError, Exception)):
            preprocess_image_for_prediction(bad_path)

    def test_preprocess_values_normalized(self, tmp_path):
        """MobileNetV2 preprocessing should normalize to [-1, 1] range."""
        img_path = str(tmp_path / 'norm_test.jpg')
        self._create_test_image(img_path)
        result = preprocess_image_for_prediction(img_path)
        # MobileNetV2 normalizes to roughly [-1, 1]
        assert result.min() >= -1.5
        assert result.max() <= 1.5


# ─── Class Labels Tests ───────────────────────────────────────────────────────

class TestClassLabels:
    """Test class label save/load functionality."""

    def test_save_and_load_class_labels(self, tmp_path):
        """Should save and reload class labels correctly."""
        test_labels = {0: 'Apple___Apple_scab', 1: 'Tomato___healthy', 2: 'Corn_(maize)___Common_rust'}
        label_path = str(tmp_path / 'class_labels.json')

        save_class_labels(test_labels, path=label_path)
        assert os.path.exists(label_path)

        loaded = load_class_labels(path=label_path)
        assert loaded == test_labels

    def test_load_missing_labels_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing labels file."""
        with pytest.raises(FileNotFoundError):
            load_class_labels(path=str(tmp_path / 'nonexistent.json'))

    def test_labels_key_types(self, tmp_path):
        """Loaded labels should have integer keys."""
        test_labels = {0: 'ClassA', 1: 'ClassB'}
        label_path = str(tmp_path / 'labels.json')
        save_class_labels(test_labels, path=label_path)
        loaded = load_class_labels(path=label_path)
        for key in loaded:
            assert isinstance(key, int)


# ─── Class Name Parser Tests ──────────────────────────────────────────────────

class TestParseClassName:
    """Test the class name parsing logic."""

    def test_tomato_disease(self):
        crop, disease = parse_class_name('Tomato___Early_blight')
        assert 'Tomato' in crop
        assert 'Early' in disease or 'early' in disease.lower()

    def test_healthy_class(self):
        crop, disease = parse_class_name('Apple___healthy')
        assert 'Apple' in crop
        assert disease.lower() in ('healthy', 'healthy')

    def test_corn_with_parentheses(self):
        crop, disease = parse_class_name('Corn_(maize)___Common_rust')
        assert len(crop) > 0
        assert len(disease) > 0

    def test_grape_disease(self):
        crop, disease = parse_class_name('Grape___Black_rot')
        assert 'Grape' in crop
        assert 'Rot' in disease or 'rot' in disease.lower()

    def test_no_separator(self):
        """Should handle class names without ___ separator gracefully."""
        crop, disease = parse_class_name('UnknownCrop')
        assert len(crop) > 0

    def test_all_plant_village_classes(self):
        """Test all 38 PlantVillage class names can be parsed without error."""
        classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        for cls in classes:
            crop, disease = parse_class_name(cls)
            assert isinstance(crop, str) and len(crop) > 0, f"Empty crop for {cls}"
            assert isinstance(disease, str) and len(disease) > 0, f"Empty disease for {cls}"


# ─── Model Availability Tests ─────────────────────────────────────────────────

class TestModelAvailability:
    """Test model and class label availability checks."""

    def test_is_model_available_false_when_missing(self):
        """Should return False when model files don't exist."""
        with patch('model.predict.MODEL_PATH', '/nonexistent/model.keras'), \
             patch('model.predict.CLASS_LABELS_PATH', '/nonexistent/labels.json'):
            result = is_model_available()
            assert result is False

    def test_predict_returns_error_when_model_missing(self, tmp_path):
        """predict() should return success=False gracefully when model is missing."""
        # Create a valid test image
        img_path = str(tmp_path / 'test.jpg')
        img = Image.new('RGB', (224, 224), color=(100, 180, 80))
        img.save(img_path)

        with patch('model.predict.MODEL_PATH', '/nonexistent/model.keras'), \
             patch('model.predict.CLASS_LABELS_PATH', '/nonexistent/labels.json'), \
             patch('model.predict._model', None), \
             patch('model.predict._class_labels', None):
            result = predict(img_path)
            assert result['success'] is False
            assert 'error' in result

    def test_predict_with_mock_model(self, tmp_path):
        """predict() should return structured result with mocked model."""
        # Create a valid test image
        img_path = str(tmp_path / 'mock_test.jpg')
        img = Image.new('RGB', (224, 224), color=(80, 160, 60))
        img.save(img_path)

        # Mock class labels (38 classes)
        mock_labels = {i: f'TestCrop___TestDisease_{i}' for i in range(38)}
        mock_labels[0] = 'Tomato___Early_blight'

        # Mock model predictions
        mock_preds = np.zeros(38, dtype=np.float32)
        mock_preds[0] = 0.95  # 95% confidence for class 0
        mock_preds[1] = 0.03
        mock_preds[2] = 0.02

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([mock_preds])

        with patch('model.predict._model', mock_model), \
             patch('model.predict._class_labels', mock_labels), \
             patch('model.predict.MODEL_PATH', '/fake/model.keras'), \
             patch('model.predict.CLASS_LABELS_PATH', '/fake/labels.json'), \
             patch('os.path.exists', return_value=True):
            result = predict(img_path)

        assert result['success'] is True
        assert result['class_name'] == 'Tomato___Early_blight'
        assert result['crop'] == 'Tomato'
        assert abs(result['confidence'] - 95.0) < 1.0
        assert result['is_healthy'] is False
        assert len(result['top_predictions']) == 3


# ─── Healthy Class Detection Tests ───────────────────────────────────────────

class TestHealthyDetection:
    """Test that healthy classes are correctly identified."""

    def test_healthy_flag_for_healthy_class(self):
        # Simulate a prediction for a healthy class
        from model.predict import parse_class_name
        _, disease = parse_class_name('Tomato___healthy')
        assert 'Healthy' in disease or 'healthy' in disease.lower()

    def test_disease_flag_for_diseased_class(self):
        from model.predict import parse_class_name
        _, disease = parse_class_name('Tomato___Early_blight')
        assert 'healthy' not in disease.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
