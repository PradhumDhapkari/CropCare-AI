"""
Automated tests for the CropCare AI Flask application.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import io
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app

flask_app.app.config['TESTING'] = True
flask_app.app.config['WTF_CSRF_ENABLED'] = False


@pytest.fixture
def client():
    """Create a Flask test client."""
    flask_app.init_db()
    with flask_app.app.test_client() as client:
        yield client


# ─── Route Tests ──────────────────────────────────────────────────────────────

class TestRoutes:
    """Test all GET routes return 200 OK."""

    def test_home_page(self, client):
        response = client.get('/')
        assert response.status_code == 200
        assert b'CropCare' in response.data

    def test_detect_page(self, client):
        response = client.get('/detect')
        assert response.status_code == 200
        assert b'Detect' in response.data or b'Upload' in response.data

    def test_diseases_page(self, client):
        response = client.get('/diseases')
        assert response.status_code == 200

    def test_about_page(self, client):
        response = client.get('/about')
        assert response.status_code == 200

    def test_contact_page(self, client):
        response = client.get('/contact')
        assert response.status_code == 200

    def test_history_page(self, client):
        response = client.get('/history')
        assert response.status_code == 200

    def test_api_status(self, client):
        response = client.get('/api/status')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'running'
        assert 'model_trained' in data

    def test_api_diseases(self, client):
        response = client.get('/api/diseases')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_404_page(self, client):
        response = client.get('/nonexistent-page-xyz')
        assert response.status_code == 404


# ─── Predict Endpoint Tests ───────────────────────────────────────────────────

class TestPredictEndpoint:
    """Test /predict endpoint validation and responses."""

    def test_predict_no_file(self, client):
        """Should return 400 when no file is provided."""
        response = client.post('/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_predict_empty_filename(self, client):
        """Should return 400 for empty filename."""
        data = {'image': (io.BytesIO(b''), '')}
        response = client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')
        assert response.status_code == 400

    def test_predict_invalid_extension(self, client):
        """Should reject non-image file types."""
        fake_file = io.BytesIO(b'not an image')
        data = {'image': (fake_file, 'malware.exe')}
        response = client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')
        assert response.status_code == 400
        resp_data = json.loads(response.data)
        assert resp_data['success'] is False

    def test_predict_invalid_image_content(self, client):
        """Should reject files with image extension but invalid content (when model exists)."""
        # Only test format rejection (model check comes first if model not available)
        fake_file = io.BytesIO(b'this is not a real jpeg')
        data = {'image': (fake_file, 'fake.jpg')}
        response = client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')
        # Either 400 (invalid image) or 503 (model not trained) are acceptable
        assert response.status_code in (400, 503)
        resp_data = json.loads(response.data)
        assert resp_data['success'] is False

    def test_predict_pdf_rejected(self, client):
        """PDF files should be rejected."""
        fake_pdf = io.BytesIO(b'%PDF-1.4 fake pdf content')
        data = {'image': (fake_pdf, 'document.pdf')}
        response = client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')
        assert response.status_code == 400

    def test_predict_model_missing_returns_503(self, client):
        """When model file doesn't exist, should return 503 with model_missing flag."""
        # Temporarily rename model to simulate missing
        model_path = flask_app.MODEL_PATH
        backup_path = model_path + '.bak'
        renamed = False

        if os.path.exists(model_path):
            os.rename(model_path, backup_path)
            renamed = True

        try:
            # Create a valid small PNG (1x1 pixel)
            png_1x1 = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk length + type
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # bit depth, color type
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
                0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
                0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND
                0x44, 0xAE, 0x42, 0x60, 0x82
            ])
            img_file = io.BytesIO(png_1x1)
            data = {'image': (img_file, 'test.png')}
            response = client.post('/predict',
                                    data=data,
                                    content_type='multipart/form-data')
            assert response.status_code == 503
            resp_data = json.loads(response.data)
            assert resp_data['success'] is False
            assert resp_data.get('model_missing') is True
        finally:
            if renamed and os.path.exists(backup_path):
                os.rename(backup_path, model_path)


# ─── Contact Form Tests ───────────────────────────────────────────────────────

class TestContactForm:
    """Test contact form validation."""

    def test_contact_post_valid(self, client):
        response = client.post('/contact', data={
            'name': 'Test User',
            'email': 'test@example.com',
            'message': 'This is a test message from the automated test suite.'
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_contact_post_missing_name(self, client):
        response = client.post('/contact', data={
            'name': '',
            'email': 'test@example.com',
            'message': 'Hello'
        })
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False

    def test_contact_post_invalid_email(self, client):
        response = client.post('/contact', data={
            'name': 'Test User',
            'email': 'not-an-email',
            'message': 'Hello'
        })
        assert response.status_code == 400

    def test_contact_post_missing_message(self, client):
        response = client.post('/contact', data={
            'name': 'Test',
            'email': 'test@example.com',
            'message': ''
        })
        assert response.status_code == 400


# ─── Database Tests ───────────────────────────────────────────────────────────

class TestDatabase:
    """Test SQLite history functionality."""

    def test_init_db(self):
        flask_app.init_db()
        import sqlite3
        conn = sqlite3.connect(flask_app.HISTORY_DB)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert 'predictions' in table_names

    def test_save_and_retrieve_prediction(self):
        flask_app.init_db()
        flask_app.save_prediction('test_img.jpg', 'Tomato', 'Early Blight', 95.5, False)
        records = flask_app.get_history(limit=1)
        assert len(records) >= 1
        last = records[0]
        assert last['crop'] == 'Tomato'
        assert last['disease'] == 'Early Blight'
        assert abs(last['confidence'] - 95.5) < 0.01

    def test_get_history_limit(self):
        flask_app.init_db()
        records = flask_app.get_history(limit=5)
        assert len(records) <= 5


# ─── Disease Info Tests ───────────────────────────────────────────────────────

class TestDiseaseInfo:
    """Test disease information loading."""

    def test_load_disease_info(self):
        diseases = flask_app.load_disease_info()
        # Either empty list (no DB) or populated list
        assert isinstance(diseases, list)

    def test_disease_info_has_required_fields(self):
        diseases = flask_app.load_disease_info()
        if diseases:
            d = diseases[0]
            assert 'class_name' in d
            assert 'crop' in d
            assert 'disease' in d
            assert 'symptoms' in d

    def test_disease_count(self):
        diseases = flask_app.load_disease_info()
        # We should have all 38 PlantVillage classes
        if diseases:
            assert len(diseases) == 38

    def test_allowed_file_extension(self):
        assert flask_app.allowed_file('test.jpg') is True
        assert flask_app.allowed_file('test.jpeg') is True
        assert flask_app.allowed_file('test.png') is True
        assert flask_app.allowed_file('test.gif') is False
        assert flask_app.allowed_file('test.pdf') is False
        assert flask_app.allowed_file('test.exe') is False
        assert flask_app.allowed_file('noextension') is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
