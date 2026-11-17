"""
CropCare AI – Flask Application
Main entry point for the crop disease detection web application.
"""

import os
import json
import uuid
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, send_from_directory, abort
)

# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'cropcare-ai-secret-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
GRAPHS_FOLDER  = os.path.join(BASE_DIR, 'static', 'graphs')
DB_DIR         = os.path.join(BASE_DIR, 'database')
DISEASE_INFO   = os.path.join(DB_DIR, 'disease_information.json')
HISTORY_DB     = os.path.join(DB_DIR, 'history.db')
MODEL_PATH     = os.path.join(BASE_DIR, 'model', 'crop_disease_model.keras')
REPORTS_DIR    = os.path.join(BASE_DIR, 'reports')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    """Initialize SQLite history database."""
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            crop        TEXT,
            disease     TEXT,
            confidence  REAL,
            is_healthy  INTEGER,
            timestamp   TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction(filename, crop, disease, confidence, is_healthy):
    """Save a prediction result to the history database."""
    try:
        conn = sqlite3.connect(HISTORY_DB)
        conn.execute(
            'INSERT INTO predictions (filename, crop, disease, confidence, is_healthy, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
            (filename, crop, disease, confidence, int(is_healthy), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARN] Could not save prediction history: {e}")


def get_history(limit=50):
    """Retrieve prediction history from SQLite."""
    try:
        conn = sqlite3.connect(HISTORY_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_disease_info() -> list:
    """Load disease information from JSON database."""
    if not os.path.exists(DISEASE_INFO):
        return []
    with open(DISEASE_INFO, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_disease_details(class_name: str) -> dict:
    """Find disease details for a given class name."""
    diseases = load_disease_info()
    for d in diseases:
        if d.get('class_name') == class_name:
            return d
    # Fallback: match by partial name
    class_lower = class_name.lower().replace('___', ' ').replace('_', ' ')
    for d in diseases:
        d_name = d.get('class_name', '').lower().replace('___', ' ').replace('_', ' ')
        if d_name == class_lower:
            return d
    return {}


def model_is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


# ─── PDF Report Generation ────────────────────────────────────────────────────

def generate_prediction_pdf(prediction_data: dict, image_path: str) -> str:
    """Generate a downloadable PDF report for a prediction."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        report_id = str(uuid.uuid4())[:8]
        pdf_path = os.path.join(REPORTS_DIR, f'report_{report_id}.pdf')

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph('CropCare AI – Disease Detection Report', styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Report ID: {report_id}", styles['Normal']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Image
        if os.path.exists(image_path):
            story.append(RLImage(image_path, width=200, height=200))
            story.append(Spacer(1, 12))

        # Result table
        status = 'Healthy ✓' if prediction_data.get('is_healthy') else 'Diseased ✗'
        data = [
            ['Field', 'Result'],
            ['Crop', prediction_data.get('crop', 'N/A')],
            ['Disease', prediction_data.get('disease', 'N/A')],
            ['Status', status],
            ['Confidence', f"{prediction_data.get('confidence', 0):.1f}%"],
        ]
        table = Table(data, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d7a2d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Disease details
        details = prediction_data.get('disease_details', {})
        for section, label in [('description', 'Description'), ('symptoms', 'Symptoms'),
                                ('causes', 'Causes'), ('prevention', 'Prevention'),
                                ('management', 'Management')]:
            content = details.get(section)
            if content:
                story.append(Paragraph(label, styles['Heading2']))
                if isinstance(content, list):
                    for item in content:
                        story.append(Paragraph(f'• {item}', styles['Normal']))
                else:
                    story.append(Paragraph(str(content), styles['Normal']))
                story.append(Spacer(1, 10))

        # Disclaimer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            '⚠️ Disclaimer: Consult local agricultural experts for specific pesticide '
            'recommendations and treatment dosages.',
            styles['Italic']
        ))

        doc.build(story)
        return pdf_path
    except Exception as e:
        print(f"[WARN] PDF generation failed: {e}")
        return None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home/Dashboard page."""
    disease_count = len(load_disease_info())
    return render_template('index.html',
                           model_trained=model_is_trained(),
                           disease_count=disease_count)


@app.route('/detect')
def detect():
    """Disease detection page."""
    return render_template('detect.html', model_trained=model_is_trained())


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    POST endpoint for image classification.
    Accepts multipart/form-data with 'image' field.
    Returns JSON with prediction results.
    """
    # Validate file presence
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file format. Please upload JPG, JPEG, or PNG images only.'
        }), 400

    # Check model availability before saving file
    if not model_is_trained():
        return jsonify({
            'success': False,
            'error': (
                'The AI model has not been trained yet. '
                'Please download the PlantVillage dataset and run: '
                'python model/train_model.py\n'
                'See README.md for detailed instructions.'
            ),
            'model_missing': True
        }), 503

    # Save uploaded file
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Could not save uploaded file: {str(e)}'}), 500

    # Validate image content
    try:
        from PIL import Image as PILImage
        with PILImage.open(save_path) as img:
            img.verify()
    except Exception:
        os.remove(save_path)
        return jsonify({'success': False, 'error': 'Uploaded file is not a valid image.'}), 400

    # Run prediction
    try:
        from model.predict import predict as run_predict
        result = run_predict(save_path)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

    if not result.get('success'):
        return jsonify(result), 500

    # Enrich with disease details from DB
    disease_details = get_disease_details(result.get('class_name', ''))
    result['disease_details'] = disease_details

    # Save to history
    save_prediction(
        filename=unique_name,
        crop=result.get('crop', ''),
        disease=result.get('disease', ''),
        confidence=result.get('confidence', 0),
        is_healthy=result.get('is_healthy', False)
    )

    # Add image URL for frontend display
    result['image_url'] = url_for('static', filename=f'uploads/{unique_name}')
    result['report_url'] = url_for('download_report_route', filename=unique_name)

    return jsonify(result)


@app.route('/diseases')
def diseases():
    """Disease library page."""
    disease_list = load_disease_info()
    crops = sorted(set(d.get('crop', '') for d in disease_list if d.get('crop')))
    categories = sorted(set(d.get('category', '') for d in disease_list if d.get('category')))
    return render_template('diseases.html',
                           diseases=disease_list,
                           crops=crops,
                           categories=categories)


@app.route('/about')
def about():
    """About project page."""
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page with form handling."""
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        email   = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()

        errors = []
        if not name:    errors.append('Name is required.')
        if not email or '@' not in email: errors.append('Valid email is required.')
        if not message: errors.append('Message is required.')

        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        # In a real app, send email here. For now, log and confirm.
        print(f"[CONTACT] From: {name} <{email}> | Message: {message[:100]}")
        return jsonify({'success': True, 'message': 'Thank you! Your message has been received.'})

    return render_template('contact.html')


@app.route('/history')
def history():
    """Prediction history page."""
    records = get_history(limit=50)
    return render_template('history.html', records=records)


@app.route('/download-report/<filename>')
def download_report_route(filename):
    """Generate and download a PDF report for a prediction."""
    image_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(image_path):
        abort(404)

    # Try to get prediction data from history
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        'SELECT * FROM predictions WHERE filename = ? LIMIT 1', (filename,)
    ).fetchone()
    conn.close()

    if not row:
        abort(404)

    prediction_data = dict(row)
    disease_details = get_disease_details(
        f"{prediction_data.get('crop', '')}___{prediction_data.get('disease', '')}"
    )
    prediction_data['disease_details'] = disease_details

    pdf_path = generate_prediction_pdf(prediction_data, image_path)
    if not pdf_path:
        return jsonify({'error': 'Could not generate PDF report.'}), 500

    return send_from_directory(
        REPORTS_DIR,
        os.path.basename(pdf_path),
        as_attachment=True,
        download_name=f'cropcare_report_{filename[:8]}.pdf'
    )


@app.route('/api/diseases')
def api_diseases():
    """JSON API for disease library data (used by JS filtering)."""
    return jsonify(load_disease_info())


@app.route('/api/status')
def api_status():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'model_trained': model_is_trained(),
        'disease_db_loaded': os.path.exists(DISEASE_INFO),
        'timestamp': datetime.now().isoformat()
    })


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16 MB.'}), 413


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 55)
    print("  🌿 CropCare AI – Starting Server")
    print("=" * 55)
    print(f"  Model trained: {'✓ YES' if model_is_trained() else '✗ NO (run train_model.py)'}")
    print(f"  Disease DB:    {'✓ Loaded' if os.path.exists(DISEASE_INFO) else '✗ Missing'}")
    print(f"  URL:           http://0.0.0.0:{port}")
    print("=" * 55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
