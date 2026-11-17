# 🌿 CropCare AI – AI-Based Crop Disease Detection Using Images

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![MobileNetV2](https://img.shields.io/badge/Architecture-MobileNetV2-green.svg)](https://keras.io/api/applications/mobilenet/)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen.svg)](https://www.kaggle.com/datasets/emmarex/plantdisease)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, deep learning-powered agricultural diagnostic web application that detects plant diseases from leaf photographs with high precision using transfer learning on **MobileNetV2**. Built with **Python**, **Flask**, **TensorFlow/Keras**, and modern responsive CSS/JavaScript.

---

## 📌 Table of Contents

- [Overview & Objectives](#-overview--objectives)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [PlantVillage Dataset](#-plantvillage-dataset)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Screenshots & UI Walkthrough](#-screenshots--ui-walkthrough)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Agricultural & Safety Disclaimer](#-agricultural--safety-disclaimer)
- [License & Contributions](#-license--contributions)

---

## 🎯 Overview & Objectives

Agricultural crop diseases reduce worldwide crop yields by 20% to 40% annually, severely affecting food security and smallholder farmers' livelihoods. Traditional visual scouting requires specialized phytopathologists who are rarely accessible on-site in remote farming regions.

**CropCare AI** bridges this gap by providing an accessible, real-time diagnostic tool:
1. **Instant Disease Diagnosis:** Analyzes leaf photos in seconds with confidence scoring.
2. **Actionable Agronomic Guidance:** Provides symptoms, etiology, cultural prevention practices, and integrated pest management (IPM) guidelines for 38 disease categories.
3. **Traceability:** Maintains an on-device SQLite prediction history and generates downloadable PDF diagnostic certificates for field records.
4. **Resilient Architecture:** Features graceful offline degradation, dynamic dataset discovery, and safe file validation.

---

## ✨ Key Features

- **MobileNetV2 Transfer Learning:** Pre-trained on ImageNet with custom GlobalAveragePooling, BatchNormalization, Dense(256), Dropout(0.5), and Softmax classification head.
- **Dynamic Dataset Discovery:** Automatically detects classes, directory structures, and labels without hardcoded indices.
- **Interactive Multi-Page Web Interface:**
  - 🏠 **Home / Dashboard:** Mission, feature highlights, step-by-step workflow, supported crop catalog.
  - 🔍 **Disease Detection:** Drag-and-drop file upload, live client-side image preview, async AJAX predictions, animated SVG circular confidence meter, top-3 alternate candidate classes.
  - 📚 **Disease Library:** Searchable and filterable reference encyclopedia covering all 38 PlantVillage classes with expandable treatment sheets.
  - 📖 **About Project:** Deep learning architecture breakdown, technical justifications, dataset metrics, step-by-step training terminal commands.
  - ✉️ **Contact:** Direct feedback form with client/server validation.
  - 🕐 **Prediction History:** Persistent SQLite record keeping with thumbnails, confidence badges, and timestamps.
  - 📄 **PDF Diagnostic Reports:** Generate and download structured PDF reports via ReportLab.
- **Robust Error Handling:** File format verification (magic bytes + extensions), upload size capping (16 MB), model-missing guidance, and HTTP 404/500 error pages.

---

## 🧠 System Architecture

```text
               +-------------------------------------------+
               |              Client Browser               |
               | (HTML5 / Bootstrap 5 / JS Drag-Drop / UI) |
               +-------------------------------------------+
                                     |
                         HTTP POST /predict (AJAX)
                                     v
               +-------------------------------------------+
               |               Flask Backend               |
               | (app.py: validation, security, routing)   |
               +-------------------------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +-----------------------+         +-----------------------+
        |   model/predict.py    |         |   SQLite History DB   |
        |  (Singleton Loader)   |         |      (history.db)     |
        +-----------------------+         +-----------------------+
                    |
      +-------------+-------------+
      |                           |
      v                           v
+-----------------------+   +-------------------------------+
|  crop_disease_model   |   |   disease_information.json    |
|       (.keras)        |   | (Symptoms, causes, prevention)|
+-----------------------+   +-------------------------------+
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ | Core development language |
| **Deep Learning** | TensorFlow 2.x / Keras | Model definition, training, inference |
| **Computer Vision** | OpenCV / Pillow / NumPy | Image decoding, resizing, color-space management |
| **Web Backend** | Flask | RESTful endpoints, Jinja2 template rendering |
| **Frontend** | HTML5, CSS3, JavaScript (ES6) | Responsive, mobile-first agricultural theme |
| **Database** | SQLite3 | Local storage for prediction history |
| **PDF Generation** | ReportLab | Diagnostic certificate creation |
| **Testing** | Pytest / Unittest | Unit and endpoint testing |

---

## 📊 PlantVillage Dataset

The model is trained on the benchmark **PlantVillage** dataset:
- **Total Images:** 54,306 RGB leaf images
- **Number of Classes:** 38 distinct crop-disease classes
- **Crop Types Supported:** 14 species:
  - Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
- **Data Source:** [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Supported Classes (38 Total):
1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___Cedar_apple_rust`
4. `Apple___healthy`
5. `Blueberry___healthy`
6. `Cherry_(including_sour)___Powdery_mildew`
7. `Cherry_(including_sour)___healthy`
8. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
9. `Corn_(maize)___Common_rust_`
10. `Corn_(maize)___Northern_Leaf_Blight`
11. `Corn_(maize)___healthy`
12. `Grape___Black_rot`
13. `Grape___Esca_(Black_Measles)`
14. `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
15. `Grape___healthy`
16. `Orange___Haunglongbing_(Citrus_greening)`
17. `Peach___Bacterial_spot`
18. `Peach___healthy`
19. `Pepper,_bell___Bacterial_spot`
20. `Pepper,_bell___healthy`
21. `Potato___Early_blight`
22. `Potato___Late_blight`
23. `Potato___healthy`
24. `Raspberry___healthy`
25. `Soybean___healthy`
26. `Squash___Powdery_mildew`
27. `Strawberry___Leaf_scorch`
28. `Strawberry___healthy`
29. `Tomato___Bacterial_spot`
30. `Tomato___Early_blight`
31. `Tomato___Late_blight`
32. `Tomato___Leaf_Mold`
33. `Tomato___Septoria_leaf_spot`
34. `Tomato___Spider_mites Two-spotted_spider_mite`
35. `Tomato___Target_Spot`
36. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
37. `Tomato___Tomato_mosaic_virus`
38. `Tomato___healthy`

---

## 📂 Project Directory Structure

```text
Crop_Disease_Detection/
│
├── app.py                      # Flask backend application with all routes
├── config.py                   # Centralized configuration and hyperparameter constants
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
│
├── model/
│   ├── mobilenet_model.py      # MobileNetV2 architecture with transfer learning head
│   ├── preprocessing.py        # Image transformations, dataset split & data pipeline
│   ├── train_model.py          # 2-phase training, early stopping, metrics & plots
│   ├── predict.py              # Singleton model inference engine
│   ├── crop_disease_model.keras# Trained weights (generated upon training)
│   └── class_labels.json       # Index-to-class label mapping (generated upon training)
│
├── dataset/
│   └── PlantVillage/           # Dataset directory containing class folders
│
├── database/
│   ├── disease_information.json# Complete agronomic profiles for all 38 classes
│   └── history.db              # SQLite prediction database (auto-created)
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom responsive styles & theme
│   ├── js/
│   │   └── script.js           # Drag-and-drop, AJAX requests & interactive UI
│   ├── images/                 # Static graphical assets
│   ├── uploads/                # User uploaded leaf images
│   └── graphs/                 # Training loss/accuracy curves and confusion matrix
│
├── templates/
│   ├── base.html               # Shared master layout template
│   ├── index.html              # Landing page / Dashboard
│   ├── detect.html             # Upload & disease detection workspace
│   ├── diseases.html           # Searchable disease catalog
│   ├── about.html              # Technical and methodology documentation
│   ├── contact.html            # Contact & inquiry form
│   ├── history.html            # Prediction logs with filter and thumbnails
│   ├── 404.html                # Page not found error handler
│   └── 500.html                # Internal server error handler
│
├── reports/
│   └── training_report.pdf     # Model evaluation report (generated after training)
│
└── tests/
    ├── test_app.py             # Flask route, error handling and validation tests
    └── test_prediction.py      # Model inference, preprocessing & logic unit tests
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or 3.11 recommended
- Git installed on your system
- 4GB+ RAM (8GB+ recommended if training the model)

### 1. Clone or Open the Repository
```bash
cd Crop_Disease_Detection
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️ Model Training & Evaluation

> **Note:** If you want to use the application without retraining, the app will run in informative mode, guiding you on how to set up the dataset.

### 1. Download PlantVillage Dataset
1. Download the dataset from [Kaggle PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Extract the class folders into `dataset/PlantVillage/`.
   The directory should look like:
   ```text
   dataset/PlantVillage/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ├── ...
   └── Tomato___healthy/
   ```

### 2. Execute Training Script
Run the automated two-stage training script:
```bash
python model/train_model.py
```

### What happens during training:
1. **Dynamic Class Discovery:** Scans subdirectories in `dataset/PlantVillage/` and generates `model/class_labels.json`.
2. **Stratified Split:** 70% Training, 15% Validation, 15% Test.
3. **Phase 1 (Feature Extraction):** MobileNetV2 base weights frozen. Trains top classification layers with Adam ($lr = 10^{-3}$) for 10 epochs.
4. **Phase 2 (Fine-Tuning):** Unfreezes top 30 layers of MobileNetV2. Trains with low learning rate ($lr = 10^{-5}$) with EarlyStopping and ReduceLROnPlateau.
5. **Artifacts Exported:**
   - Saved Model: `model/crop_disease_model.keras`
   - Graphs: `static/graphs/training_history.png` and `confusion_matrix.png`
   - Diagnostic Summary: `reports/training_report.pdf`

---

## 💻 Running the Application

Start the Flask development server:
```bash
python app.py
```

Then open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📡 API Documentation

### 1. Health Status
- **Endpoint:** `GET /api/status`
- **Response:**
  ```json
  {
    "status": "running",
    "model_trained": true,
    "model_path": ".../crop_disease_model.keras",
    "classes_count": 38
  }
  ```

### 2. Disease Prediction
- **Endpoint:** `POST /predict`
- **Content-Type:** `multipart/form-data`
- **Body:** `image: <binary file>` (JPG, JPEG, PNG, max 16MB)
- **Success Response (200 OK):**
  ```json
  {
    "success": true,
    "class_name": "Tomato___Early_blight",
    "crop": "Tomato",
    "disease": "Early Blight",
    "is_healthy": false,
    "confidence": 97.4,
    "top_predictions": [
      {"class": "Tomato___Early_blight", "crop": "Tomato", "disease": "Early Blight", "confidence": 97.4},
      {"class": "Tomato___Target_Spot", "crop": "Tomato", "disease": "Target Spot", "confidence": 1.8},
      {"class": "Tomato___Late_blight", "crop": "Tomato", "disease": "Late Blight", "confidence": 0.5}
    ],
    "disease_info": {
      "description": "Fungal infection caused by Alternaria solani...",
      "symptoms": ["Dark brown spots with concentric rings", "Yellowing halo around spots"],
      "causes": ["High humidity", "Warm temperatures (24-29°C)"],
      "prevention": ["Rotate crops every 2-3 years", "Mulch base to prevent splash"],
      "management": ["Remove infected leaves immediately", "Apply copper fungicide"],
      "disclaimer": "Consult local agricultural extension for registered products."
    },
    "image_filename": "uploaded_leaf.jpg"
  }
  ```

### 3. Disease Library Listing
- **Endpoint:** `GET /api/diseases`
- **Response:** Array of disease info objects with full agronomic profiles.

---

## 📸 Screenshots & UI Walkthrough

1. **Dashboard (`/`):** Features hero header, quick crop stats, 4-step workflow graphic, and direct access to detection.
2. **Detection Workspace (`/detect`):** Real-time drop zone with client-side thumbnail validation, live progress indicator, and circular confidence indicator.
3. **Disease Encyclopedia (`/diseases`):** Interactive filter bar allowing instantaneous multi-parameter filtering across crop types, severity categories, and keyword searches.
4. **Diagnostic Reports (`/report/<filename>`):** Generates printable PDF diagnosis certificates.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite covering routing, file uploads, SQLite history, and model inference mocks:

```bash
python -m pytest tests/ -v
```

Or using standard Python unittest:
```bash
python -m unittest discover tests/
```

---

## ⚠️ Agricultural & Safety Disclaimer

> **IMPORTANT:**
> Predictions generated by **CropCare AI** are powered by artificial intelligence and computer vision models intended as diagnostic aids. They **do not constitute professional agronomic certification**.
>
> - Environmental conditions, atypical symptoms, and nutritional deficiencies may produce visual similarities.
> - **Chemical & Pesticide Notice:** Never apply chemical fungicides, bactericides, or pesticides without consulting your local certified agricultural extension service or certified crop advisor for regional regulations, approved active ingredients, and precise safety intervals.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
