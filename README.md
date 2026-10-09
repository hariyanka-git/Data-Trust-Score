# Dataset Reliability & Trust Scoring Platform 🔍

A robust, AI-powered internal tool designed to evaluate tabular datasets (CSVs) across standardized quality dimensions. It generates a unified trust score (0–100), offers explainable baseline algorithmic insights, and provides actionable data-cleaning recommendations.

![Platform Screenshot Placeholder](https://via.placeholder.com/1000x500.png?text=Data+Trust+Platform+Dashboard)

## 📌 Project Overview
Data trust is critical before model training and analytics. This platform serves as a completely uncoupled, standalone Web Dashboard backed by Python's advanced array manipulation libraries. Once a dataset is dropped in, it generates a comprehensive diagnostic report categorizing safety levels into **Safe (75-100)**, **Moderate Risk (50-74)**, and **High Risk (0-49)**. 

## 📊 Core Quality Dimensions

1. **Completeness (25%)**: Evaluates missing values per column and overall null percentages.
2. **Consistency (20%)**: Identifies exact duplicate rows, mixed data types in a single column, and leading/trailing whitespace formatting inconsistencies.
3. **Accuracy (20%)**: Employs static IQR metrics to flag egregious outliers and checks for mathematical infinite (`inf`) values.
4. **Bias (20%)**: Utilizes Shannon Entropy alongside Imbalance Ratios and Skewness algorithms to flag heavy categorical representation dominance or numerical distribution bias.
5. **Reliability (15%)**: Assesses table width depth, detects constant/zero-variance columns (100% redundant data), and checks for ID-leaking high-cardinality metadata columns.

## ⚙️ Core Algorithmic Drivers

- **Interquartile Range (IQR)**: Used by Accuracy metrics for structural distribution limits.
- **Shannon Entropy**: Measures proportional uncertainty across class labels to enforce target evenness.
- **Lexical Format Heuristics**: Applies Regex-style string replacements and strips to detect human-input anomalies computationally safely.

## 🌐 Platform APIs

The front-end strictly communicates with the backend via the following REST Endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/evaluate` | `POST` | Primary engine. Feed it a CSV file (or sample dataset name) and get unified JSON health matrices. |
| `/api/compare` | `POST` | Multi-dataset ranking engine. Submit multiple CSVs simultaneously to gauge overall comparative health. |
| `/api/sample/<type>` | `GET` | Generates on-the-fly synthetic data: "clean", "messy", or "mixed". |
| `/api/export/<fmt>` | `POST` | Generates a downloadable string formatting of an existing JSON configuration tree (CSV or tree). |
| `/api/health` | `GET` | Probe status API. |

## 🛠️ Tech Stack & Architecture

- **Backend Routing**: Flask REST server (configured with CORS functionality for independent serving).
- **Compute Engine**: Pandas & NumPy for heavily vectorized columnar dataset operations.
- **Visualizations**: Headless Matplotlib streaming base64 images directly into memory.
- **User Interface**: HTML5 with Vanilla JavaScript ES6 and flex/grid CSS architectures (zero Node.js/NPM dependencies).

## 🚀 How to Run Locally

### Requirements
- Python 3.8+

### Setup

1. **Clone the repository** (if applicable) and navigate to the root directory.
2. **Install Python dependencies**:
   ```bash
   pip install flask flask-cors pandas numpy matplotlib
   ```
3. **Launch the Flask Server**:
   ```bash
   python flask_app.py
   ```
4. **Open the Dashboard**:
   The web app operates straight out of the box. Head over to:
   [http://127.0.0.1:5001](http://127.0.0.1:5001)

## 📁 Repository Structure
```
├── core/
│   └── trust_engine.py      # Heavy-lifting Pandas metric computations
├── static/                  # Optional static assets 
├── templates/               # Contains index.html if using Jinja templates
├── flask_app.py             # Server routing and REST APIs
├── index.html               # Main frontend entry point (App Dashboard)
├── requirements.txt         # Dependency declarations
└── README.md                # Documentation
```
