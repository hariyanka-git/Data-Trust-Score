# DataTrust Score — Tabular Data Health & Quality Assessor 🎯

A lightweight, high-performance web tool engineered to audit tabular datasets (CSVs) across core data quality metrics. DataTrust Score analyzes uploaded files to calculate a unified reliability score (0–100), identify potential risks, and generate actionable insights prior to downstream machine learning or business intelligence ingestion.

---

## 💡 Overview

Data quality directly impacts model accuracy and analytics reliability. DataTrust Score operates as a standalone web application backed by Python data-processing libraries. Once a dataset is ingested, the engine generates an instant diagnostic assessment categorizing overall dataset health:

* 🟢 **High Integrity (75–100)**: Clean data ready for analytics pipelines.
* 🟡 **Moderate Risk (50–74)**: Minor quality issues detected requiring review.
* 🔴 **High Risk (0–49)**: Critical data defects requiring immediate remediation.

---

## 📐 Scoring & Evaluation Breakdown

The overall score is calculated using a weighted matrix across five core dimensions:

1. **Completeness (25%)** — Measures overall missing value ratios and column-level null frequencies.
2. **Consistency (20%)** — Identifies exact duplicate records, conflicting data types, and formatting anomalies.
3. **Accuracy (20%)** — Leverages Interquartile Range (IQR) bounds to flag structural outliers and numerical overflow (`inf`).
4. **Distribution & Bias (20%)** — Applies Shannon Entropy and Skewness algorithms to flag heavily imbalanced distributions or high-cardinality flags.
5. **Structure & Reliability (15%)** — Detects zero-variance (constant) columns and high-cardinality metadata identifiers.

---

## 🛠️ Built With

* **Core Engine:** Python, Pandas, NumPy
* **Visualization:** Matplotlib (headless memory-buffer streaming)
* **API Framework:** Flask (RESTful architecture)
* **Frontend:** HTML5, CSS3, JavaScript (ES6)

---

## 🚀 Quickstart Guide

### Prerequisites
* Python 3.8 or higher installed on your system.

### Running Locally

1. **Clone or download the repository:**
   ```bash
   git clone [https://github.com/hariyanka-git/Data-Trust-Score.git](https://github.com/hariyanka-git/Data-Trust-Score.git)
   cd Data-Trust-Score
