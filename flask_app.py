"""
=============================================================
 Data Trust Score — Flask Web Application
=============================================================
 Run:   python flask_app.py
 Open:  http://localhost:5001
=============================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import io
import json
import base64
import traceback
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.trust_engine import DataTrustEngine

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
engine = DataTrustEngine()

PALETTE = {
    "bg": "#050810", "card": "#0a0f1e", "border": "#1a2540",
    "safe": "#00e676", "moderate": "#ffeb3b", "risk": "#ff4757",
    "accent": "#00d4ff", "text": "#e8eaf6", "muted": "#546e7a",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def score_color(score):
    if score >= 75: return PALETTE["safe"]
    if score >= 50: return PALETTE["moderate"]
    return PALETTE["risk"]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=PALETTE["bg"], transparent=False)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def generate_radar_chart(metrics_data):
    names  = list(metrics_data.keys())
    scores = [metrics_data[n]["score"] for n in names]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    values = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    # Background rings
    for r in [20, 40, 60, 80, 100]:
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200,
                color=PALETTE["border"], linewidth=0.6, alpha=0.5)

    # Fill gradient simulation
    ax.fill(angles, values, alpha=0.18, color=PALETTE["accent"])
    ax.plot(angles, values, "o-", linewidth=2.5, color=PALETTE["accent"],
            markerfacecolor=PALETTE["accent"], markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, color=PALETTE["text"], fontsize=10, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.grid(color=PALETTE["border"], linewidth=0.5)
    ax.spines["polar"].set_color(PALETTE["border"])

    return fig_to_base64(fig)


def generate_bar_chart(metrics_data):
    names  = list(metrics_data.keys())
    scores = [metrics_data[n]["score"] for n in names]
    colors = [score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    bars = ax.barh(names, scores, color=colors, edgecolor="none", height=0.55)
    for bar, val in zip(bars, scores):
        ax.text(min(val + 1, 108), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", color=PALETTE["text"], fontsize=9, fontweight="bold")

    ax.axvline(75, color=PALETTE["safe"],     linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(50, color=PALETTE["moderate"], linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Score / 100", color=PALETTE["text"], fontsize=9)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.set_facecolor(PALETTE["card"])

    return fig_to_base64(fig)


def generate_missing_chart(col_profiles):
    miss = {col: p["null_pct"] for col, p in col_profiles.items()}
    cols = sorted(miss, key=miss.get, reverse=True)[:12]
    vals = [miss[c] for c in cols]
    colors = [score_color(100 - v) for v in vals]

    fig, ax = plt.subplots(figsize=(6, max(3, len(cols) * 0.4 + 1)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    ax.barh(cols, vals, color=colors, edgecolor="none", height=0.55)
    ax.axvline(20, color=PALETTE["moderate"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Missing %", color=PALETTE["text"], fontsize=9)
    ax.set_title("Missing Values by Column", color=PALETTE["text"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])

    return fig_to_base64(fig)


def report_to_dict(report):
    return {
        "dataset_name":    report.dataset_name,
        "evaluated_at":    report.evaluated_at,
        "shape":           {"rows": report.shape[0], "columns": report.shape[1]},
        "trust_score":     report.trust_score,
        "risk_label":      report.risk_label,
        "summary":         report.summary,
        "metrics": {
            name: {
                "score":          m.score,
                "weight":         m.weight,
                "weighted_score": m.weighted_score,
                "issues":         m.issues,
                "details":        m.details,
            }
            for name, m in report.metrics.items()
        },
        "recommendations": [
            {
                "metric":        r.metric,
                "severity":      r.severity,
                "issue":         r.issue,
                "suggestion":    r.suggestion,
                "expected_gain": r.expected_gain,
            }
            for r in report.recommendations
        ],
        "column_profiles": report.column_profiles,
    }


# ─────────────────────────────────────────────────────────────
# Sample Dataset Generators
# ─────────────────────────────────────────────────────────────

def make_clean(n=600):
    np.random.seed(0)
    return pd.DataFrame({
        "customer_id":   range(1, n + 1),
        "age":           np.random.randint(18, 70, n),
        "annual_income": np.random.normal(55000, 12000, n).clip(15000).round(2),
        "credit_score":  np.random.randint(300, 850, n),
        "gender":        np.random.choice(["M", "F", "Non-binary"], n, p=[0.49, 0.49, 0.02]),
        "education":     np.random.choice(["High School", "Bachelor", "Master", "PhD"],
                                           n, p=[0.3, 0.40, 0.20, 0.10]),
        "region":        np.random.choice(["North", "South", "East", "West"], n),
        "loan_default":  np.random.choice([0, 1], n, p=[0.80, 0.20]),
    })


def make_messy(n=600):
    np.random.seed(7)
    df = pd.DataFrame({
        "id":            [f"USR_{i:04d}" for i in range(n)],
        "Age":           np.random.randint(18, 70, n).astype(float),
        "Income":        np.random.normal(55000, 12000, n).round(2),
        "Category":      np.random.choice(["A","B","C","D"], n, p=[0.88,0.06,0.04,0.02]),
        "score":         np.append(np.random.normal(50, 10, n - 40),
                                   np.random.uniform(200, 800, 40)),
        "constant_flag": ["FLAG_X"] * n,
        "target":        np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })
    for col in ["Age", "Income", "score"]:
        mask = np.random.random(n) < 0.30
        df.loc[mask, col] = np.nan
    df = pd.concat([df, df.sample(80, random_state=3)], ignore_index=True)
    df["Age"] = df["Age"].astype(object)
    df.loc[df.sample(15, random_state=5).index, "Age"] = "unknown"
    return df


def make_mixed(n=600):
    np.random.seed(42)
    df = pd.DataFrame({
        "age":    np.random.randint(18, 70, n),
        "income": np.random.normal(55000, 15000, n).round(2),
        "region": np.random.choice(["North","South","East","West"], n),
        "score":  np.append(np.random.normal(50,10,n-20), np.random.uniform(120,250,20)),
        "grade":  np.random.choice(["A","B","C","D"], n, p=[0.55,0.25,0.15,0.05]),
        "target": np.random.choice([0, 1], n, p=[0.65, 0.35]),
    })
    df.loc[np.random.random(n) < 0.07, "income"] = np.nan
    df.loc[np.random.random(n) < 0.05, "score"]  = np.nan
    df = pd.concat([df, df.sample(15, random_state=9)], ignore_index=True)
    return df


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/sample/<dataset_type>", methods=["GET"])
def get_sample(dataset_type):
    """Return sample CSV data."""
    generators = {"clean": make_clean, "messy": make_messy, "mixed": make_mixed}
    if dataset_type not in generators:
        return jsonify({"error": "Unknown sample type"}), 400
    df = generators[dataset_type]()
    return jsonify({
        "csv": df.to_csv(index=False),
        "name": f"{dataset_type.title()} Sample Dataset",
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Evaluate an uploaded CSV and return the full trust report + charts."""
    try:
        if "file" in request.files:
            f    = request.files["file"]
            df   = pd.read_csv(f)
            name = f.filename.replace(".csv", "")
        elif request.json and "sample" in request.json:
            generators = {"clean": make_clean, "messy": make_messy, "mixed": make_mixed}
            df   = generators[request.json["sample"]]()
            name = f"{request.json['sample'].title()} Sample Dataset"
        else:
            return jsonify({"error": "No file or sample specified"}), 400

        report = engine.evaluate(df, name)
        data   = report_to_dict(report)

        # Generate charts
        data["charts"] = {
            "radar":   generate_radar_chart(data["metrics"]),
            "bars":    generate_bar_chart(data["metrics"]),
            "missing": generate_missing_chart(data["column_profiles"]),
        }

        return jsonify({"success": True, "report": data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare multiple uploaded CSV files."""
    try:
        files = request.files.getlist("files")
        if len(files) < 2:
            return jsonify({"error": "Upload at least 2 files"}), 400

        results = []
        for f in files:
            df     = pd.read_csv(f)
            name   = f.filename.replace(".csv", "")
            report = engine.evaluate(df, name)
            d      = report_to_dict(report)
            d["charts"] = {
                "radar":   generate_radar_chart(d["metrics"]),
                "bars":    generate_bar_chart(d["metrics"]),
                "missing": generate_missing_chart(d["column_profiles"]),
            }
            results.append(d)

        # Comparison chart
        names  = [r["dataset_name"] for r in results]
        scores = [r["trust_score"] for r in results]
        colors = [score_color(s) for s in scores]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["card"])
        bars = ax.bar(names, scores, color=colors, edgecolor="none", width=0.55)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{score:.1f}", ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.axhline(75, color=PALETTE["safe"],     linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(50, color=PALETTE["moderate"], linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Trust Score", color=PALETTE["text"])
        ax.set_title("Dataset Comparison", color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=PALETTE["text"])
        for spine in ax.spines.values(): spine.set_color(PALETTE["border"])
        comp_chart = fig_to_base64(fig)

        return jsonify({
            "success": True,
            "reports": results,
            "comparison_chart": comp_chart,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export/<fmt>", methods=["POST"])
def export_report(fmt):
    """Export report as JSON or CSV."""
    data = request.json
    if not data:
        return jsonify({"error": "No report data"}), 400
    if fmt == "json":
        return app.response_class(
            response=json.dumps(data, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="trust_report.json"'}
        )
    elif fmt == "csv":
        lines = ["Metric,Score,Weight,Issues"]
        for name, m in data.get("metrics", {}).items():
            issues = "; ".join(m.get("issues", []))
            lines.append(f"{name},{m['score']},{m['weight']},\"{issues}\"")
        lines.append("")
        lines.append(f"Trust Score,{data['trust_score']}")
        lines.append(f"Risk Label,{data['risk_label']}")
        lines.append("")
        lines.append("Severity,Metric,Issue,Suggestion,Expected Gain")
        for r in data.get("recommendations", []):
            lines.append(f"{r['severity']},{r['metric']},\"{r['issue']}\",\"{r['suggestion']}\",{r['expected_gain']}")
        return app.response_class(
            response="\n".join(lines),
            mimetype="text/csv",
            headers={"Content-Disposition": 'attachment; filename="trust_summary.csv"'}
        )
    return jsonify({"error": "Unknown format"}), 400


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🔍 Data Trust Score Platform — Web Server")
    print("=" * 60)
    print("  Open in browser: http://localhost:5001")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
