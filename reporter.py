"""
Report Generator — Produces rich terminal reports and JSON exports.
"""

import json
import os
from datetime import datetime
from core.trust_engine import TrustReport


SEVERITY_COLORS = {
    "Critical": "\033[91m",   # red
    "High":     "\033[93m",   # yellow
    "Medium":   "\033[94m",   # blue
    "Low":      "\033[92m",   # green
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def print_full_report(report: TrustReport):
    """Print a beautifully formatted terminal report."""

    WIDTH = 70
    SEP = "─" * WIDTH

    def header(title):
        print(f"\n{SEP}")
        print(f"  {BOLD}{title}{RESET}")
        print(SEP)

    # ── Title Block
    print("\n" + "═" * WIDTH)
    print(f"  {BOLD}DATASET RELIABILITY & TRUST SCORE REPORT{RESET}")
    print("  SRM Institute of Science and Technology – Batch 18")
    print("═" * WIDTH)
    print(f"  Dataset   : {report.dataset_name}")
    print(f"  Evaluated : {report.evaluated_at}")
    print(f"  Shape     : {report.shape[0]:,} rows × {report.shape[1]} columns")

    # ── Trust Score Gauge
    score = report.trust_score
    filled = int(score // 2.5)
    bar = "█" * filled + "░" * (40 - filled)
    color = "\033[92m" if score >= 75 else "\033[93m" if score >= 50 else "\033[91m"
    risk_badge = {
        "Safe":           _color("✅  SAFE",          "\033[92m"),
        "Moderate Risk":  _color("⚠️   MODERATE RISK", "\033[93m"),
        "High Risk":      _color("🔴  HIGH RISK",      "\033[91m"),
    }.get(report.risk_label, report.risk_label)

    header("OVERALL TRUST SCORE")
    print(f"  {color}[{bar}]{RESET}")
    print(f"  Score : {BOLD}{_color(f'{score:.1f} / 100', color)}{RESET}")
    print(f"  Label : {risk_badge}")

    # ── Metric Breakdown
    header("METRIC BREAKDOWN")
    print(f"  {'Metric':<16} {'Score':>7}  {'Weight':>7}  {'Contribution':>13}  Status")
    print(f"  {'─'*14}  {'─'*6}  {'─'*7}  {'─'*12}  {'─'*10}")
    for name, m in report.metrics.items():
        bar_len = int(m.score // 10)
        mini_bar = "▓" * bar_len + "░" * (10 - bar_len)
        status = "✅" if m.score >= 75 else "⚠️" if m.score >= 50 else "❌"
        print(f"  {name:<16} {m.score:>6.1f}  {m.weight:>6.0%}  {m.weighted_score:>11.1f}   [{mini_bar}] {status}")

    # ── Issues Per Metric
    header("DETECTED ISSUES")
    for name, m in report.metrics.items():
        if m.issues:
            print(f"\n  {BOLD}{name}{RESET}")
            for issue in m.issues:
                print(f"    ⚠  {issue}")

    # ── Recommendations
    header("RECOMMENDATIONS")
    if not report.recommendations:
        print("  🎉 No major issues found!")
    else:
        for i, rec in enumerate(report.recommendations, 1):
            color_code = SEVERITY_COLORS.get(rec.severity, "")
            sev_label  = _color(f"[{rec.severity}]", color_code)
            print(f"\n  {i}. {sev_label} {BOLD}{rec.metric}{RESET}")
            print(f"     Issue      : {rec.issue}")
            print(f"     Suggestion : {rec.suggestion}")
            print(f"     Est. Gain  : +{rec.expected_gain:.1f} trust points")

    # ── Column Profiles
    header("COLUMN PROFILES (Top Stats)")
    for col, stats in list(report.column_profiles.items())[:10]:  # cap at 10
        null_pct = stats.get("null_pct", 0)
        null_indicator = f"\033[91m{null_pct:.1f}% null\033[0m" if null_pct > 20 else f"{null_pct:.1f}% null"
        extras = ""
        if "mean" in stats:
            extras = f"mean={stats['mean']}, std={stats['std']}, skew={stats.get('skewness', 'n/a')}"
        else:
            top = list(stats.get("top_values", {}).keys())[:3]
            extras = f"top: {top}"
        print(f"  {col:<25} {str(stats['dtype']):<12} {null_indicator:<22} {extras}")

    if len(report.column_profiles) > 10:
        print(f"  ... and {len(report.column_profiles) - 10} more columns")

    print("\n" + "═" * WIDTH + "\n")


def export_json(report: TrustReport, output_dir: str = "reports") -> str:
    """Export trust report to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = report.dataset_name.replace(" ", "_").replace("/", "_")
    filename = f"{output_dir}/trust_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = {
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

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  📄 JSON report saved → {filename}")
    return filename


def export_csv_summary(report: TrustReport, output_dir: str = "reports") -> str:
    """Export metric summary to CSV."""
    import csv
    os.makedirs(output_dir, exist_ok=True)
    safe_name = report.dataset_name.replace(" ", "_")
    filename = f"{output_dir}/trust_summary_{safe_name}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Trust Score", "Risk Label", "Evaluated At"])
        writer.writerow([report.dataset_name, report.trust_score, report.risk_label, report.evaluated_at])
        writer.writerow([])
        writer.writerow(["Metric", "Score", "Weight", "Weighted Score", "Issues"])
        for name, m in report.metrics.items():
            writer.writerow([name, m.score, m.weight, m.weighted_score, "; ".join(m.issues)])
        writer.writerow([])
        writer.writerow(["Severity", "Metric", "Issue", "Suggestion", "Expected Gain"])
        for r in report.recommendations:
            writer.writerow([r.severity, r.metric, r.issue, r.suggestion, r.expected_gain])

    print(f"  📊 CSV summary saved → {filename}")
    return filename
