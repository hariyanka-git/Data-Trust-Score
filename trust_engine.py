"""
=============================================================
 Dataset Reliability & Trust Scoring Engine
 SRM Institute of Science and Technology – Batch 18
=============================================================
 Authors : Dhanush G Reddy | Dinesh Shanmugavel | Hariyanka D
 Supervisor: Dr. N. Suganthi
=============================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    score: float                   # 0–100
    weight: float                  # contribution weight
    weighted_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.weighted_score = round(self.score * self.weight, 4)


@dataclass
class Recommendation:
    metric: str
    severity: str          # Critical / High / Medium / Low
    issue: str
    suggestion: str
    expected_gain: float   # trust score points gained after fix


@dataclass
class TrustReport:
    dataset_name: str
    evaluated_at: str
    shape: Tuple[int, int]
    trust_score: float
    risk_label: str          # Safe / Moderate Risk / High Risk
    metrics: Dict[str, MetricResult]
    recommendations: List[Recommendation]
    column_profiles: Dict
    summary: str


# ─────────────────────────────────────────────────────────────
# Individual Metric Calculators
# ─────────────────────────────────────────────────────────────

class CompletenessMetric:
    """Evaluates missing-value coverage across all columns."""

    WEIGHT = 0.25

    def compute(self, df: pd.DataFrame) -> MetricResult:
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        score = round(completeness_ratio * 100, 2)

        missing_per_col = df.isnull().mean() * 100
        high_missing = missing_per_col[missing_per_col > 20].to_dict()

        issues = []
        if missing_cells > 0:
            issues.append(f"{missing_cells} missing values detected ({missing_cells/total_cells*100:.1f}% of dataset)")
        for col, pct in high_missing.items():
            issues.append(f"Column '{col}' has {pct:.1f}% missing values")

        return MetricResult(
            name="Completeness",
            score=score,
            weight=self.WEIGHT,
            issues=issues,
            details={
                "total_cells": total_cells,
                "missing_cells": int(missing_cells),
                "completeness_pct": round(completeness_ratio * 100, 2),
                "missing_per_column": {k: round(v, 2) for k, v in missing_per_col.items()},
                "high_missing_columns": high_missing
            }
        )


class ConsistencyMetric:
    """Detects duplicates, data type mismatches, and format inconsistencies."""

    WEIGHT = 0.20

    def compute(self, df: pd.DataFrame) -> MetricResult:
        n = len(df)
        issues = []
        penalties = 0.0

        # Duplicate rows
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / n * 100 if n > 0 else 0
        if dup_pct > 0:
            penalties += min(dup_pct * 1.5, 30)
            issues.append(f"{dup_count} duplicate rows ({dup_pct:.1f}%)")

        # Mixed type columns
        mixed_cols = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(500)
            has_num = sample.apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).any()
            has_str = sample.apply(lambda x: not str(x).replace(".", "").replace("-", "").isdigit()).any()
            if has_num and has_str:
                mixed_cols.append(col)

        if mixed_cols:
            penalties += len(mixed_cols) * 5
            issues.append(f"Mixed data types in columns: {mixed_cols}")

        # Whitespace / case inconsistencies in string cols
        ws_cols = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().astype(str)
            if (sample != sample.str.strip()).any():
                ws_cols.append(col)

        if ws_cols:
            penalties += len(ws_cols) * 2
            issues.append(f"Leading/trailing whitespace in: {ws_cols}")

        score = max(0, round(100 - penalties, 2))

        return MetricResult(
            name="Consistency",
            score=score,
            weight=self.WEIGHT,
            issues=issues,
            details={
                "duplicate_rows": int(dup_count),
                "duplicate_pct": round(dup_pct, 2),
                "mixed_type_columns": mixed_cols,
                "whitespace_columns": ws_cols
            }
        )


class AccuracyMetric:
    """Statistical outlier detection and range validation."""

    WEIGHT = 0.20

    def compute(self, df: pd.DataFrame) -> MetricResult:
        issues = []
        penalties = 0.0
        outlier_report = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
            outlier_pct = len(outliers) / len(series) * 100
            outlier_report[col] = round(outlier_pct, 2)

            if outlier_pct > 5:
                penalties += min(outlier_pct * 0.8, 15)
                issues.append(f"Column '{col}': {outlier_pct:.1f}% outliers detected")

            # Infinite values
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                penalties += 10
                issues.append(f"Column '{col}': {inf_count} infinite values")

        score = max(0, round(100 - penalties, 2))

        return MetricResult(
            name="Accuracy",
            score=score,
            weight=self.WEIGHT,
            issues=issues,
            details={
                "numeric_columns_checked": len(numeric_cols),
                "outlier_percentage_per_column": outlier_report
            }
        )


class BiasMetric:
    """Class imbalance, demographic skew, and representation bias detection."""

    WEIGHT = 0.20

    def compute(self, df: pd.DataFrame) -> MetricResult:
        issues = []
        penalties = 0.0
        bias_report = {}

        cat_cols = df.select_dtypes(include="object").columns.tolist()

        for col in cat_cols:
            vc = df[col].value_counts(normalize=True)
            if len(vc) < 2:
                continue
            max_freq = vc.iloc[0]
            min_freq = vc.iloc[-1]
            imbalance_ratio = max_freq / min_freq if min_freq > 0 else float("inf")
            entropy = -np.sum(vc * np.log2(vc + 1e-9))
            max_entropy = np.log2(len(vc))
            evenness = entropy / max_entropy if max_entropy > 0 else 1

            bias_report[col] = {
                "imbalance_ratio": round(imbalance_ratio, 2),
                "evenness_score": round(evenness, 3),
                "dominant_class": str(vc.index[0]),
                "dominant_pct": round(max_freq * 100, 1)
            }

            if imbalance_ratio > 10 or max_freq > 0.85:
                penalty = min((1 - evenness) * 30, 20)
                penalties += penalty
                issues.append(
                    f"Column '{col}': dominant class '{vc.index[0]}' = {max_freq*100:.1f}% "
                    f"(imbalance ratio {imbalance_ratio:.1f}x)"
                )

        # Numeric skewness check
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 10:
                skew = abs(series.skew())
                if skew > 2.0:
                    penalties += min(skew * 2, 10)
                    issues.append(f"Column '{col}': high skewness ({skew:.2f}) may introduce bias")

        score = max(0, round(100 - penalties, 2))

        return MetricResult(
            name="Bias",
            score=score,
            weight=self.WEIGHT,
            issues=issues,
            details={"category_bias": bias_report}
        )


class ReliabilityMetric:
    """Schema stability, variance, and row-level reliability signals."""

    WEIGHT = 0.15

    def compute(self, df: pd.DataFrame) -> MetricResult:
        issues = []
        penalties = 0.0
        n, m = df.shape

        # Constant columns (zero variance)
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if const_cols:
            penalties += len(const_cols) * 5
            issues.append(f"Constant/zero-variance columns: {const_cols}")

        # Near-constant columns (>98% same value)
        near_const = []
        for col in df.columns:
            top_freq = df[col].value_counts(normalize=True).iloc[0] if df[col].nunique() > 0 else 1
            if top_freq > 0.98 and col not in const_cols:
                near_const.append(col)
        if near_const:
            penalties += len(near_const) * 3
            issues.append(f"Near-constant columns (>98% same value): {near_const}")

        # Too few rows
        if n < 50:
            penalties += 20
            issues.append(f"Dataset too small ({n} rows); reliability estimates are uncertain")
        elif n < 200:
            penalties += 10
            issues.append(f"Small dataset ({n} rows); consider gathering more samples")

        # High-cardinality text cols (potential ID leakage)
        id_like = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() / n > 0.9]
        if id_like:
            penalties += len(id_like) * 4
            issues.append(f"Possible ID/key columns (very high cardinality): {id_like}")

        score = max(0, round(100 - penalties, 2))

        return MetricResult(
            name="Reliability",
            score=score,
            weight=self.WEIGHT,
            issues=issues,
            details={
                "row_count": n,
                "col_count": m,
                "constant_columns": const_cols,
                "near_constant_columns": near_const,
                "id_like_columns": id_like
            }
        )


# ─────────────────────────────────────────────────────────────
# Column Profiler
# ─────────────────────────────────────────────────────────────

class ColumnProfiler:
    """Generates per-column statistical profiles."""

    def profile(self, df: pd.DataFrame) -> Dict:
        profiles = {}
        for col in df.columns:
            s = df[col]
            p = {
                "dtype": str(s.dtype),
                "null_count": int(s.isnull().sum()),
                "null_pct": round(s.isnull().mean() * 100, 2),
                "unique_count": int(s.nunique()),
                "unique_pct": round(s.nunique() / len(s) * 100, 2) if len(s) > 0 else 0,
            }
            if pd.api.types.is_numeric_dtype(s):
                desc = s.describe()
                p.update({
                    "mean": round(float(desc["mean"]), 4) if "mean" in desc else None,
                    "std":  round(float(desc["std"]),  4) if "std"  in desc else None,
                    "min":  round(float(desc["min"]),  4) if "min"  in desc else None,
                    "max":  round(float(desc["max"]),  4) if "max"  in desc else None,
                    "median": round(float(s.median()), 4),
                    "skewness": round(float(s.skew()), 4),
                    "kurtosis": round(float(s.kurtosis()), 4),
                })
            else:
                top_vals = s.value_counts().head(5).to_dict()
                p["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
            profiles[col] = p
        return profiles


# ─────────────────────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────────────────────

class RecommendationEngine:
    """Generates actionable recommendations with severity & expected gain."""

    SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

    def generate(self, metrics: Dict[str, MetricResult]) -> List[Recommendation]:
        recs = []

        # ── Completeness
        comp = metrics.get("Completeness")
        if comp:
            miss_pct = 100 - comp.score
            if miss_pct > 30:
                recs.append(Recommendation("Completeness", "Critical",
                    "More than 30% of data is missing",
                    "Impute with median/mode or use KNN imputation. Consider dropping columns >60% missing.",
                    min(miss_pct * 0.4, 20)))
            elif miss_pct > 10:
                recs.append(Recommendation("Completeness", "High",
                    f"{miss_pct:.1f}% missing values detected",
                    "Apply mean/median imputation for numeric columns; mode for categoricals.",
                    min(miss_pct * 0.35, 15)))
            elif miss_pct > 0:
                recs.append(Recommendation("Completeness", "Low",
                    f"Minor missing values ({miss_pct:.1f}%)",
                    "Drop rows with missing target variable; impute others.",
                    miss_pct * 0.3))

        # ── Consistency
        cons = metrics.get("Consistency")
        if cons and cons.details.get("duplicate_rows", 0) > 0:
            dup_pct = cons.details["duplicate_pct"]
            sev = "Critical" if dup_pct > 10 else "High" if dup_pct > 3 else "Medium"
            recs.append(Recommendation("Consistency", sev,
                f"{cons.details['duplicate_rows']} duplicate rows ({dup_pct:.1f}%)",
                "Run df.drop_duplicates() to remove exact duplicates. Investigate near-duplicates with fuzzy matching.",
                min(dup_pct * 1.2, 20)))

        if cons and cons.details.get("mixed_type_columns"):
            recs.append(Recommendation("Consistency", "High",
                f"Mixed data types in {cons.details['mixed_type_columns']}",
                "Standardize column types; use pd.to_numeric(errors='coerce') for numeric columns.",
                len(cons.details["mixed_type_columns"]) * 4))

        if cons and cons.details.get("whitespace_columns"):
            recs.append(Recommendation("Consistency", "Low",
                "Whitespace detected in string columns",
                "Apply df[col] = df[col].str.strip() to all object columns.",
                2))

        # ── Accuracy
        acc = metrics.get("Accuracy")
        if acc:
            outlier_cols = {k: v for k, v in acc.details.get("outlier_percentage_per_column", {}).items() if v > 5}
            for col, pct in outlier_cols.items():
                sev = "Critical" if pct > 20 else "High" if pct > 10 else "Medium"
                recs.append(Recommendation("Accuracy", sev,
                    f"Column '{col}' has {pct:.1f}% outliers",
                    f"Cap outliers using IQR method: Q1-1.5*IQR to Q3+1.5*IQR, or use robust scalers (RobustScaler).",
                    min(pct * 0.5, 12)))

        # ── Bias
        bias = metrics.get("Bias")
        if bias:
            for col, info in bias.details.get("category_bias", {}).items():
                if info["imbalance_ratio"] > 10:
                    recs.append(Recommendation("Bias", "High",
                        f"Column '{col}' severely imbalanced (dominant: {info['dominant_pct']}%)",
                        "Apply SMOTE oversampling, class weighting, or stratified sampling to address imbalance.",
                        10))
                elif info["imbalance_ratio"] > 5:
                    recs.append(Recommendation("Bias", "Medium",
                        f"Column '{col}' moderately imbalanced",
                        "Use stratified train/test splits and consider class_weight='balanced' in models.",
                        5))

        # ── Reliability
        rel = metrics.get("Reliability")
        if rel:
            if rel.details.get("constant_columns"):
                recs.append(Recommendation("Reliability", "High",
                    f"Constant columns found: {rel.details['constant_columns']}",
                    "Drop constant columns — they provide zero information for ML models.",
                    len(rel.details["constant_columns"]) * 4))
            if rel.details.get("id_like_columns"):
                recs.append(Recommendation("Reliability", "Medium",
                    f"ID-like columns detected: {rel.details['id_like_columns']}",
                    "Exclude ID/key columns from feature sets to prevent data leakage.",
                    len(rel.details["id_like_columns"]) * 3))
            if rel.details.get("row_count", 1000) < 200:
                recs.append(Recommendation("Reliability", "High",
                    "Dataset has too few rows for robust analytics",
                    "Collect more data or apply data augmentation techniques.",
                    15))

        # Sort by severity
        recs.sort(key=lambda r: self.SEVERITY_ORDER.get(r.severity, 9))
        return recs


# ─────────────────────────────────────────────────────────────
# Trust Score Calculator
# ─────────────────────────────────────────────────────────────

class TrustScoreCalculator:
    """Aggregates all metric scores into a unified trust score."""

    RISK_THRESHOLDS = {
        "Safe":           (75, 100),
        "Moderate Risk":  (50, 74.99),
        "High Risk":      (0,  49.99),
    }

    def calculate(self, metrics: Dict[str, MetricResult]) -> Tuple[float, str]:
        total_weight = sum(m.weight for m in metrics.values())
        trust_score = sum(m.weighted_score for m in metrics.values()) / total_weight if total_weight else 0
        trust_score = round(trust_score, 2)

        risk_label = "High Risk"
        for label, (lo, hi) in self.RISK_THRESHOLDS.items():
            if lo <= trust_score <= hi:
                risk_label = label
                break

        return trust_score, risk_label


# ─────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────

class DataTrustEngine:
    """
    Master engine — runs all metrics, generates trust score and full report.
    """

    def __init__(self):
        self.metrics_suite = {
            "Completeness": CompletenessMetric(),
            "Consistency":  ConsistencyMetric(),
            "Accuracy":     AccuracyMetric(),
            "Bias":         BiasMetric(),
            "Reliability":  ReliabilityMetric(),
        }
        self.profiler    = ColumnProfiler()
        self.recommender = RecommendationEngine()
        self.scorer      = TrustScoreCalculator()

    def evaluate(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> TrustReport:
        """Run full evaluation pipeline on a DataFrame."""
        print(f"\n{'='*60}")
        print(f"  Evaluating: {dataset_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        print(f"{'='*60}")

        # 1. Run all metrics
        metrics = {}
        for name, metric in self.metrics_suite.items():
            result = metric.compute(df)
            metrics[name] = result
            bar = "█" * int(result.score // 5) + "░" * (20 - int(result.score // 5))
            print(f"  {name:<15} [{bar}] {result.score:6.1f}/100  (weight {result.weight:.0%})")

        # 2. Trust score
        trust_score, risk_label = self.scorer.calculate(metrics)

        # 3. Column profiles
        col_profiles = self.profiler.profile(df)

        # 4. Recommendations
        recs = self.recommender.generate(metrics)

        # 5. Summary
        risk_emoji = {"Safe": "✅", "Moderate Risk": "⚠️", "High Risk": "🔴"}
        summary = (
            f"{risk_emoji.get(risk_label, '')} Trust Score: {trust_score}/100 — {risk_label}. "
            f"{len(recs)} recommendations generated."
        )
        print(f"\n  {summary}\n")

        return TrustReport(
            dataset_name=dataset_name,
            evaluated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            shape=df.shape,
            trust_score=trust_score,
            risk_label=risk_label,
            metrics=metrics,
            recommendations=recs,
            column_profiles=col_profiles,
            summary=summary
        )

    def compare(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare multiple datasets and rank them by trust score."""
        records = []
        for name, df in datasets.items():
            report = self.evaluate(df, name)
            row = {
                "Dataset":     name,
                "Trust Score": report.trust_score,
                "Risk":        report.risk_label,
                "Rows":        df.shape[0],
                "Columns":     df.shape[1],
            }
            for m_name, m_result in report.metrics.items():
                row[m_name] = m_result.score
            row["Recommendations"] = len(report.recommendations)
            records.append(row)

        comparison = pd.DataFrame(records).sort_values("Trust Score", ascending=False)
        comparison["Rank"] = range(1, len(comparison) + 1)
        return comparison[["Rank"] + [c for c in comparison.columns if c != "Rank"]]
