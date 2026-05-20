from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from retention_ai.config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    ID_COLUMN,
    KAGGLE_DATASET_SLUG,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)

FEATURE_GROUPS = {
    "Profil": [
        "gender",
        "age",
        "country",
        "city",
        "customer_segment",
        "signup_channel",
        "contract_type",
    ],
    "Usage": [
        "monthly_logins",
        "weekly_active_days",
        "avg_session_time",
        "features_used",
        "usage_growth_rate",
        "last_login_days_ago",
    ],
    "Finance": [
        "monthly_fee",
        "total_revenue",
        "payment_method",
        "payment_failures",
        "discount_applied",
        "price_increase_last_3m",
    ],
    "Support": [
        "support_tickets",
        "avg_resolution_time",
        "complaint_type",
        "csat_score",
        "escalations",
    ],
    "Marketing": [
        "email_open_rate",
        "marketing_click_rate",
        "nps_score",
        "survey_response",
        "referral_count",
        "tenure_months",
    ],
}


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dataset_available() -> Path:
    ensure_directories()
    if DATASET_PATH.exists():
        return DATASET_PATH

    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            f"Le dataset est introuvable dans {DATASET_PATH} et kagglehub n'est pas installe."
        ) from exc

    download_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    candidates = sorted(download_root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun CSV n'a ete trouve apres le telechargement du dataset {KAGGLE_DATASET_SLUG}."
        )

    shutil.copy2(candidates[0], DATASET_PATH)
    return DATASET_PATH


def load_dataset() -> pd.DataFrame:
    dataset_path = ensure_dataset_available()
    return pd.read_csv(dataset_path)


def _to_python_number(value: Any) -> int | float:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    number = float(value)
    return int(number) if number.is_integer() else number


def build_schema(df: pd.DataFrame) -> dict[str, Any]:
    feature_df = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    categorical_columns = feature_df.select_dtypes(include="object").columns.tolist()
    numeric_columns = [column for column in feature_df.columns if column not in categorical_columns]

    numeric_details: dict[str, dict[str, Any]] = {}
    default_values: dict[str, Any] = {}
    for column in numeric_columns:
        series = feature_df[column]
        median_value = series.median()
        numeric_details[column] = {
            "min": _to_python_number(series.min()),
            "max": _to_python_number(series.max()),
            "median": _to_python_number(median_value),
            "dtype": str(series.dtype),
        }
        default_values[column] = _to_python_number(median_value)

    categorical_options: dict[str, list[str]] = {}
    for column in categorical_columns:
        options = sorted(feature_df[column].dropna().astype(str).unique().tolist())
        categorical_options[column] = options
        mode_value = feature_df[column].mode(dropna=True)
        default_values[column] = str(mode_value.iloc[0]) if not mode_value.empty else options[0]

    schema = {
        "feature_columns": feature_df.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "categorical_options": categorical_options,
        "numeric_details": numeric_details,
        "default_values": default_values,
        "field_groups": FEATURE_GROUPS,
        "dataset_summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_distribution": {
                str(key): int(value)
                for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
            },
            "missing_values_total": int(df.isna().sum().sum()),
        },
    }
    return schema


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

