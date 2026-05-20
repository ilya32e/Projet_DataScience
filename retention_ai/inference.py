from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from retention_ai.config import (
    FINAL_MODEL_BUNDLE_PATH,
    LEADERBOARD_PATH,
    MODELS_DIR,
    SCHEMA_PATH,
)
from retention_ai.data import load_json


def load_schema() -> dict[str, Any]:
    return load_json(SCHEMA_PATH)


def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv(LEADERBOARD_PATH)


def available_models() -> list[str]:
    leaderboard = load_leaderboard()
    return leaderboard["name"].tolist()


def load_bundle(model_name: str | None = None) -> dict[str, Any]:
    if model_name is None:
        return joblib.load(FINAL_MODEL_BUNDLE_PATH)

    bundle_path = MODELS_DIR / f"{model_name}_bundle.joblib"
    if not bundle_path.exists():
        raise ValueError(f"Modele inconnu: {model_name}")
    return joblib.load(bundle_path)


def _coerce_record(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    defaults = schema["default_values"]
    numeric_details = schema["numeric_details"]

    for column in schema["feature_columns"]:
        value = record.get(column, defaults[column])
        if column in schema["numeric_columns"]:
            dtype = numeric_details[column]["dtype"]
            if "int" in dtype:
                payload[column] = int(round(float(value)))
            else:
                payload[column] = float(value)
        else:
            payload[column] = str(value)
    return payload


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= max(0.75, threshold + 0.10):
        return "Critique"
    if probability >= threshold:
        return "Élevé"
    if probability >= max(0.35, threshold * 0.70):
        return "Modéré"
    return "Faible"


def _recommended_action(payload: dict[str, Any], probability: float, threshold: float) -> str:
    if probability >= max(0.75, threshold + 0.10):
        if payload["payment_failures"] >= 2:
            return "Prioriser un contact humain et une action de recouvrement douce."
        if payload["support_tickets"] >= 3:
            return "Déclencher une cellule support premium et une offre de rétention."
        return "Lancer une campagne de rétention personnalisée sous 24h."
    if probability >= threshold:
        return "Prévoir une relance CRM ciblée avec avantage commercial limité."
    if payload["nps_score"] < 0 or payload["csat_score"] <= 2:
        return "Surveiller le compte et corriger rapidement l'expérience client."
    return "Maintenir le compte dans le portefeuille standard avec suivi mensuel."


def predict_record(record: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    schema = load_schema()
    payload = _coerce_record(record, schema)
    bundle = load_bundle(model_name)
    frame = pd.DataFrame([payload], columns=schema["feature_columns"])
    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    predicted_class = int(probability >= threshold)

    result = {
        "model_name": bundle["name"],
        "model_label": bundle["label"],
        "threshold": threshold,
        "churn_probability": probability,
        "predicted_class": predicted_class,
        "risk_level": _risk_level(probability, threshold),
        "expected_monthly_loss": float(payload["monthly_fee"] * probability),
        "expected_revenue_at_risk": float(payload["total_revenue"] * probability),
        "recommended_action": _recommended_action(payload, probability, threshold),
        "input_record": payload,
    }
    return result
