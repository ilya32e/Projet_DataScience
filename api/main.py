from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retention_ai.inference import available_models, load_bundle, predict_record

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Retention Risk API",
    description="API REST pour scorer le risque de churn d'un client.",
    version="1.0.0",
    docs_url=None,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui() -> HTMLResponse:
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
        swagger_ui_parameters={
            "syntaxHighlight.theme": "obsidian",
            "defaultModelsExpandDepth": 1,
            "displayRequestDuration": True,
            "tryItOutEnabled": True,
        },
    )
    html = response.body.decode("utf-8").replace(
        "</head>",
        '  <link rel="stylesheet" type="text/css" href="/static/swagger-dark.css">\n</head>',
    )
    headers = {key: value for key, value in response.headers.items() if key.lower() != "content-length"}
    return HTMLResponse(content=html, status_code=response.status_code, headers=headers)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=18, le=90, example=42)
    country: str = Field(..., example="France")
    city: str = Field(..., example="Paris")
    customer_segment: str = Field(..., example="SME")
    tenure_months: int = Field(..., ge=1, example=24)
    signup_channel: str = Field(..., example="Web")
    contract_type: str = Field(..., example="Monthly")
    monthly_logins: int = Field(..., ge=0, example=14)
    weekly_active_days: int = Field(..., ge=0, le=7, example=4)
    avg_session_time: float = Field(..., ge=0, example=18.5)
    features_used: int = Field(..., ge=0, example=4)
    usage_growth_rate: float = Field(..., example=-0.12)
    last_login_days_ago: int = Field(..., ge=0, example=9)
    monthly_fee: int = Field(..., ge=0, example=40)
    total_revenue: int = Field(..., ge=0, example=820)
    payment_method: str = Field(..., example="Card")
    payment_failures: int = Field(..., ge=0, example=1)
    discount_applied: str = Field(..., example="No")
    price_increase_last_3m: str = Field(..., example="Yes")
    support_tickets: int = Field(..., ge=0, example=3)
    avg_resolution_time: float = Field(..., ge=0, example=21.4)
    complaint_type: str = Field(..., example="Billing")
    csat_score: float = Field(..., ge=1, le=5, example=2.0)
    escalations: int = Field(..., ge=0, example=1)
    email_open_rate: float = Field(..., ge=0, le=1, example=0.48)
    marketing_click_rate: float = Field(..., ge=0, le=1, example=0.16)
    nps_score: int = Field(..., ge=-100, le=100, example=-15)
    survey_response: str = Field(..., example="Neutral")
    referral_count: int = Field(..., ge=0, example=1)

    class Config:
        extra = "forbid"


class PredictionRequest(BaseModel):
    model_name: Optional[str] = Field(None, example="random_forest")
    customer: CustomerFeatures

    class Config:
        extra = "forbid"


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "status": "ok",
        "final_model": bundle["name"],
        "available_models": available_models(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "name": bundle["name"],
        "label": bundle["label"],
        "family": bundle["family"],
        "threshold": bundle["threshold"],
        "metrics": bundle["metrics"],
        "available_models": available_models(),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    model_name = request.model_name
    if model_name is not None and model_name not in available_models():
        raise HTTPException(status_code=400, detail=f"Modele inconnu: {model_name}")

    try:
        return predict_record(request.customer.dict(), model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Les artefacts du modele sont introuvables. Lance d'abord l'entrainement.",
        ) from exc
    except Exception as exc:  # pragma: no cover - garde-fou runtime
        raise HTTPException(status_code=500, detail=f"Erreur interne: {exc}") from exc
