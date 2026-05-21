from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from retention_ai.config import (
    CLV_IMPORTANCE_PATH,
    CLV_LEADERBOARD_PATH,
    ENG_CLF_LEADERBOARD_PATH,
    ENG_IMPORTANCE_PATH,
    ENG_REG_LEADERBOARD_PATH,
    FEATURE_IMPORTANCE_PATH,
    OVERVIEW_PATH,
    RAR_IMPORTANCE_PATH,
    RAR_LEADERBOARD_PATH,
    SCORED_CUSTOMERS_PATH,
)
from retention_ai.data import load_json
from retention_ai.inference import available_models, load_leaderboard, load_schema, predict_record

st.set_page_config(
    page_title="Pulse Retention AI",
    page_icon="📉",
    layout="wide",
)

PLOT_BG = "#0f172a"
TEXT_COLOR = "#e5eef7"
GRID_COLOR = "rgba(148, 163, 184, 0.18)"


@st.cache_data
def get_schema() -> dict[str, Any]:
    return load_schema()


@st.cache_data
def get_leaderboard() -> pd.DataFrame:
    return load_leaderboard()


@st.cache_data
def get_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def get_scored_customers() -> pd.DataFrame:
    return pd.read_csv(SCORED_CUSTOMERS_PATH)


@st.cache_data
def get_overview() -> dict[str, Any]:
    return load_json(OVERVIEW_PATH)


@st.cache_data
def get_secondary() -> dict[str, pd.DataFrame] | None:
    try:
        return {
            "revenue_at_risk": pd.read_csv(RAR_LEADERBOARD_PATH),
            "rar_importance": pd.read_csv(RAR_IMPORTANCE_PATH),
            "clv": pd.read_csv(CLV_LEADERBOARD_PATH),
            "clv_importance": pd.read_csv(CLV_IMPORTANCE_PATH),
            "engagement_regression": pd.read_csv(ENG_REG_LEADERBOARD_PATH),
            "engagement_importance": pd.read_csv(ENG_IMPORTANCE_PATH),
            "engagement_classification": pd.read_csv(ENG_CLF_LEADERBOARD_PATH),
        }
    except FileNotFoundError:
        return None


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@500;700&display=swap');
        :root {
            color-scheme: dark;
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(216, 101, 79, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(36, 118, 145, 0.18), transparent 30%),
                linear-gradient(180deg, #040812 0%, #0a1220 40%, #111827 100%);
            color: #e5eef7;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 13, 23, 0.98), rgba(15, 23, 42, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.94)),
                radial-gradient(circle at top right, rgba(216, 101, 79, 0.32), transparent 35%);
            color: white;
            border: 1px solid rgba(148, 163, 184, 0.14);
            box-shadow: 0 24px 70px rgba(2, 6, 23, 0.45);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            font-family: 'Source Serif 4', serif;
            font-size: 2.2rem;
            margin-bottom: 0.2rem;
        }
        .hero p {
            margin-bottom: 0;
            opacity: 0.92;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(17,24,39,0.82));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.26);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #e5eef7 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            color: #cbd5e1;
            padding: 0.45rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(216, 101, 79, 0.20));
            color: #f8fafc !important;
        }
        [data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.64);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        .stTextInput input,
        .stNumberInput input {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #e5eef7 !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
        }
        .stForm {
            background: rgba(2, 6, 23, 0.28);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 20px;
            padding: 1rem;
        }
        .stButton button,
        .stFormSubmitButton button {
            background: linear-gradient(135deg, #256f8f, #d8654f);
            color: #f8fafc;
            border: none;
            border-radius: 999px;
            box-shadow: 0 12px 30px rgba(37, 111, 143, 0.22);
        }
        .stAlert {
            background: rgba(15, 23, 42, 0.86);
            color: #e5eef7;
        }
        [data-testid="stDataFrame"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def render_header(overview: dict[str, Any]) -> None:
    final_model = overview["final_model_label"]
    threshold = overview["final_model_threshold"]
    st.markdown(
        f"""
        <div class="hero">
            <h1>Pulse Retention AI</h1>
            <p>
                Dashboard métier pour suivre le churn, prioriser les clients sensibles et simuler
                l'impact financier d'un scénario client. Modèle final retenu: <strong>{final_model}</strong>
                avec un seuil opérationnel de <strong>{threshold:.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(scored: pd.DataFrame) -> None:
    total_customers = len(scored)
    observed_churn = scored["churn"].mean()
    high_risk_count = int(scored["predicted_churn"].sum())
    expected_monthly_loss = scored["expected_monthly_loss"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients suivis", f"{total_customers:,}".replace(",", " "))
    col2.metric("Churn observé", f"{observed_churn:.1%}")
    col3.metric("Clients à risque", f"{high_risk_count:,}".replace(",", " "))
    col4.metric("Perte mensuelle attendue", f"{expected_monthly_loss:,.0f} €".replace(",", " "))


def render_model_zone(leaderboard: pd.DataFrame) -> None:
    st.subheader("Comparaison des modèles")
    chart_df = leaderboard[["label", "test_pr_auc", "test_f1", "test_recall"]].copy()
    chart_df = chart_df.melt(id_vars="label", var_name="metric", value_name="score")
    chart_df["metric"] = chart_df["metric"].map(
        {
            "test_pr_auc": "PR-AUC",
            "test_f1": "F1",
            "test_recall": "Recall",
        }
    )
    fig = px.bar(
        chart_df,
        x="label",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=["#4cc9f0", "#ff7b72", "#8ecae6"],
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=400, legend_title="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        leaderboard[
            [
                "label",
                "family",
                "threshold",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc",
                "cv_pr_auc_mean",
                "cv_fit_time_mean",
            ]
        ]
        .rename(
            columns={
                "label": "Modèle",
                "family": "Famille",
                "threshold": "Seuil",
                "test_precision": "Precision",
                "test_recall": "Recall",
                "test_f1": "F1",
                "test_roc_auc": "ROC-AUC",
                "test_pr_auc": "PR-AUC",
                "cv_pr_auc_mean": "CV PR-AUC",
                "cv_fit_time_mean": "Temps moyen CV (s)",
            }
        )
        .round(3),
        use_container_width=True,
        hide_index=True,
    )


def render_portfolio_zone(scored: pd.DataFrame) -> None:
    st.subheader("Portefeuille clients")
    risk_df = scored.copy()
    risk_df["risk_bucket"] = pd.cut(
        risk_df["churn_probability"],
        bins=[0, 0.35, 0.60, 1.0],
        labels=["Faible", "Modéré", "Critique"],
        include_lowest=True,
    )
    col1, col2 = st.columns([1.2, 1])

    with col1:
        scatter = px.scatter(
            risk_df,
            x="total_revenue",
            y="churn_probability",
            color="risk_bucket",
            size="expected_revenue_at_risk",
            hover_data=["customer_id", "contract_type", "customer_segment"],
            color_discrete_map={
                "Faible": "#4cc9f0",
                "Modéré": "#f4b860",
                "Critique": "#ff7b72",
            },
        )
        apply_plotly_theme(scatter)
        scatter.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(scatter, use_container_width=True)

    with col2:
        top_risk = (
            risk_df.sort_values("expected_revenue_at_risk", ascending=False)
            .head(12)[
                [
                    "customer_id",
                    "contract_type",
                    "monthly_fee",
                    "total_revenue",
                    "churn_probability",
                    "expected_revenue_at_risk",
                ]
            ]
            .rename(
                columns={
                    "customer_id": "Client",
                    "contract_type": "Contrat",
                    "monthly_fee": "Fee mensuel",
                    "total_revenue": "CA cumulé",
                    "churn_probability": "Proba churn",
                    "expected_revenue_at_risk": "CA à risque",
                }
            )
        )
        st.dataframe(
            top_risk.style.format({"Proba churn": "{:.1%}", "CA à risque": "{:,.0f}"}),
            use_container_width=True,
        )


def render_feature_zone(feature_importance: pd.DataFrame) -> None:
    st.subheader("Variables les plus influentes")
    top_features = feature_importance.head(12).sort_values("importance_mean")
    fig = px.bar(
        top_features,
        x="importance_mean",
        y="feature",
        orientation="h",
        color="importance_mean",
        color_continuous_scale=["#4cc9f0", "#256f8f", "#ff7b72"],
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=430, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


COUNTRY_CITY_MAP: dict[str, str] = {
    "Australia": "Sydney",
    "Bangladesh": "Dhaka",
    "Canada": "Toronto",
    "Germany": "Berlin",
    "India": "Delhi",
    "UK": "London",
    "USA": "New York",
}


def build_record_from_form(
    schema: dict[str, Any], exclude: set[str] | None = None
) -> dict[str, Any]:
    defaults = schema["default_values"]
    groups = schema["field_groups"]
    numeric_details = schema["numeric_details"]
    exclude = exclude or set()
    record: dict[str, Any] = {}

    for group_name, columns in groups.items():
        visible = [c for c in columns if c not in exclude]
        if not visible:
            continue
        with st.expander(group_name, expanded=group_name in {"Profil", "Usage", "Finance"}):
            left, right = st.columns(2)
            for index, column in enumerate(visible):
                target = left if index % 2 == 0 else right
                with target:
                    if column in schema["categorical_columns"]:
                        options = schema["categorical_options"][column]
                        default_index = (
                            options.index(str(defaults[column])) if str(defaults[column]) in options else 0
                        )
                        record[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            options=options,
                            index=default_index,
                            key=f"field_{column}",
                        )
                    else:
                        details = numeric_details[column]
                        is_integer = "int" in details["dtype"]
                        minimum = details["min"]
                        maximum = details["max"]
                        default_value = defaults[column]
                        if is_integer:
                            record[column] = st.slider(
                                column.replace("_", " ").title(),
                                min_value=int(minimum),
                                max_value=int(maximum),
                                value=int(default_value),
                                key=f"field_{column}",
                            )
                        else:
                            step = 0.01 if maximum <= 1 else 0.1
                            record[column] = st.number_input(
                                column.replace("_", " ").title(),
                                min_value=float(minimum),
                                max_value=float(maximum),
                                value=float(default_value),
                                step=step,
                                key=f"field_{column}",
                            )
    return record


def request_prediction(api_mode: bool, api_url: str, record: dict[str, Any], model_name: str | None) -> dict[str, Any]:
    if api_mode:
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/predict",
                json={"model_name": model_name, "customer": record},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            st.warning(f"API indisponible, fallback local utilise. Détail: {exc}")

    return predict_record(record, model_name=model_name)


def render_prediction_zone(schema: dict[str, Any]) -> None:
    st.subheader("Simulateur client")
    st.caption("Modifiez le scénario d'un client et estimez immédiatement son risque de churn.")

    sidebar_mode = st.sidebar.toggle("Utiliser l'API REST", value=False)
    api_url = st.sidebar.text_input("URL API", value="http://127.0.0.1:8000")
    model_options = available_models()
    selected_model = st.sidebar.selectbox("Modèle de scoring", options=model_options, index=0)

    # Country/city sélectionnés hors du formulaire pour permettre la réactivité
    countries = schema["categorical_options"].get("country", list(COUNTRY_CITY_MAP.keys()))
    default_country = schema["default_values"].get("country", countries[0])
    default_idx = countries.index(str(default_country)) if str(default_country) in countries else 0

    with st.expander("Localisation", expanded=True):
        col_country, col_city = st.columns(2)
        with col_country:
            selected_country = st.selectbox("Country", options=countries, index=default_idx, key="loc_country")
        with col_city:
            city = COUNTRY_CITY_MAP.get(selected_country, selected_country)
            st.text_input("City", value=city, disabled=True)

    with st.form("simulation_form"):
        record = build_record_from_form(schema, exclude={"country", "city"})
        submitted = st.form_submit_button("Lancer la prédiction", type="primary")

    if submitted:
        record["country"] = selected_country
        record["city"] = city

    if not submitted:
        return

    result = request_prediction(sidebar_mode, api_url, record, selected_model)  # type: ignore[possibly-undefined]
    probability = float(result["churn_probability"])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probabilité de churn"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4cc9f0"},
                "steps": [
                    {"range": [0, 35], "color": "#16324a"},
                    {"range": [35, 60], "color": "#4d3b1f"},
                    {"range": [60, 100], "color": "#4a1f26"},
                ],
                "threshold": {
                    "line": {"color": "#ff7b72", "width": 4},
                    "thickness": 0.9,
                    "value": float(result["threshold"]) * 100,
                },
            },
        )
    )
    gauge.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
    )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        st.metric("Niveau de risque", result["risk_level"])
        st.metric("Perte mensuelle attendue", f"{result['expected_monthly_loss']:.2f} €")
        st.metric("Revenu cumulé à risque", f"{result['expected_revenue_at_risk']:.2f} €")
        st.info(result["recommended_action"])

    st.json(
        {
            "modele": result["model_label"],
            "classe_predite": result["predicted_class"],
            "seuil_operationnel": round(float(result["threshold"]), 4),
        }
    )


def render_regression_leaderboard(leaderboard: pd.DataFrame) -> None:
    col_map = {
        "label": "Modèle", "family": "Famille",
        "test_rmse": "RMSE", "test_mae": "MAE", "test_r2": "R²",
        "cv_rmse_mean": "CV RMSE", "cv_r2_mean": "CV R²",
    }
    chart_df = leaderboard[["label", "test_r2", "cv_r2_mean"]].melt(
        id_vars="label", var_name="metric", value_name="score"
    )
    chart_df["metric"] = chart_df["metric"].map({"test_r2": "R² test", "cv_r2_mean": "R² CV"})
    fig = px.bar(chart_df, x="label", y="score", color="metric", barmode="group",
                 color_discrete_sequence=["#4cc9f0", "#ff7b72"])
    apply_plotly_theme(fig)
    fig.update_layout(height=340, legend_title="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    cols = [c for c in col_map if c in leaderboard.columns]
    st.dataframe(leaderboard[cols].rename(columns=col_map).round(4),
                 use_container_width=True, hide_index=True)


def render_classification_leaderboard(leaderboard: pd.DataFrame) -> None:
    col_map = {
        "label": "Modèle", "family": "Famille",
        "test_accuracy": "Accuracy", "test_f1_macro": "F1 Macro",
        "cv_f1_macro_mean": "CV F1 Macro", "cv_accuracy_mean": "CV Accuracy",
    }
    chart_df = leaderboard[["label", "test_f1_macro", "cv_f1_macro_mean"]].melt(
        id_vars="label", var_name="metric", value_name="score"
    )
    chart_df["metric"] = chart_df["metric"].map(
        {"test_f1_macro": "F1 Macro test", "cv_f1_macro_mean": "F1 Macro CV"}
    )
    fig = px.bar(chart_df, x="label", y="score", color="metric", barmode="group",
                 color_discrete_sequence=["#4cc9f0", "#ff7b72"])
    apply_plotly_theme(fig)
    fig.update_layout(height=340, legend_title="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    cols = [c for c in col_map if c in leaderboard.columns]
    st.dataframe(leaderboard[cols].rename(columns=col_map).round(4),
                 use_container_width=True, hide_index=True)


def render_importance_chart(importance: pd.DataFrame) -> None:
    top = importance.head(10).sort_values("importance_mean")
    fig = px.bar(top, x="importance_mean", y="feature", orientation="h",
                 color="importance_mean",
                 color_continuous_scale=["#4cc9f0", "#256f8f", "#ff7b72"])
    apply_plotly_theme(fig)
    fig.update_layout(height=360, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_secondary_zone(secondary: dict[str, pd.DataFrame]) -> None:
    TASKS = {
        "Revenu à Risque (Régression)": {
            "lb_key": "revenue_at_risk",
            "imp_key": "rar_importance",
            "type": "regression",
            "description": "Cible : `expected_monthly_loss = monthly_fee × churn_probability`  \n"
                           "Features : toutes (profil, usage, finance, support, marketing).",
        },
        "CLV — Valeur Vie Client (Régression)": {
            "lb_key": "clv",
            "imp_key": "clv_importance",
            "type": "regression",
            "description": "Cible : `total_revenue` (revenu cumulé du client)  \n"
                           "Features : toutes sauf `total_revenue` (retirée pour éviter la fuite).",
        },
        "Score d'Engagement (Régression)": {
            "lb_key": "engagement_regression",
            "imp_key": "engagement_importance",
            "type": "regression",
            "description": "Cible : `engagement_score` (formule pondérée sur les 6 colonnes d'usage)  \n"
                           "Features : profil + finance + support + marketing (sans les colonnes d'usage).",
        },
        "Catégorie d'Engagement (Classification)": {
            "lb_key": "engagement_classification",
            "imp_key": None,
            "type": "classification",
            "description": "Cible : Faible / Moyen / Fort selon l'engagement_score  \n"
                           "Features : mêmes que la régression d'engagement.",
        },
    }

    selected = st.selectbox("Tâche prédictive", list(TASKS.keys()))
    task = TASKS[selected]
    st.caption(task["description"])

    lb = secondary[task["lb_key"]]
    if task["type"] == "regression":
        render_regression_leaderboard(lb)
        if task["imp_key"]:
            st.subheader("Variables les plus influentes (meilleur modèle)")
            render_importance_chart(secondary[task["imp_key"]])
    else:
        render_classification_leaderboard(lb)


def main() -> None:
    apply_theme()

    try:
        schema = get_schema()
        leaderboard = get_leaderboard()
        feature_importance = get_feature_importance()
        scored = get_scored_customers()
        overview = get_overview()
    except FileNotFoundError:
        st.error("Les artefacts sont absents. Lance d'abord `python -m retention_ai.train`.")
        st.stop()

    secondary = get_secondary()

    render_header(overview)
    render_kpis(scored)

    tab1, tab2, tab3, tab4 = st.tabs(["Pilotage", "Portefeuille", "Simulation", "Tâches secondaires"])
    with tab1:
        render_model_zone(leaderboard)
        render_feature_zone(feature_importance)
    with tab2:
        render_portfolio_zone(scored)
    with tab3:
        render_prediction_zone(schema)
    with tab4:
        st.subheader("Tâches Prédictives Secondaires")
        if secondary is None:
            st.info("Les tâches secondaires ne sont pas encore entraînées. "
                    "Lance `python -m retention_ai.train` pour les générer.")
        else:
            render_secondary_zone(secondary)


if __name__ == "__main__":
    main()
