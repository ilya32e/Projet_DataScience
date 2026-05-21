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
from retention_ai.config import MODELS_DIR, TARGET_COLUMN, ID_COLUMN
from retention_ai.data import load_dataset, load_json
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
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def get_shap_global() -> pd.DataFrame:
    """SHAP global importance — top 15 features, calculé sur 300 clients test."""
    import joblib, numpy as np, shap
    from sklearn.model_selection import train_test_split

    bundle = joblib.load(MODELS_DIR / "final_model_bundle.joblib")
    pipeline = bundle["pipeline"]

    df = load_dataset()
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    fe   = pipeline.named_steps["features"]
    prep = pipeline.named_steps["preprocess"]
    gb   = pipeline.named_steps["model"]

    X_proc = prep.transform(fe.transform(X_te))
    feat_names = list(prep.get_feature_names_out())

    sv = shap.TreeExplainer(gb).shap_values(X_proc[:300])
    return (
        pd.DataFrame({"feature": feat_names, "shap_mean": np.abs(sv).mean(axis=0)})
        .sort_values("shap_mean", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


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

    st.markdown("---")
    with st.spinner("Calcul de l'explication SHAP..."):
        shap_local = _shap_local(record)
    if shap_local is not None:
        _render_shap_local(shap_local)

    with st.expander("Détails techniques"):
        st.json(
            {
                "modele": result["model_label"],
                "classe_predite": result["predicted_class"],
                "seuil_operationnel": round(float(result["threshold"]), 4),
            }
        )


def _reg_leaderboard_chart(leaderboard: pd.DataFrame) -> None:
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
    fig.update_layout(height=300, legend_title="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    cols = [c for c in col_map if c in leaderboard.columns]
    st.dataframe(leaderboard[cols].rename(columns=col_map).round(4),
                 use_container_width=True, hide_index=True)


def _clf_leaderboard_chart(leaderboard: pd.DataFrame) -> None:
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
    fig.update_layout(height=300, legend_title="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    cols = [c for c in col_map if c in leaderboard.columns]
    st.dataframe(leaderboard[cols].rename(columns=col_map).round(4),
                 use_container_width=True, hide_index=True)


def _importance_chart(importance: pd.DataFrame) -> None:
    top = importance.head(10).sort_values("importance_mean")
    fig = px.bar(top, x="importance_mean", y="feature", orientation="h",
                 color="importance_mean",
                 color_continuous_scale=["#4cc9f0", "#256f8f", "#ff7b72"])
    apply_plotly_theme(fig)
    fig.update_layout(height=340, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_analyse_zone(df: pd.DataFrame, scored: pd.DataFrame) -> None:
    sec_corr, sec_shap, sec_err = st.tabs([
        "Corrélation des variables",
        "SHAP — Importance globale",
        "Analyse des erreurs",
    ])

    # ── Corrélation ──────────────────────────────────────────────────────────
    with sec_corr:
        st.markdown(
            "<p style='color:#94a3b8;font-size:13px;margin-bottom:14px;'>"
            "Coefficients de corrélation de Pearson entre toutes les variables numériques. "
            "Plus la valeur est proche de ±1, plus la relation est forte.</p>",
            unsafe_allow_html=True,
        )
        num_cols = df.select_dtypes(include="number").drop(columns=[TARGET_COLUMN], errors="ignore")
        corr = num_cols.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(title="r"),
        ))
        apply_plotly_theme(fig)
        fig.update_layout(
            height=620,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top corrélations avec la cible
        st.markdown("**Corrélations les plus fortes avec le churn**")
        target_corr = (
            num_cols.assign(churn=df[TARGET_COLUMN])
            .corr()["churn"]
            .drop("churn")
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        target_corr.columns = ["Variable", "|r| avec churn"]
        st.dataframe(target_corr.round(4), use_container_width=True, hide_index=True)

    # ── SHAP global ──────────────────────────────────────────────────────────
    with sec_shap:
        st.markdown(
            "<p style='color:#94a3b8;font-size:13px;margin-bottom:14px;'>"
            "Impact moyen de chaque variable sur la probabilité de churn, "
            "calculé via SHAP TreeExplainer sur 300 clients du jeu de test. "
            "Plus la valeur SHAP est élevée, plus la variable influence la prédiction.</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("Calcul des valeurs SHAP en cours..."):
            shap_df = get_shap_global()

        top = shap_df.sort_values("shap_mean")
        fig = px.bar(
            top, x="shap_mean", y="feature", orientation="h",
            color="shap_mean",
            color_continuous_scale=["#1e3a5f", "#4cc9f0", "#ff7b72"],
            labels={"shap_mean": "SHAP moyen |valeur|", "feature": "Variable"},
        )
        apply_plotly_theme(fig)
        fig.update_layout(
            height=420, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Tableau détaillé**")
        display_df = shap_df.copy()
        display_df.columns = ["Variable", "SHAP moyen |valeur|"]
        st.dataframe(display_df.round(5), use_container_width=True, hide_index=True)

        _note(
            "SHAP complète la permutation importance : il mesure la contribution de chaque "
            "variable à chaque prédiction individuelle, puis en fait la moyenne. "
            "Les deux méthodes convergent sur les mêmes variables clés.", "info",
        )

    # ── Analyse des erreurs ──────────────────────────────────────────────────
    with sec_err:
        merged = scored.copy()
        merged["actual_churn"] = df[TARGET_COLUMN].values

        churners     = merged[merged["actual_churn"] == 1]["churn_probability"]
        non_churners = merged[merged["actual_churn"] == 0]["churn_probability"]

        st.markdown("**Distribution des probabilités prédites par classe réelle**")
        st.markdown(
            "<p style='color:#94a3b8;font-size:13px;margin-bottom:10px;'>"
            "Un bon modèle sépare clairement les deux distributions. "
            "Le chevauchement correspond aux cas difficiles.</p>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=non_churners, name="Non-churner (réel)", nbinsx=40,
            marker_color="#4cc9f0", opacity=0.7,
        ))
        fig.add_trace(go.Histogram(
            x=churners, name="Churner (réel)", nbinsx=40,
            marker_color="#ff7b72", opacity=0.7,
        ))
        apply_plotly_theme(fig)
        fig.update_layout(
            barmode="overlay", height=340,
            legend=dict(orientation="h", y=1.05),
            xaxis_title="Probabilité de churn prédite",
            yaxis_title="Nombre de clients",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_fn, col_fp = st.columns(2)
        with col_fn:
            st.markdown("**Faux négatifs — churners manqués**")
            st.markdown(
                "<p style='color:#94a3b8;font-size:12px;'>Clients réellement partis "
                "que le modèle n'a pas détectés. Triés par revenu à risque décroissant.</p>",
                unsafe_allow_html=True,
            )
            fn = merged[
                (merged["actual_churn"] == 1) & (merged["predicted_churn"] == 0)
            ].sort_values("expected_monthly_loss", ascending=False).head(10)
            cols_show = [c for c in
                ["customer_id", "churn_probability", "expected_monthly_loss"]
                if c in fn.columns]
            st.dataframe(fn[cols_show].round(3), use_container_width=True, hide_index=True)

        with col_fp:
            st.markdown("**Faux positifs — fausses alarmes**")
            st.markdown(
                "<p style='color:#94a3b8;font-size:12px;'>Clients signalés à risque "
                "mais qui ne sont pas partis. Coût : efforts de rétention inutiles.</p>",
                unsafe_allow_html=True,
            )
            fp = merged[
                (merged["actual_churn"] == 0) & (merged["predicted_churn"] == 1)
            ].sort_values("churn_probability", ascending=False).head(10)
            cols_show = [c for c in
                ["customer_id", "churn_probability", "expected_monthly_loss"]
                if c in fp.columns]
            st.dataframe(fp[cols_show].round(3), use_container_width=True, hide_index=True)

        st.markdown("---")
        tp = int(((merged["actual_churn"] == 1) & (merged["predicted_churn"] == 1)).sum())
        fp_n = int(((merged["actual_churn"] == 0) & (merged["predicted_churn"] == 1)).sum())
        fn_n = int(((merged["actual_churn"] == 1) & (merged["predicted_churn"] == 0)).sum())
        tn = int(((merged["actual_churn"] == 0) & (merged["predicted_churn"] == 0)).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vrais Positifs (TP)", tp, help="Churners correctement détectés")
        c2.metric("Faux Positifs (FP)", fp_n, help="Non-churners signalés à tort")
        c3.metric("Faux Négatifs (FN)", fn_n, help="Churners manqués — les plus coûteux")
        c4.metric("Vrais Négatifs (TN)", tn, help="Non-churners correctement classés")


def _shap_local(record: dict[str, Any]) -> pd.DataFrame | None:
    """Calcule les SHAP values pour un seul client."""
    try:
        import joblib, numpy as np, shap
        bundle = joblib.load(MODELS_DIR / "final_model_bundle.joblib")
        pipeline = bundle["pipeline"]
        fe   = pipeline.named_steps["features"]
        prep = pipeline.named_steps["preprocess"]
        gb   = pipeline.named_steps["model"]
        X_proc = prep.transform(fe.transform(pd.DataFrame([record])))
        feat_names = list(prep.get_feature_names_out())
        sv = shap.TreeExplainer(gb).shap_values(X_proc)[0]
        df = pd.DataFrame({"feature": feat_names, "shap": sv})
        df["abs"] = df["shap"].abs()
        df["sens"] = df["shap"].apply(lambda v: "Vers churn" if v > 0 else "Vers rétention")
        return df.sort_values("abs", ascending=False).head(10).reset_index(drop=True)
    except Exception:
        return None


def _render_shap_local(shap_df: pd.DataFrame) -> None:
    st.markdown("**Explication de cette prédiction (SHAP)**")
    st.markdown(
        "<p style='color:#94a3b8;font-size:12px;margin-bottom:8px;'>"
        "Contribution de chaque variable à la probabilité prédite pour ce client.</p>",
        unsafe_allow_html=True,
    )
    fig = px.bar(
        shap_df.sort_values("shap"),
        x="shap", y="feature", orientation="h",
        color="sens",
        color_discrete_map={"Vers churn": "#ff7b72", "Vers rétention": "#4cc9f0"},
        labels={"shap": "Valeur SHAP", "feature": "Variable", "sens": ""},
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    fig.add_vline(x=0, line_color="#64748b", line_width=1)
    st.plotly_chart(fig, use_container_width=True)


def _task_header(task_type: str, target: str, features: str) -> None:
    st.markdown(
        f"""
        <div style="
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            gap: 0;
            border: 1px solid #1e293b;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        ">
            <div style="background:#0f172a; padding: 12px 18px; border-right: 1px solid #1e293b;">
                <div style="font-size:10px; color:#64748b; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:4px;">Type</div>
                <div style="font-size:14px; color:#e2e8f0; font-weight:600;">{task_type}</div>
            </div>
            <div style="background:#0f172a; padding: 12px 18px; border-right: 1px solid #1e293b;">
                <div style="font-size:10px; color:#64748b; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:4px;">Variable cible</div>
                <div style="font-size:14px; color:#e2e8f0; font-family:monospace;">{target}</div>
            </div>
            <div style="background:#0f172a; padding: 12px 18px;">
                <div style="font-size:10px; color:#64748b; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:4px;">Features utilisées</div>
                <div style="font-size:14px; color:#e2e8f0;">{features}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _score_card(label: str, value: str, sub: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: #0f172a;
            border: 1px solid #1e293b;
            border-top: 3px solid {color};
            border-radius: 8px;
            padding: 18px 20px;
            text-align: center;
        ">
            <div style="font-size:11px; color:#64748b; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:6px;">{label}</div>
            <div style="font-size:38px; font-weight:800; color:{color};
                        line-height:1.1;">{value}</div>
            <div style="font-size:12px; color:#475569; margin-top:6px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _explanation(rows: list[tuple[str, str]]) -> None:
    """Bloc d'explication structuré : liste de (titre, texte)."""
    items_html = "".join(
        f"""
        <div style="margin-bottom:12px;">
            <div style="font-size:12px; color:#94a3b8; font-weight:700;
                        text-transform:uppercase; letter-spacing:.9px;
                        margin-bottom:3px;">{title}</div>
            <div style="font-size:13.5px; color:#cbd5e1; line-height:1.65;">{body}</div>
        </div>
        """
        for title, body in rows
    )
    st.markdown(
        f"""
        <div style="
            background:#0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 18px 22px;
            margin: 10px 0 16px 0;
        ">
            <div style="font-size:11px; color:#64748b; font-weight:700;
                        text-transform:uppercase; letter-spacing:1px;
                        margin-bottom:14px; border-bottom:1px solid #1e293b;
                        padding-bottom:8px;">Comprendre ces résultats</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _note(text: str, kind: str = "info") -> None:
    colors = {"info": "#3b82f6", "warning": "#f59e0b", "result": "#64748b"}
    labels = {"info": "Résultat", "warning": "Attention", "result": "Conclusion"}
    c = colors.get(kind, "#3b82f6")
    lbl = labels.get(kind, "Note")
    st.markdown(
        f"""
        <div style="
            border-left: 3px solid {c};
            background: #0f172a;
            border-radius: 0 6px 6px 0;
            padding: 11px 16px;
            margin: 6px 0;
        ">
            <span style="font-size:11px; color:{c}; font-weight:700;
                         text-transform:uppercase; letter-spacing:.8px;">{lbl} — </span>
            <span style="color:#cbd5e1; font-size:13.5px;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_secondary_zone(secondary: dict[str, pd.DataFrame]) -> None:
    tab_rar, tab_clv, tab_eng_reg, tab_eng_clf = st.tabs([
        "Revenu à Risque",
        "CLV",
        "Score d'Engagement",
        "Catégorie Engagement",
    ])

    # ── Revenu à Risque ──────────────────────────────────────────────────────
    with tab_rar:
        lb = secondary["revenue_at_risk"]
        best_row = lb.loc[lb["test_r2"].idxmax()]

        _task_header(
            "Régression",
            "monthly_fee × churn_probability",
            "Toutes les features",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            _score_card("R² test", f"{best_row['test_r2']:.3f}", "meilleur modèle", "#4cc9f0")
        with c2:
            _score_card("CV R²", f"{best_row['cv_r2_mean']:.3f}", "validation croisée", "#4cc9f0")
        with c3:
            _score_card("Meilleur modèle", best_row["label"], best_row["family"], "#4cc9f0")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _note(
            "Le Random Forest atteint R²=0.914 — le profil client prédit bien "
            "le revenu financier exposé par client.", "info",
        )

        _explanation([
            ("Objectif",
             "Prédire combien l'entreprise risque de perdre chaque mois par client, "
             "en combinant sa probabilité de churn avec son tarif mensuel."),
            ("Random Forest — R² 0.914",
             "Meilleur modèle. Il capte les interactions non-linéaires entre le tarif "
             "mensuel, l'ancienneté et la probabilité de churn. RMSE de 3,85 € : "
             "en moyenne, l'erreur de prédiction est inférieure à 4 € par client."),
            ("Gradient Boosting — R² 0.904",
             "Très proche du Random Forest. Légèrement moins précis sur ce jeu de test "
             "mais CV R² similaire (0.930) — les deux modèles sont stables."),
            ("Ridge (linéaire) — R² 0.532",
             "La régression linéaire décroche. Le revenu à risque dépend "
             "d'interactions entre variables que le modèle linéaire ne peut pas capturer. "
             "RMSE de 9 € contre 3,85 € pour le RF."),
        ])

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Comparaison des 4 modèles**")
            _reg_leaderboard_chart(lb)
        with col_right:
            st.markdown("**Variables les plus décisives**")
            _importance_chart(secondary["rar_importance"])

    # ── CLV ──────────────────────────────────────────────────────────────────
    with tab_clv:
        lb = secondary["clv"]
        best_row = lb.loc[lb["test_r2"].idxmax()]

        _task_header(
            "Régression",
            "total_revenue",
            "Toutes sauf total_revenue",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            _score_card("R² test", f"{best_row['test_r2']:.4f}", "meilleur modèle", "#4cc9f0")
        with c2:
            _score_card("CV R²", f"{best_row['cv_r2_mean']:.4f}", "validation croisée", "#4cc9f0")
        with c3:
            _score_card("Meilleur modèle", best_row["label"], best_row["family"], "#4cc9f0")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _note(
            "R² ≈ 0.9999 : dans ce dataset synthétique, total_revenue = tenure_months × monthly_fee. "
            "La relation est quasi-linéaire et déterministe.", "warning",
        )
        _note(
            "Ce n'est pas le modèle qui est exceptionnel — c'est la variable cible "
            "qui est structurellement calculable depuis les features.", "result",
        )

        _explanation([
            ("Objectif",
             "Prédire le revenu total cumulé d'un client (sa valeur vie) "
             "sans utiliser la colonne total_revenue directement."),
            ("Pourquoi R² ≈ 1 ?",
             "Dans ce dataset synthétique, total_revenue est construit comme "
             "tenure_months × monthly_fee. Le Random Forest retrouve cette formule "
             "presque exactement — RMSE de seulement 1,95 €."),
            ("Random Forest vs Ridge",
             "Le Random Forest (RMSE 1,95 €) écrase la régression Ridge (RMSE 400 €). "
             "Même si la relation semble linéaire, il existe des effets de seuil "
             "et des interactions que Ridge ne modélise pas."),
            ("Limite importante",
             "Ce R² quasi-parfait ne signifie pas que le modèle est bon en général. "
             "Sur des données réelles, le CLV dépend d'upsells, de promotions, "
             "d'événements extérieurs — et serait bien plus difficile à prédire."),
        ])

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Comparaison des 4 modèles**")
            _reg_leaderboard_chart(lb)
        with col_right:
            st.markdown("**Variables les plus décisives**")
            _importance_chart(secondary["clv_importance"])

    # ── Score d'Engagement ───────────────────────────────────────────────────
    with tab_eng_reg:
        lb = secondary["engagement_regression"]
        best_row = lb.loc[lb["test_r2"].idxmax()]
        best_r2 = max(best_row["test_r2"], 0)

        _task_header(
            "Régression",
            "engagement_score",
            "Profil + finance + support (sans usage)",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            _score_card("R² test", f"{best_r2:.3f}", "meilleur modèle", "#ff7b72")
        with c2:
            _score_card("CV R²", f"{max(best_row['cv_r2_mean'], 0):.3f}",
                        "validation croisée", "#ff7b72")
        with c3:
            _score_card("Tous les modèles", "R² ≈ 0", "convergent", "#ff7b72")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _note(
            "L'engagement comportemental (logins, sessions, activité…) n'est pas prédictible "
            "depuis le profil administratif du client.", "warning",
        )
        _note(
            "Résultat négatif mais valide : l'engagement est intrinsèque au client, "
            "pas lisible dans son contrat, sa région ou son historique de paiement.", "result",
        )

        _explanation([
            ("Objectif",
             "Prédire le score d'engagement d'un client (combinaison pondérée de ses 6 métriques "
             "d'usage) en utilisant uniquement son profil, ses données financières et son support — "
             "sans les colonnes d'usage."),
            ("Tous les modèles : R² ≤ 0",
             "Ridge : -0.004 · Gradient Boosting : -0.015 · Random Forest : -0.025 · MLP : -0.162. "
             "Un R² négatif signifie que le modèle est moins précis qu'une simple moyenne. "
             "Aucun modèle n'apprend quoi que ce soit d'utile."),
            ("Pourquoi ce résultat ?",
             "L'engagement dépend exclusivement des comportements d'usage — logins, sessions, "
             "jours actifs. Ces informations ne sont pas contenues dans le profil administratif "
             "(âge, région, contrat, paiements). Les deux mondes sont indépendants."),
            ("Ce que ça veut dire concrètement",
             "Pour savoir si un client est engagé, il faut observer comment il utilise "
             "la plateforme. Aucun raccourci par le profil n'est possible. "
             "Ce résultat guide la stratégie de collecte de données."),
        ])

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Comparaison des 4 modèles**")
            _reg_leaderboard_chart(lb)
        with col_right:
            st.markdown("**Importance des variables (impact quasi nul)**")
            _importance_chart(secondary["engagement_importance"])

    # ── Catégorie d'Engagement ───────────────────────────────────────────────
    with tab_eng_clf:
        lb = secondary["engagement_classification"]
        best_row = lb.loc[lb["test_f1_macro"].idxmax()]

        _task_header(
            "Classification multi-classe",
            "Faible / Moyen / Fort",
            "Profil + finance + support (sans usage)",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            _score_card("F1 Macro test", f"{best_row['test_f1_macro']:.3f}",
                        "meilleur modèle", "#4cc9f0")
        with c2:
            _score_card("CV F1 Macro", f"{best_row['cv_f1_macro_mean']:.3f}",
                        "validation croisée", "#4cc9f0")
        with c3:
            _score_card("Baseline aléatoire", "0.333", "référence 3 classes", "#64748b")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _note(
            "F1 ≈ 0.32 : les modèles font à peine mieux qu'un classifieur aléatoire. "
            "Les 3 classes ne sont pas séparables depuis le profil.", "warning",
        )
        _note(
            "Cohérent avec la régression d'engagement : les deux tâches confirment "
            "que l'engagement ne peut pas être prédit sans données d'usage.", "result",
        )

        _explanation([
            ("Objectif",
             "Classer chaque client dans une catégorie d'engagement : Faible, Moyen ou Fort, "
             "en se basant uniquement sur le profil — sans les données d'usage."),
            ("Piège : l'Accuracy trompeuse",
             "Random Forest et Gradient Boosting affichent 87 % d'accuracy — "
             "mais F1 Macro à 0.31. Explication : ils prédisent presque toujours "
             'la classe majoritaire "Moyen" et ignorent les autres. '
             "L'accuracy monte mécaniquement sans que le modèle apprenne quoi que ce soit."),
            ("F1 Macro : la vraie métrique",
             "Sur 3 classes équilibrées, un classifieur aléatoire obtient F1 ≈ 0.333. "
             "Tous nos modèles sont en dessous ou à ce niveau — ils ne font pas mieux "
             "que le hasard. La Logistic Regression à 38 % d'accuracy confirme l'échec complet."),
            ("Conclusion commune avec la tâche précédente",
             "Régression (R² ≈ 0) et classification (F1 ≈ 0.33) disent la même chose : "
             "l'engagement ne se prédit pas sans données comportementales. "
             "C'est un résultat négatif cohérent et instructif."),
        ])

        st.markdown("---")
        st.markdown("**Comparaison des 4 modèles**")
        _clf_leaderboard_chart(lb)


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

    df = get_dataset()

    render_header(overview)
    render_kpis(scored)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Pilotage", "Portefeuille", "Simulation",
        "Tâches secondaires", "Analyse",
    ])
    with tab1:
        render_model_zone(leaderboard)
        render_feature_zone(feature_importance)
    with tab2:
        render_portfolio_zone(scored)
    with tab3:
        render_prediction_zone(schema)
    with tab4:
        if secondary is None:
            st.info("Les tâches secondaires ne sont pas encore entraînées. "
                    "Lance `python -m retention_ai.train` pour les générer.")
        else:
            render_secondary_zone(secondary)
    with tab5:
        render_analyse_zone(df, scored)


if __name__ == "__main__":
    main()
