# Système Multi-Modèles pour la Rétention Client

**Projet Data Science — M1 Data Engineering & AI — EFREI Paris 2025-2026**
Étudiants : Mouradi Iliasse & Kosutic Alexandre | Enseignante : Sarah Malaeb

---

## Présentation

Ce projet construit un système intelligent de rétention client sur un dataset SaaS/abonnement de 10 000 clients. Il couvre cinq tâches prédictives distinctes (une classification principale + quatre tâches secondaires), un dashboard interactif et une API REST.

Dataset : [Customer Churn Prediction Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) — 10 000 clients, 32 variables.

---

## Lancer le projet

### Avec Docker (recommandé)

```bash
docker-compose up --build
```

| Service             | URL                        |
| ------------------- | -------------------------- |
| Dashboard Streamlit | http://localhost:8501      |
| API FastAPI         | http://localhost:8000      |
| Documentation API   | http://localhost:8000/docs |

### En local

```bash
pip install -r requirements.txt
python -m retention_ai.train         # entraîne tous les modèles (churn + tâches secondaires)
streamlit run app/streamlit_app.py   # lance le dashboard
uvicorn api.main:app --reload        # lance l'API
```

> L'entraînement doit être lancé avant le dashboard — il génère les artefacts dans `artifacts/`.

---

## Tâches prédictives

### Tâche principale : Prédiction du Churn (Classification binaire)

| Modèle             | Famille       | PR-AUC | Recall | F1    |
| ------------------- | ------------- | ------ | ------ | ----- |
| Logistic Regression | Baseline      | 0.243  | 0.500  | 0.304 |
| Random Forest       | Ensemble      | 0.298  | 0.819  | 0.388 |
| **Gradient Boosting** | **Boosting** | **0.306** | **0.809** | 0.379 |
| MLP Classifier      | Deep Learning | 0.195  | 0.456  | 0.295 |

Modèle final retenu : **Gradient Boosting** (meilleure PR-AUC test et CV). Seuil opérationnel : 0.447.

### Tâches secondaires

Quatre tâches additionnelles entraînées avec 4 modèles chacune (Ridge/LogReg, Random Forest, Gradient Boosting, MLP) + validation croisée + permutation importance :

| Tâche | Type | Cible | Meilleur R²/F1 |
|-------|------|-------|----------------|
| **Revenu à Risque** | Régression | `monthly_fee × churn_proba` | R² = 0.914 (RF) |
| **CLV** | Régression | `total_revenue` | R² = 0.9999 (RF) |
| **Score d'Engagement** | Régression | score pondéré usage | R² ≈ 0 (tous) |
| **Catégorie d'Engagement** | Classification multi-classe | Faible/Moyen/Fort | F1 macro = 0.32 (MLP) |

**Résultat clé** : l'engagement ne peut pas être prédit à partir du profil/finance/support — c'est un comportement intrinsèque au client.

---

## Dashboard décisionnel (Streamlit)

Cinq onglets :

- **Pilotage** : KPI globaux, comparaison des 4 modèles churn, importance des variables
- **Portefeuille** : clients les plus à risque, scatter revenu vs probabilité de churn
- **Simulation** : saisir le profil d'un client → prédiction en temps réel + explication SHAP locale (top variables qui poussent vers / contre le churn pour ce client spécifique)
- **Tâches secondaires** : 4 sous-onglets, un par tâche — résultat clé, explication, comparaison des 4 modèles, feature importance
- **Analyse** : heatmap de corrélation (Pearson) · SHAP global (TreeExplainer, 300 clients test) · analyse des erreurs (distribution des probabilités par classe réelle, faux négatifs, faux positifs, TP/FP/FN/TN)

### API REST (FastAPI)

```
GET  /health       → état du service + modèle chargé
GET  /model-info   → métriques du modèle final
POST /predict      → prédiction churn pour un client (JSON in → JSON out)
```

---

## Structure du projet

```
├── retention_ai/       # logique métier (données, features, modèles, tâches secondaires, inférence)
│   ├── config.py       # chemins et constantes centralisés
│   ├── data.py         # chargement et schéma
│   ├── features.py     # feature engineering (engagement_score, payment_risk_index…)
│   ├── modeling.py     # 4 modèles de classification churn
│   ├── extra_tasks.py  # 4 tâches secondaires (3 régressions + 1 classification)
│   ├── train.py        # pipeline d'entraînement complet
│   ├── inference.py    # inférence en production
│   └── reporting.py    # génération des figures
├── app/                # dashboard Streamlit (4 onglets)
├── api/                # API FastAPI
├── artifacts/          # modèles entraînés, métriques, figures
│   └── metrics/secondary/  # résultats des tâches secondaires (CSV)
├── data/raw/           # dataset CSV
├── docker-compose.yml
├── Dockerfile
└── rapport_retention_client.tex / .pdf
```

---

## Résultats principaux

- Taux de churn dans le dataset : **10.21 %** (classes déséquilibrées → PR-AUC et Recall privilégiés)
- Le Gradient Boosting détecte **80.9 %** des churners réels (recall test)
- Sur 10 000 clients scorés : **3 237 clients à risque**, perte mensuelle estimée à **116 441 €**
- Variables les plus influentes : `tenure_months`, `csat_score`, `monthly_logins`, `payment_failures`
- Le CLV (total_revenue) est quasi-déterministe depuis les features (R² = 0.9999) — confirmé par la structure du dataset
- L'engagement comportemental n'est pas prédictible depuis le profil client (R² ≈ 0)

---

## Limites identifiées

- **Incohérence CV/test** : le CV utilise le seuil par défaut (0.5), pas le seuil optimisé → scores CV et test non directement comparables
- **Signal faible** : PR-AUC = 0.306 indique une limite du dataset synthétique, pas du modèle
- **Pas d'ablation** : impact réel de chaque feature dérivée non quantifié
- **Dataset synthétique** : les performances ne garantissent pas un transfert à des données réelles
