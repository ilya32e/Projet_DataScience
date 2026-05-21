# Structure du Projet — Prédiction de la Rétention Client

> Chaque fichier et dossier est décrit ci-dessous selon son rôle dans le pipeline ML, le déploiement ou la documentation.

---

## Vue d'ensemble

```
Projet_DataScience/
│
├── 📦 retention_ai/          ← Package Python principal (logique ML)
├── 🖥  app/                   ← Dashboard Streamlit
├── ⚡  api/                   ← API REST FastAPI
├── 🗄  data/                  ← Données brutes
├── 📊 artifacts/             ← Sorties du pipeline (modèles, métriques, figures)
│
├── 🐳 Dockerfile             ← Image Docker commune (Streamlit + API)
├── 🐳 docker-compose.yml     ← Orchestration des deux services
├── 📋 requirements.txt       ← Dépendances Python
├── 🔧 Makefile               ← Commandes raccourcies
│
├── 📄 rapport_retention_client.tex / .pdf   ← Rapport académique LaTeX
├── 📊 presentation_retention_client.pptx   ← Présentation PowerPoint
├── 📝 script_presentation_retention_client.docx ← Script de présentation
├── 📘 README.md              ← Documentation principale du projet
└── 📘 STRUCTURE.md           ← Ce fichier
```

---

## 1. Package ML — `retention_ai/`

> Cœur du projet. Contient tout le code réutilisable : données, features, modèles, entraînement, inférence.

```
retention_ai/
│
├── __init__.py                    ← Initialise le package Python
│
├── config.py                      ← Chemins, constantes, hyperparamètres globaux
│   └── Centralise BASE_DIR, MODELS_DIR, TARGET_COLUMN, CV_FOLDS…
│       Inclut les chemins pour les artefacts des tâches secondaires
│       (RAR_LEADERBOARD_PATH, CLV_LEADERBOARD_PATH, ENG_REG/CLF…)
│       Tous les autres modules importent depuis ici → un seul endroit à modifier
│
├── data.py                        ← Chargement et préparation des données
│   └── Téléchargement Kaggle, lecture CSV, split train/test,
│       mapping des groupes de features (Profil / Usage / Finance / Support / Marketing)
│
├── features.py                    ← Feature Engineering
│   └── Transformateur sklearn (fit/transform) qui ajoute 4 features dérivées :
│       • engagement_score       (combinaison pondérée des 6 métriques d'usage)
│       • support_ticket_rate    (tickets ramenés à la durée d'ancienneté)
│       • revenue_per_month      (revenu cumulé / tenure)
│       • payment_risk_index     (payment_failures × monthly_fee)
│
├── modeling.py                    ← 4 modèles de classification churn + optimisation RandomizedSearch
│   └── Construit les 4 pipelines sklearn (preprocessing + oversampling + classifieur) :
│       • LogisticRegression    (baseline interprétable)
│       • RandomForestClassifier (ensemble arbres)
│       • GradientBoostingClassifier (boosting — modèle final)
│       • MLPClassifier         (deep learning)
│       Inclut : validation croisée StratifiedKFold, optimisation du seuil,
│       permutation importance, sélection automatique du meilleur modèle
│       + tune_pipeline() : RandomizedSearchCV (20 iter, cv=3, scoring=PR-AUC)
│         appliqué au meilleur modèle après sélection → best_params sauvegardés
│
├── extra_tasks.py                 ← 4 tâches prédictives secondaires
│   └── Chaque tâche utilise 4 modèles + CV + séparation train/test :
│       • Revenu à Risque (Régression) — cible : monthly_fee × churn_proba
│         Modèles : Ridge, RandomForestRegressor, GradientBoostingRegressor, MLPRegressor
│         Résultat : RF R²=0.914 (meilleur)
│       • CLV — Valeur Vie Client (Régression) — cible : total_revenue
│         Modèles : mêmes 4 régresseurs
│         Résultat : RF R²=0.9999 (quasi-déterministe)
│       • Score d'Engagement (Régression) — cible : engagement_score calculé sur usage
│         Features : profil + finance + support + marketing (sans colonnes usage)
│         Résultat : R²≈0 pour tous les modèles — l'engagement est intrinsèque
│       • Catégorie d'Engagement (Classification multi-classe) — cible : Faible/Moyen/Fort
│         Modèles : LogReg, RandomForest, GradientBoosting, MLPClassifier
│         Résultat : F1 Macro ≈ 0.31 (cohérent avec la régression)
│
├── train.py                       ← Script d'entraînement complet (point d'entrée)
│   └── Orchestre l'ensemble du pipeline :
│       1. chargement → features → 4 modèles churn → sélection finale
│       2. RandomizedSearch sur le meilleur modèle (20 iter × 3 folds)
│          → meilleurs hyperparamètres → seuil re-optimisé → modèle final
│       3. scoring du portefeuille (churn_probability pour tous les clients)
│       4. run_all_secondary() → 4 tâches secondaires → sauvegarde CSV
│       Sortie : artefacts churn + artifacts/metrics/secondary/*.csv
│               + best_params dans training_overview.json
│
├── inference.py                   ← Inférence en production
│   └── Charge le modèle sérialisé et expose predict_record()
│       utilisé par l'API FastAPI et le dashboard Streamlit
│
├── reporting.py                   ← Génération des figures
│   └── Produit les graphiques sauvegardés dans artifacts/figures/ :
│       class_balance, contract_churn_rate, model_comparison,
│       final_confusion_matrix, feature_importance
│
├── explainability.py              ← Wrapper SHAP (module d'extension)
│   └── Unifie TreeExplainer (arbres) et KernelExplainer (modèles génériques)
│       Expose : explain(), feature_importance(), summary_plot(), force_plot()
│       Correspondance rapport : section 19.2 "Explicabilité individuelle par SHAP"
│       [Prêt à l'emploi — non branché sur le pipeline principal]
│
├── calibration.py                 ← Calibration probabiliste (module d'extension)
│   └── ProbabilityCalibrator : CalibratedClassifierCV (Platt/Isotonic)
│       + métriques ECE, MCE, Brier Score, Log Loss
│       ThresholdOptimizer : seuil optimal par F1 ou Youden's J
│       Correspondance rapport : section 19.3 "Calibration probabiliste absente"
│       [Prêt à l'emploi — non branché sur le pipeline principal]
│
└── drift_monitor.py               ← Détection de drift (module d'extension)
    └── DataDriftDetector  : test KS (numérique) + Chi² (catégoriel)
        ModelDriftMonitor  : alerte si dégradation métrique > seuil %
        TemporalValidator  : split temporel strict + walk-forward validation
        Correspondance rapport : section 19.6 "Absence de validation temporelle"
        [Prêt à l'emploi — non branché sur le pipeline principal]
```

---

## 2. Dashboard — `app/`

```
app/
└── streamlit_app.py               ← Application Streamlit complète (5 onglets)
    │
    ├── Onglet Pilotage
    │   └── KPI globaux (taux de churn, revenu exposé, nb clients à risque)
    │       + tableau de comparaison des 4 modèles churn
    │       + graphique de permutation importance
    │
    ├── Onglet Portefeuille
    │   └── Scatter plot risque/revenu par client (Faible / Modéré / Critique)
    │       + top 12 clients par revenu à risque
    │
    ├── Onglet Simulation
    │   └── Formulaire de saisie d'un profil client (32 champs groupés)
    │       + bouton "Lancer la prédiction" → appel inference.py ou API REST
    │       → jauge de probabilité, niveau de risque, action recommandée
    │       → explication SHAP locale : top 10 variables (rouge = vers churn, bleu = vers rétention)
    │
    ├── Onglet Tâches secondaires
    │   └── 4 sous-onglets (un par tâche) :
    │       • bandeau type/cible/features + 3 cartes métriques (R² ou F1)
    │       • notes Résultat / Attention / Conclusion
    │       • bloc "Comprendre ces résultats" (explication par modèle)
    │       • comparaison 4 modèles (bar chart + tableau) + feature importance
    │
    └── Onglet Analyse
        ├── Corrélation des variables
        │   └── Heatmap Pearson (toutes les features numériques)
        │       + top 10 corrélations avec le churn
        ├── SHAP — Importance globale
        │   └── Bar chart TreeExplainer sur 300 clients test
        │       + tableau des 15 features les plus influentes (mise en cache)
        └── Analyse des erreurs
            └── Distribution des probabilités par classe réelle (churner vs non-churner)
                + table faux négatifs (churners manqués, triés par revenu)
                + table faux positifs (fausses alarmes)
                + compteurs TP / FP / FN / TN
```

---

## 3. API REST — `api/`

```
api/
├── main.py                        ← Application FastAPI
│   ├── GET  /health               → Vérification que le service tourne
│   ├── GET  /model-info           → Métriques et métadonnées du modèle déployé
│   └── POST /predict              → Scoring d'un client (JSON in → JSON out)
│       Entrée : profil client (32 champs)
│       Sortie : churn_probability, risk_level, expected_monthly_loss, recommended_action
│
└── static/
    └── swagger-dark.css           ← Thème sombre pour la documentation Swagger UI
```

---

## 4. Données — `data/`

```
data/
└── raw/
    └── customer_churn_business_dataset.csv   ← Dataset source
        10 000 clients × 32 features
        Téléchargé depuis Kaggle (miadul/customer-churn-prediction-business-dataset)
        Variable cible : churn (0 = fidèle, 1 = résilié) — taux 10,2 %
```

---

## 5. Artefacts — `artifacts/`

> Toutes les sorties générées par le pipeline d'entraînement. Ne jamais modifier à la main.

```
artifacts/
│
├── models/                        ← Modèles sérialisés (joblib)
│   ├── final_model_bundle.joblib          ← Gradient Boosting — modèle en production
│   ├── gradient_boosting_bundle.joblib
│   ├── random_forest_bundle.joblib
│   ├── logistic_regression_bundle.joblib
│   └── mlp_classifier_bundle.joblib
│       Chaque bundle : pipeline sklearn + seuil optimal + métadonnées
│
├── metrics/                       ← Métriques de performance (CSV)
│   ├── model_comparison.csv       ← Tableau comparatif des 4 modèles churn
│   ├── feature_importance.csv     ← Permutation importance du modèle final churn
│   │
│   └── secondary/                 ← Résultats des tâches secondaires
│       ├── revenue_at_risk.csv    ← Comparaison 4 régresseurs (RMSE, MAE, R², CV)
│       ├── rar_importance.csv     ← Feature importance — Revenu à Risque
│       ├── clv.csv                ← Comparaison 4 régresseurs CLV
│       ├── clv_importance.csv     ← Feature importance — CLV
│       ├── engagement_regression.csv   ← Comparaison 4 régresseurs engagement
│       ├── engagement_importance.csv   ← Feature importance — Engagement
│       └── engagement_classification.csv ← Comparaison 4 classifieurs engagement
│
├── figures/                       ← Graphiques générés (PNG)
│   ├── class_balance.png
│   ├── contract_churn_rate.png
│   ├── model_comparison.png
│   ├── final_confusion_matrix.png
│   ├── feature_importance.png
│   ├── screenshot_dashboard_pilotage.png
│   ├── screenshot_dashboard_portefeuille.png
│   ├── screenshot_dashboard_simulation.png
│   └── screenshot_api_docs.png
│
├── scored_customers.csv           ← Scoring complet du portefeuille
│   Chaque ligne = 1 client avec : churn_probability, predicted_churn,
│   expected_monthly_loss, expected_revenue_at_risk
│
├── training_overview.json         ← Résumé de l'entraînement
│   Contient : date, taille du dataset, modèle final, métriques clés,
│   résumé risque portfolio (3 237 clients à risque, 3,17 M€ exposés)
│
└── schema.json                    ← Schéma du dataset
    Contient les types de chaque feature, les groupes (Profil/Usage/…),
    utilisé par le dashboard et l'API pour valider les données
```

---

## 6. Configuration & Déploiement

```
Dockerfile                         ← Image Docker unique partagée par les deux services
    Base : python:3.11-slim
    Installe les dépendances de requirements.txt
    Point de montage : /app (volume partagé avec l'hôte)

docker-compose.yml                 ← Orchestre les deux conteneurs
    streamlit  → port 8501  (commande : streamlit run app/streamlit_app.py)
    api        → port 8000  (commande : uvicorn api.main:app)
    Réseau interne partagé : retention_network

requirements.txt                   ← Toutes les dépendances Python du projet
    scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, seaborn,
    streamlit, fastapi, uvicorn, joblib, plotly, shap, scipy, evidently…

Makefile                           ← Raccourcis de commandes
    make train     → lance le pipeline d'entraînement complet
    make app       → démarre Streamlit en local
    make api       → démarre FastAPI en local
    make docker    → build + up docker-compose
    make clean     → supprime les artefacts générés
```

---

## 7. Documentation & Rapports

```
README.md                          ← Documentation principale du projet
rapport_retention_client.tex       ← Rapport académique complet (LaTeX)
rapport_retention_client.pdf       ← Version compilée
presentation_retention_client.pptx ← Présentation PowerPoint
script_presentation_retention_client.docx ← Script de présentation
STRUCTURE.md                       ← Ce fichier
```

---

## Flux de données

```
data/raw/customer_churn_business_dataset.csv
        │
        ▼
retention_ai/data.py          ← chargement + split train/test/validation
        │
        ▼
retention_ai/features.py      ← feature engineering (4 features dérivées)
        │
        ▼
retention_ai/modeling.py      ← 4 modèles churn + optimisation seuil + CV
        │
        ▼
retention_ai/train.py         ← sérialisation + scoring portefeuille
        │
        ├──► artifacts/models/                    (bundles .joblib)
        ├──► artifacts/metrics/model_comparison.csv
        ├──► artifacts/metrics/feature_importance.csv
        ├──► artifacts/figures/                   (PNG)
        ├──► artifacts/scored_customers.csv       (churn_probability par client)
        │
        ▼
retention_ai/extra_tasks.py   ← 4 tâches secondaires (dépend de scored_customers)
        │
        ├──► artifacts/metrics/secondary/revenue_at_risk.csv
        ├──► artifacts/metrics/secondary/rar_importance.csv
        ├──► artifacts/metrics/secondary/clv.csv
        ├──► artifacts/metrics/secondary/clv_importance.csv
        ├──► artifacts/metrics/secondary/engagement_regression.csv
        ├──► artifacts/metrics/secondary/engagement_importance.csv
        └──► artifacts/metrics/secondary/engagement_classification.csv
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
app/streamlit_app.py      api/main.py
(port 8501)               (port 8000)
5 onglets dashboard       POST /predict → inference.py
```
