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
│       Tous les autres modules importent depuis ici → un seul endroit à modifier
│
├── data.py                        ← Chargement et préparation des données
│   └── Téléchargement Kaggle, lecture CSV, split train/test,
│       mapping des groupes de features (Profil / Usage / Finance / Support / Marketing)
│
├── features.py                    ← Feature Engineering
│   └── Transformateur sklearn (fit/transform) qui ajoute 3 features dérivées :
│       • engagement_score       (combinaison pondérée des métriques d'usage)
│       • payment_risk_index     (combinaison failures + fee + revenue)
│       • support_ticket_rate    (tickets ramenés à la durée d'ancienneté)
│
├── modeling.py                    ← Définition et entraînement des modèles
│   └── Construit les 4 pipelines sklearn (preprocessing + oversampling + classifieur) :
│       • GradientBoostingClassifier
│       • RandomForestClassifier
│       • LogisticRegression
│       • MLPClassifier
│       Inclut l'optimisation du seuil de décision par maximisation du recall
│
├── train.py                       ← Script d'entraînement complet (point d'entrée)
│   └── Orchestre l'ensemble du pipeline :
│       chargement → features → cross-validation → sélection du meilleur modèle
│       → sérialisation joblib → scoring du portefeuille → sauvegarde des artefacts
│
├── inference.py                   ← Inférence en production
│   └── Charge le modèle sérialisé et expose predict_one() / predict_batch()
│       utilisé par l'API FastAPI et le dashboard Streamlit
│
├── reporting.py                   ← Génération des figures et rapports
│   └── Produit les graphiques sauvegardés dans artifacts/figures/ :
│       class_balance, contract_churn_rate, model_comparison,
│       final_confusion_matrix, feature_importance
│
├── calibration.py                 ← Calibration probabiliste (module avancé)
│   └── Wrapper sklearn CalibratedClassifierCV (Platt / Isotonic)
│       + optimisation du seuil par courbe précision-rappel
│       [Non utilisé dans le pipeline principal — prévu pour v2]
│
├── diagnostics.py                 ← Diagnostics du modèle (module avancé)
│   └── Outils d'analyse : biais train/test, courbes ROC/PR,
│       détection de data leakage, rapport de classification complet
│       [Non utilisé dans le pipeline principal — prévu pour v2]
│
├── explainability.py              ← Explicabilité SHAP (module avancé)
│   └── Wrapper TreeExplainer / KernelExplainer
│       pour générer des force plots et dependence plots
│       [Non utilisé dans le pipeline principal — prévu pour v2]
│
├── drift_monitor.py               ← Monitoring de data drift (module avancé)
│   └── Tests KS (numérique), Chi² (catégoriel), KL divergence
│       pour détecter une dérive des données en production
│       [Non utilisé dans le pipeline principal — prévu pour v2]
│
└── hyperparameter_optimization.py ← Optimisation Optuna (module avancé)
    └── Recherche bayésienne d'hyperparamètres multi-objectif via Optuna
        [Non utilisé dans le pipeline principal — prévu pour v2]
```

---

## 2. Dashboard — `app/`

```
app/
└── streamlit_app.py               ← Application Streamlit complète (3 onglets)
    │
    ├── Onglet Pilotage
    │   └── KPI globaux (taux de churn, revenu exposé, nb clients à risque)
    │       + tableau de comparaison des modèles
    │
    ├── Onglet Portefeuille
    │   └── Tableau interactif des clients scorés (filtres, tri, export)
    │       avec probabilité de churn et revenu mensuel exposé par client
    │
    └── Onglet Simulation
        └── Formulaire de saisie d'un profil client (32 champs groupés)
            + bouton "Lancer la prédiction" → appel inference.py
            → affichage jauge de probabilité, niveau de risque, revenu exposé
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
│       Sortie : churn_probability, risk_level, expected_monthly_loss
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
│   ├── gradient_boosting_bundle.joblib    ← même modèle (sauvegarde nommée)
│   ├── random_forest_bundle.joblib        ← Random Forest entraîné
│   ├── logistic_regression_bundle.joblib  ← Régression Logistique entraînée
│   └── mlp_classifier_bundle.joblib       ← MLP entraîné
│       Chaque bundle contient : pipeline sklearn + seuil optimal + métadonnées
│
├── metrics/                       ← Métriques de performance (CSV)
│   ├── model_comparison.csv       ← Tableau comparatif des 4 modèles
│   │   Colonnes : ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix…
│   └── feature_importance.csv     ← Permutation importance par feature
│       Colonnes : feature, importance_mean, importance_std
│
├── figures/                       ← Graphiques générés (PNG)
│   ├── class_balance.png                    ← Distribution churn / non-churn
│   ├── contract_churn_rate.png              ← Taux de churn par type de contrat
│   ├── model_comparison.png                 ← Barplot comparatif des modèles
│   ├── final_confusion_matrix.png           ← Matrice de confusion du modèle final
│   ├── feature_importance.png               ← Graphique de permutation importance
│   ├── screenshot_dashboard_pilotage.png    ← Capture onglet Pilotage
│   ├── screenshot_dashboard_portefeuille.png← Capture onglet Portefeuille
│   ├── screenshot_dashboard_simulation.png  ← Capture onglet Simulation (résultat)
│   └── screenshot_api_docs.png              ← Capture Swagger UI FastAPI
│
├── scored_customers.csv           ← Scoring complet du portefeuille
│   Chaque ligne = 1 client avec : churn_probability, risk_level,
│   expected_monthly_loss, expected_revenue_loss
│
├── training_overview.json         ← Résumé de l'entraînement
│   Contient : date, taille du dataset, modèle final, métriques clés,
│   résumé risque portfolio (3 237 clients à risque, 3,17 M€ exposés)
│
└── schema.json                    ← Schéma du dataset
    Contient les types de chaque feature, les groupes (Profil/Usage/…),
    utilisé par l'API pour valider les requêtes entrantes
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
    streamlit, fastapi, uvicorn, joblib, plotly, optuna, shap…

Makefile                           ← Raccourcis de commandes
    make train     → lance le pipeline d'entraînement
    make app       → démarre Streamlit en local
    make api       → démarre FastAPI en local
    make docker    → build + up docker-compose
    make clean     → supprime les artefacts générés

.streamlit/config.toml             ← Configuration Streamlit
    Thème sombre, port, options d'affichage

.dockerignore                      ← Fichiers exclus du build Docker (.venv, __pycache__…)
.gitignore                         ← Fichiers exclus du dépôt Git
.env.example                       ← Variables d'environnement à définir (template)
```

---

## 7. Documentation & Rapports

```
README.md                          ← Documentation principale du projet
    Installation, lancement, description des fonctionnalités, architecture

rapport_retention_client.tex       ← Rapport académique complet (LaTeX, ~33 pages)
rapport_retention_client.pdf       ← Version compilée du rapport
rapport_retention_client.aux/.log/.toc/.out  ← Fichiers intermédiaires LaTeX (ignorables)

presentation_retention_client.pptx ← Présentation PowerPoint 11 diapositives
    Titre, Données, Méthodologie, Comparaison modèles, Performance,
    Features, Dashboard, API, Portfolio, Conclusion, Démo live

script_presentation_retention_client.docx ← Script de présentation (Word)
    Texte mot-à-mot pour chaque diapositive + durées + conseils

MATRICE_CONFORMITE_RNCP.md         ← Grille de conformité RNCP (certification)
    Mapping compétences visées ↔ livrables du projet

Projet M1 DE Sujet 2 (...).pdf     ← Sujet original du projet
```

---

## 8. Scripts utilitaires (racine)

```
simulate.py                        ← Automation Playwright pour la démo Simulation
    Remplit le formulaire Streamlit avec un profil client à haut risque,
    soumet la prédiction et capture le screenshot du résultat
    (utilisé pour générer screenshot_dashboard_simulation.png)

debug_vis.py                       ← Script de debug DOM / CSS (développement uniquement)
    Inspecte les éléments Streamlit dans le navigateur headless pour
    diagnostiquer les problèmes de sélecteurs CSS

generate_ppt.py                    ← Script de génération du fichier .pptx
    Construit presentation_retention_client.pptx via python-pptx

generate_script.py                 ← Script de génération du script Word
    Construit script_presentation_retention_client.docx via python-docx
```

---

## Flux de données simplifié

```
data/raw/customer_churn_business_dataset.csv
        │
        ▼
retention_ai/data.py          ← chargement + split train/test
        │
        ▼
retention_ai/features.py      ← feature engineering (3 features dérivées)
        │
        ▼
retention_ai/modeling.py      ← entraînement 4 modèles + optimisation seuil
        │
        ▼
retention_ai/train.py         ← sérialisation + scoring portefeuille
        │
        ├──► artifacts/models/          (modèles .joblib)
        ├──► artifacts/metrics/         (CSV de performance)
        ├──► artifacts/figures/         (graphiques PNG)
        ├──► artifacts/scored_customers.csv
        └──► artifacts/training_overview.json
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
app/streamlit_app.py      api/main.py
(port 8501)               (port 8000)
Dashboard interactif      API REST /predict
```
