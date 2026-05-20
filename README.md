# Système Multi-Modèles pour la Rétention Client

**Projet Data Science — M1 Data Engineering & AI — EFREI Paris 2025-2026**  
Étudiants : Mouradi Iliasse & Kosutic Alexandre | Enseignante : Sarah Malaeb

---

## Présentation

Ce projet a pour objectif de prédire le churn (résiliation) de clients dans un contexte SaaS/abonnement, et d'estimer le revenu financier exposé. On compare quatre modèles de machine learning, on construit un dashboard interactif utilisable par un responsable marketing ou CRM, et on expose une API REST optionnelle pour l'inférence.

Le dataset vient de Kaggle : [Customer Churn Prediction Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) — 10 000 clients, 32 variables, cible binaire `churn`.

---

## Lancer le projet

### Avec Docker (recommandé)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| API FastAPI | http://localhost:8000 |
| Documentation API | http://localhost:8000/docs |

### En local

```bash
pip install -r requirements.txt
python -m retention_ai.train         # entraîne les 4 modèles
streamlit run app/streamlit_app.py   # lance le dashboard
uvicorn api.main:app --reload        # lance l'API (optionnel)
```

> Il faut lancer l'entraînement une première fois avant le dashboard — il génère les artefacts nécessaires (`artifacts/`).

---

## Ce qu'on a fait

### 4 modèles comparés

| Modèle | Famille | Rôle dans le projet |
|---|---|---|
| Logistic Regression | Baseline | Modèle simple et interprétable |
| Random Forest | Ensemble | Capture les non-linéarités |
| Gradient Boosting | Boosting | Meilleure performance globale |
| MLP Classifier | Deep Learning | Exigence du sujet |

Le modèle final retenu est le **Gradient Boosting** (PR-AUC test = 0.306, Recall = 0.809). Le MLP est intentionnellement moins performant — ça démontre que le deep learning n'est pas toujours supérieur sur des données tabulaires.

### Dashboard décisionnel (Streamlit)

Trois onglets :
- **Pilotage** : KPI globaux, comparaison des modèles, importance des variables
- **Portefeuille** : clients les plus à risque, revenu exposé
- **Simulation** : saisir le profil d'un client et obtenir sa probabilité de churn en temps réel

### API REST (FastAPI) — optionnelle

```
GET  /health       → état du service + modèle chargé
GET  /model-info   → métriques du modèle final
POST /predict      → prédiction pour un client
```

---

## Structure du projet

```
├── retention_ai/       # logique métier (données, features, modèles, inférence)
├── app/                # dashboard Streamlit
├── api/                # API FastAPI
├── artifacts/          # modèles entraînés, métriques, figures
├── data/raw/           # dataset CSV
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── rapport_retention_client.tex / .pdf
```

---

## Résultats principaux

- Taux de churn dans le dataset : **10.21%** (classes déséquilibrées → on utilise PR-AUC et Recall)
- Le Gradient Boosting détecte **80.9%** des churners réels (recall test)
- Sur 10 000 clients scorés : **3 237 clients à risque**, perte mensuelle estimée à **116 441 €**
- Variables les plus influentes : `tenure_months`, `csat_score`, `monthly_logins`, `payment_failures`

---

## Limites identifiées

On a documenté honnêtement les limites dans le rapport :

- **Incohérence CV/test** : le CV utilise le seuil par défaut (0.5), pas le seuil optimisé → les scores CV et test ne sont pas directement comparables
- **Signal faible** : PR-AUC = 0.306 indique une limite du dataset synthétique, pas du modèle
- **Pas d'ablation** : on n'a pas quantifié l'impact réel de chaque feature dérivée
- **Dataset synthétique** : les performances ne garantissent pas un transfert à des données réelles
