# Prêt à Dépenser - MLOps Scoring API

## 📋 Description du Projet

Projet de mise en production d'un modèle de scoring de crédit pour l'entreprise "Prêt à Dépenser". Ce projet démontre une implémentation complète MLOps incluant :

- ✅ API REST avec FastAPI
- ✅ Conteneurisation Docker
- ✅ Pipeline CI/CD avec GitHub Actions
- ✅ Monitoring et détection de drift avec Streamlit
- ✅ Tests unitaires automatisés
- ✅ Déploiement sur Docker HUB

## 🏗️ Architecture du Projet

```
ReadyToSpendMonitor/
├── api/                    # Code de l'API FastAPI
│   ├── main.py            # Point d'entrée de l'API
│   └── schemas.py         # Schémas de validation
├── models/                 # Modèles ML et artefacts
│   └── model.pkl          # Modèle entraîné (à ajouter)
├── monitoring/            # Dashboard de monitoring
│   └── app.py            # Application Streamlit
├── tests/                 # Tests unitaires
│   └── test_api.py       # Tests de l'API
│   └── test_app.py       # Tests de l'APP monitoring
├── notebooks/             # Notebooks d'analyse
│   └── drift_analysis.ipynb
├── scripts/               # Scripts utilitaires
│   └── train_model.py    # Entraînement du modèle
│   └── predict_data_from_dataset_thread.py    # Predictions sur un dataset
├── .github/workflows/     # CI/CD
│   └── ci-cd.yml        # Pipeline GitHub Actions
├── Dockerfile            # Configuration Docker
├── requirements.txt      # Dépendances Python
└── .gitignore           # Fichiers à ignorer
```

## 🚀 Installation et Lancement

### Prérequis
- Python 3.10+
- Docker
- Git

### Installation Locale

```bash
# Cloner le repository
git clone https://github.com/ticoraph/ReadyToSpendMonitor.git
cd ReadyToSpendMonitor

# Créer un environnement virtuel
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ajouter vos données et modèle
# - Copier votre modèle dans models/model.pkl
```

### Lancement de l'API

```bash
# Uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# L'API sera accessible sur http://localhost:8000
# Documentation interactive : http://localhost:8000/docs
```

### Lancement du Dashboard de Monitoring

```bash
# Dans un nouveau terminal
streamlit run monitoring/app.py

# Le dashboard sera accessible sur http://localhost:8501
```

### Lancement avec Docker

```bash
# Construire l'image
docker build -t readytospendmonitor .
# Executer
docker run -p 8000:8000 -p 8501:8501 -v ./logs:/app/logs readytospendmonitor

```

## 📊 Utilisation de l'API

### Exemple de requête avec curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
  "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
  "ACTIVE_AMT_CREDIT_SUM_MAX": 450000,
  "ACTIVE_DAYS_CREDIT_MAX": -753,
  "AMT_ANNUITY": 10548,
  "AMT_CREDIT": 148365,
  "AMT_GOODS_PRICE": 135000,
  "ANNUITY_INCOME_PERC": 0.1019130434782608,
  "APPROVED_AMT_ANNUITY_MEAN": 6340.785,
  "APPROVED_CNT_PAYMENT_MEAN": 14.666666666666666,
  "APPROVED_DAYS_DECISION_MAX": -348,
  "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
  "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0,
  "BURO_DAYS_CREDIT_MAX": -753,
  "BURO_DAYS_CREDIT_MEAN": -979.6666666666666,
  "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.2666666666666666,
  "CLOSED_AMT_CREDIT_SUM_MAX": 38650.5,
  "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -943,
  "CLOSED_DAYS_CREDIT_MAX": -1065,
  "CLOSED_DAYS_CREDIT_VAR": 256328,
  "CODE_GENDER": 1,
  "DAYS_BIRTH": -11716,
  "DAYS_EMPLOYED": -449,
  "DAYS_EMPLOYED_PERC": 0.0383236599522021,
  "DAYS_ID_PUBLISH": -3961,
  "DAYS_LAST_PHONE_CHANGE": -1420,
  "DAYS_REGISTRATION": -3997,
  "EXT_SOURCE_1": 0.3608707365728421,
  "EXT_SOURCE_2": 0.4285392216965799,
  "EXT_SOURCE_3": 0.7981372313187245,
  "INSTAL_AMT_PAYMENT_MEAN": 10274.82081081081,
  "INSTAL_AMT_PAYMENT_MIN": 2.7,
  "INSTAL_AMT_PAYMENT_SUM": 380168.37,
  "INSTAL_DBD_MAX": 60,
  "INSTAL_DBD_SUM": 833,
  "INSTAL_DPD_MEAN": 0.4594594594594595,
  "INSTAL_PAYMENT_PERC_MEAN": 0.945945945945946,
  "OWN_CAR_AGE": 9,
  "PAYMENT_RATE": 0.0710949347892023,
  "POS_MONTHS_BALANCE_SIZE": 40,
  "PREV_CNT_PAYMENT_MEAN": 15.142857142857142
}'
```

### Exemple de réponse

```json
{
  "client_id": "req_20260205103137658530",
  "confidence": 0.8117,
  "decision": "REJECTED",
  "inference_time_ms": 9.67,
  "score": 0.1883
}
```

### Points de terminaison disponibles

- `GET /health` : Vérification de santé de l'API
- `POST /predict` : Prédiction de score
- `POST /predict_batch` : Prédiction de scores en parallèle sur un batch de données
- `GET /docs` : Documentation Swagger interactive

## 🧪 Tests

```bash
# Lancer tous les tests
pytest -v

# Lancer avec couverture
pytest tests/ --cov-report=html
```

## 📈 Monitoring

Le dashboard Streamlit affiche :

1. **Métriques en temps réel**
   - Nombre de prédictions
   - Temps d'inférence moyen
   - Distribution des scores

2. **Détection de Data Drift**
   - Comparaison distributions (référence vs production)
   - Tests statistiques (KS, Chi2)
   - Alertes automatiques

3. **Performance opérationnelle**
   - Latence de l'API
   - Taux d'erreur
   - Logs récents

## 🔄 Pipeline CI/CD

Le pipeline GitHub Actions s'exécute automatiquement à chaque push sur `main` :

1. ✅ Installation des dépendances
2. ✅ Exécution des tests unitaires
3. ✅ Construction de l'image Docker
4. ✅ Déploiement sur Docker HUB

### Configuration requise

Ajouter ces secrets dans GitHub Settings > Actions secrets and variables > Repository secrets :

- `DOCKERHUB_TOKEN` : Docker HUB Token

## 🔍 Data Drift Analysis

Le notebook `notebooks/drift_analysis.ipynb` contient :

- Analyse comparative des distributions
- Tests statistiques (Kolmogorov-Smirnov, Chi-Square)
- Visualisations des drifts

## ⚡ Optimisations Implémentées

1. **Chargement du modèle au démarrage**
2. **Validation des entrées** avec Pydantic
3. **Logging structuré** en JSON
4. **Gestion d'erreurs robuste**
5. **Cache des prédictions**

## 📝 Structure des Logs

Les logs de production contiennent :

```json
{
{"timestamp": "2026-02-11T10:26:19.138831", 
"input": {}, 
"output": {"client_id": "req_20260211102619138813", "score": 0.0929, "decision": "REJECTED", "confidence": 0.1339, "inference_time_ms": 0.9}, 
"model_version": "1.0.0"}
}
```

## 🛡️ Sécurité

- Validation stricte des entrées
- Gestion des secrets avec variables d'environnement
- Pas de données sensibles dans les logs
- Rate limiting (à implémenter en production)

## 🤝 Contribution

Ce projet est un travail académique pour la formation Data Science.

## 📄 Licence

MIT License

## 👤 Auteur

Raphaël Montico - Data Scientist @ Prêt à Dépenser (Projet Académique)

## 🙏 Remerciements

- OpenClassrooms

---

**Note** : Ce projet est à des fins éducatives dans le cadre d'une formation MLOps.
