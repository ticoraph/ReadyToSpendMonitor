# Prêt à Dépenser - MLOps Scoring API

## 📋 Description du Projet

Projet de mise en production d'un modèle de scoring de crédit pour l'entreprise "Prêt à Dépenser". Ce projet démontre une implémentation complète MLOps incluant :

- ✅ API REST avec FastAPI
- ✅ Conteneurisation Docker
- ✅ Pipeline CI/CD avec GitHub Actions
- ✅ Monitoring et détection de drift avec Streamlit
- ✅ Tests unitaires automatisés
- ✅ Déploiement sur Hugging Face Spaces

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
├── .github/workflows/     # CI/CD
│   └── ci-cd.yml        # Pipeline GitHub Actions
├── Dockerfile            # Configuration Docker
├── docker-compose.yaml    # Configuration Docker compose
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
# Méthode 1 : Uvicorn
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
docker compose up

```

## 📊 Utilisation de l'API

### Exemple de requête avec curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
  }'
```

### Exemple de réponse

```json
{
}
```

### Points de terminaison disponibles

- `GET /health` : Vérification de santé de l'API
- `POST /predict` : Prédiction de score
- `GET /docs` : Documentation Swagger interactive

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

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
4. ✅ Déploiement sur Hugging Face Spaces 

### Configuration requise

Ajouter ces secrets dans GitHub Settings > Secrets :

- `HF_TOKEN` : Token Hugging Face (optionnel)

## 📦 Déploiement sur Hugging Face Spaces

```bash
# 1. Créer un nouveau Space sur Hugging Face
# 2. Configurer le secret HF_TOKEN dans GitHub
# 3. Pusher sur la branche main
git push origin main

# Le déploiement se fait automatiquement via GitHub Actions
```

## 🔍 Data Drift Analysis

Le notebook `notebooks/drift_analysis.ipynb` contient :

- Analyse comparative des distributions
- Tests statistiques (Kolmogorov-Smirnov, Chi-Square)
- Visualisations des drifts
- Recommandations de re-entraînement

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
  "timestamp": "2025-02-02T10:30:00",
  "client_id": "client_123",
  "input": {...},
  "output": {...},
  "inference_time_ms": 12.5,
  "model_version": "v1.0.0"
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
