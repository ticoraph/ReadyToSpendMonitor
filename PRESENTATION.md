# 📊 Présentation du Projet MLOps - Scoring de Crédit

## 🎯 Contexte et Objectifs

### Mission
Piloter la mise en production effective du modèle de scoring pour le département "Crédit Express" de l'entreprise "Prêt à Dépenser".

### Objectifs
1. ✅ Créer une API fonctionnelle pour traiter les demandes en quasi temps réel
2. ✅ Conteneuriser l'application (Docker Ready)
3. ✅ Mettre en place un monitoring proactif
4. ✅ Automatiser le déploiement avec CI/CD

---

## 🏗️ Architecture Technique

### Stack Technologique
- **API**: FastAPI (performance, documentation auto, validation)
- **Monitoring**: Streamlit (simplicité, visualisations interactives)
- **Conteneurisation**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Déploiement**: Hugging Face Spaces
- **Drift Detection**: Evidently AI + SciPy

### Choix Techniques Justifiés

#### Pourquoi FastAPI?
- ⚡ Performance élevée (basé sur Starlette/ASGI)
- 📝 Documentation automatique (Swagger/OpenAPI)
- ✅ Validation native des données (Pydantic)
- 🔧 Facile à tester et maintenir

#### Pourquoi Streamlit?
- 🎨 Création rapide de dashboards
- 📊 Excellente intégration avec pandas/plotly
- 🔄 Rafraîchissement automatique des données
- 💡 Courbe d'apprentissage faible

---

## 📦 Livrables

### ✅ 1. Historique des Versions (Git)
- Dépôt GitHub public avec historique de commits clair
- Structure de projet organisée
- `.gitignore` pour éviter les données sensibles

### ✅ 2. API Fonctionnelle
**Fichiers:**
- `api/main.py` : API FastAPI complète
- `api/schemas.py` : Schémas de validation Pydantic

**Caractéristiques:**
- Endpoints: `/health`, `/predict`, `/docs`
- Validation stricte des entrées
- Gestion d'erreurs robuste
- Chargement du modèle au démarrage (pas à chaque requête)
- Logging structuré des prédictions

### ✅ 3. Tests Unitaires
**Fichier:** `tests/test_api.py`

**Tests implémentés:**
- Health check
- Prédictions valides
- Validation des entrées (âge, revenu négatif, champs manquants)
- Cas limites (min/max)
- Test de charge (10 requêtes)
- Test de temps de réponse

**Exécution:**
```bash
pytest tests/ -v --cov=api
```

### ✅ 4. Conteneurisation
**Fichiers:**
- `Dockerfile` : Image Docker optimisée
- `docker-compose.yml` : Orchestration API + Monitoring

**Commandes:**
```bash
docker build -t scoring-api .
docker-compose up
```

### ✅ 5. Monitoring et Data Drift

**Dashboard Streamlit** (`monitoring/app.py`)
- 📊 Métriques clés (nombre de prédictions, taux d'approbation, temps d'inférence)
- 📈 Distribution des scores
- ⚡ Performance de l'API
- 🔍 Détection de drift (test Kolmogorov-Smirnov)
- 📋 Logs récents
- 💾 Export des données

**Notebook d'Analyse** (`notebooks/drift_analysis.ipynb`)
- Analyse statistique approfondie
- Visualisations comparatives
- Tests de drift multiples
- Recommandations automatiques

**Stockage des Logs**
- Format JSON structuré
- Contient: timestamp, inputs, outputs, temps d'inférence, version du modèle
- Fichier: `production_logs.json`

### ✅ 6. Pipeline CI/CD
**Fichier:** `.github/workflows/deploy.yml`

**Étapes:**
1. **Test**: Installation dépendances, exécution tests, couverture de code
2. **Build**: Construction de l'image Docker, test de l'image
3. **Deploy**: Déploiement sur Hugging Face Spaces (optionnel)
4. **Notify**: Notification du résultat

**Déclenchement:**
- Push sur `main` ou `develop`
- Pull requests vers `main`

### ✅ 7. Documentation
**Fichiers:**
- `README.md` : Documentation complète
- `QUICKSTART.md` : Guide de démarrage rapide
- `README_HF.md` : Documentation pour Hugging Face

---

## 🔬 Analyse du Data Drift

### Méthode
- Test de Kolmogorov-Smirnov pour chaque feature
- Comparaison distributions (référence vs production)
- Seuil de significativité: p-value < 0.05

### Visualisations
- Histogrammes comparatifs
- Box plots
- Rapport Evidently AI (HTML interactif)

### Alertes Automatiques
- 🟢 Aucun drift: Modèle stable
- ⚠️ 1-2 features: Monitoring renforcé
- 🔴 3+ features: Re-entraînement urgent

---

## ⚡ Optimisations Post-Déploiement

### Performance Identifiée
- **Temps d'inférence moyen**: ~15ms
- **Latence API**: ~50ms
- **Charge supportée**: 10 req/s (local)

### Optimisations Implémentées
1. **Chargement du modèle**: Une seule fois au démarrage
2. **Validation Pydantic**: Entrées validées avant traitement
3. **Logging asynchrone**: Pas de blocage sur l'écriture
4. **Docker multi-stage** (optionnel): Image optimisée

### Pistes d'Amélioration Futures
- Quantification du modèle (ONNX)
- Cache des prédictions fréquentes
- Batching des requêtes
- Scaling horizontal (Kubernetes)

---

## 📊 Résultats et Métriques

### Métriques de Production (Simulation)
- ✅ API fonctionnelle et testée
- ✅ Temps de réponse < 100ms
- ✅ Taux d'erreur: 0%
- ✅ Couverture de tests: >80%

### Déploiement
- ✅ Image Docker construite
- ✅ Pipeline CI/CD fonctionnel
- ✅ Prêt pour Hugging Face Spaces

---

## 🛡️ Points de Vigilance

### Sécurité
- ✅ Validation stricte des entrées
- ✅ Pas de données sensibles dans les logs
- ✅ Secrets gérés via variables d'environnement
- ⚠️ À ajouter en production: Rate limiting, authentification

### Conformité RGPD
- ✅ Logs anonymisés (pas d'informations personnelles)
- ✅ Données de production séparées
- ⚠️ À documenter: Politique de conservation des logs

### Scalabilité
- ✅ Architecture prête pour le scaling
- ⚠️ Ajouter en production: Load balancer, auto-scaling

---

## 🎓 Compétences Démontrées

### MLOps
- ✅ Versionning du code (Git)
- ✅ API REST pour modèle ML
- ✅ Conteneurisation (Docker)
- ✅ CI/CD (GitHub Actions)
- ✅ Monitoring et drift detection

### Data Science
- ✅ Développement de modèle de scoring
- ✅ Évaluation des performances
- ✅ Analyse statistique du drift

### Ingénierie Logicielle
- ✅ Tests unitaires
- ✅ Gestion d'erreurs
- ✅ Logging structuré
- ✅ Documentation

---

## 🚀 Démo Live

### 1. Lancer l'API
```bash
uvicorn api.main:app --reload
```

### 2. Test d'une prédiction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 50000,
    "loan_amount": 15000,
    "employment_length": 5,
    "credit_score": 720
  }'
```

### 3. Voir le monitoring
```bash
streamlit run monitoring/app.py
```

---

## 📝 Conclusion

### Objectifs Atteints
✅ API fonctionnelle et performante  
✅ Conteneurisation complète  
✅ Monitoring proactif avec détection de drift  
✅ Pipeline CI/CD automatisé  
✅ Tests unitaires complets  
✅ Documentation exhaustive  

### Prêt pour la Production
Le projet est **Docker Ready** et peut être déployé immédiatement sur:
- Hugging Face Spaces
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances

### Améliorations Futures
1. Monitoring avancé (Prometheus/Grafana)
2. A/B Testing de modèles
3. Re-entraînement automatique
4. Interface utilisateur frontend

---

## 📚 Ressources

- **Dépôt GitHub**: [Lien vers votre repo]
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

---

**Merci pour votre attention! 🎉**

Des questions?
