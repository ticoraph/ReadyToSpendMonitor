---
title: Scoring API - Prêt à Dépenser
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# API de Scoring de Crédit

Cette application déploie un modèle de scoring de crédit avec FastAPI.

## Utilisation

L'API expose les endpoints suivants:

- `GET /` : Page d'accueil
- `GET /health` : Vérification de santé
- `POST /predict` : Prédiction de score
- `GET /docs` : Documentation interactive

## Exemple de requête

```python
import requests

response = requests.post(
    "https://votre-space.hf.space/predict",
    json={
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
)

print(response.json())
```

## Note

Assurez-vous d'avoir le fichier `models/model.pkl` avant de déployer.
