# 🚀 Guide de Démarrage Rapide

## Installation en 5 minutes

### 1. Cloner le projet
```bash
git clone https://github.com/votre-username/pret-a-depenser-mlops.git
cd pret-a-depenser-mlops
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter vos données et modèle

**Option A: Utiliser vos propres données**
```bash
# Copier votre modèle entraîné
cp /chemin/vers/votre/model.pkl models/

# Copier vos données de référence
cp /chemin/vers/vos/donnees.csv data/reference_data.csv
```

**Option B: Générer des données de démonstration**
```bash
python scripts/train_model.py
```

### 5. Lancer l'API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

✅ L'API est maintenant accessible sur http://localhost:8000

### 6. Tester l'API
```bash
# Dans un nouveau terminal
python scripts/test_api.py
```

### 7. Lancer le Dashboard de Monitoring
```bash
# Dans un nouveau terminal
streamlit run monitoring/app.py
```

✅ Le dashboard est accessible sur http://localhost:8501

---

## Démarrage avec Docker (encore plus simple!)

### 1. Préparer les données
```bash
python scripts/train_model.py
```

### 2. Lancer avec Docker Compose
```bash
docker-compose up --build
```

✅ C'est tout! Les services sont maintenant actifs:
- API: http://localhost:8000
- Monitoring: http://localhost:8501

---

## Exemple d'utilisation de l'API

### Avec curl
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

### Avec Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
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

### Réponse attendue
```json
{
  "client_id": "req_20250202143500123",
  "score": 0.78,
  "decision": "APPROVED",
  "confidence": 0.85,
  "inference_time_ms": 15.3
}
```

---

## Documentation Interactive

Accédez à la documentation Swagger automatique:
👉 http://localhost:8000/docs

---

## Résolution de problèmes

### Problème: "ModuleNotFoundError: No module named 'api'"
**Solution:** Assurez-vous d'être dans le bon dossier et que l'environnement virtuel est activé.

### Problème: "Model not found"
**Solution:** Lancez `python scripts/train_model.py` pour créer un modèle de démonstration.

### Problème: Port 8000 déjà utilisé
**Solution:** 
```bash
# Changez le port
uvicorn api.main:app --port 8001

# Ou tuez le processus existant
lsof -ti:8000 | xargs kill -9  # Linux/Mac
```

### Problème: Streamlit ne se lance pas
**Solution:** Vérifiez que scipy est installé:
```bash
pip install scipy
```

---

## Structure du Projet (Résumé)

```
pret-a-depenser-mlops/
├── api/                    # Code de l'API FastAPI
├── models/                 # Modèle ML (ajoutez le vôtre ici)
├── data/                   # Données (ajoutez les vôtres ici)
├── monitoring/             # Dashboard Streamlit
├── tests/                  # Tests unitaires
├── scripts/                # Scripts utilitaires
├── Dockerfile             # Configuration Docker
└── requirements.txt       # Dépendances
```

---

## Prochaines Étapes

1. ✅ Remplacez le modèle de démo par votre vrai modèle
2. ✅ Ajoutez vos vraies données de référence
3. ✅ Configurez GitHub Actions (ajoutez HF_TOKEN si besoin)
4. ✅ Déployez sur Hugging Face Spaces
5. ✅ Analysez le drift avec le notebook `notebooks/drift_analysis.ipynb`

---

## Support

Pour toute question, consultez le README principal ou le notebook d'analyse.

🎉 **Bonne chance avec votre projet MLOps!**
