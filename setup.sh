#!/bin/bash

# Script de setup automatique du projet
# Usage: ./setup.sh

set -e  # Arrêter en cas d'erreur

echo "======================================"
echo "🚀 SETUP PROJET MLOPS - SCORING API"
echo "======================================"
echo ""

# Vérifier Python
echo "🔍 Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION trouvé"
echo ""

# Créer l'environnement virtuel
echo "📦 Création de l'environnement virtuel..."
if [ -d "venv" ]; then
    echo "⚠️  L'environnement virtuel existe déjà"
    read -p "Voulez-vous le recréer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Environnement virtuel recréé"
    fi
else
    python3 -m venv venv
    echo "✅ Environnement virtuel créé"
fi
echo ""

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate
echo "✅ Environnement activé"
echo ""

# Mise à jour de pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip --quiet
echo "✅ pip mis à jour"
echo ""

# Installation des dépendances
echo "📚 Installation des dépendances..."
pip install -r requirements.txt --quiet
echo "✅ Dépendances installées"
echo ""

# Créer les dossiers nécessaires
echo "📁 Création des dossiers..."
mkdir -p models data logs
echo "✅ Dossiers créés"
echo ""

# Entraîner un modèle de démonstration
echo "🎯 Entraînement d'un modèle de démonstration..."
read -p "Voulez-vous entraîner un modèle de démo? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/train_model.py
    echo "✅ Modèle entraîné et sauvegardé dans models/"
else
    echo "⚠️  Pensez à ajouter votre modèle dans models/model.pkl"
fi
echo ""

# Vérifier Docker
echo "🐳 Vérification de Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "✅ $DOCKER_VERSION trouvé"
else
    echo "⚠️  Docker n'est pas installé (optionnel)"
fi
echo ""

# Tests
echo "🧪 Lancement des tests..."
read -p "Voulez-vous lancer les tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/ -v
    echo "✅ Tests terminés"
fi
echo ""

# Récapitulatif
echo "======================================"
echo "✅ SETUP TERMINÉ!"
echo "======================================"
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "1. Activer l'environnement virtuel:"
echo "   source venv/bin/activate"
echo ""
echo "2. Lancer l'API:"
echo "   uvicorn api.main:app --reload"
echo "   → http://localhost:8000"
echo ""
echo "3. Lancer le monitoring:"
echo "   streamlit run monitoring/app.py"
echo "   → http://localhost:8501"
echo ""
echo "4. Tester l'API:"
echo "   python scripts/test_api.py"
echo ""
echo "5. Avec Docker:"
echo "   docker-compose up --build"
echo ""
echo "📚 Documentation:"
echo "   - README.md : Documentation complète"
echo "   - QUICKSTART.md : Guide rapide"
echo "   - http://localhost:8000/docs : API docs"
echo ""
echo "🎉 Bon développement!"
echo ""
