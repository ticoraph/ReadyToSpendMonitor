#!/usr/bin/env python3
"""
Script de vérification de l'installation du projet
Vérifie que tous les composants sont présents et fonctionnels
"""
import sys
import os
from pathlib import Path

def check_file(filepath, description):
    """Vérifie qu'un fichier existe"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} MANQUANT: {filepath}")
        return False

def check_directory(dirpath, description):
    """Vérifie qu'un dossier existe"""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} MANQUANT: {dirpath}")
        return False

def check_module(module_name):
    """Vérifie qu'un module Python peut être importé"""
    try:
        __import__(module_name)
        print(f"✅ Module Python installé: {module_name}")
        return True
    except ImportError:
        print(f"❌ Module Python MANQUANT: {module_name}")
        return False

def main():
    print("=" * 70)
    print("🔍 VÉRIFICATION DE L'INSTALLATION DU PROJET")
    print("=" * 70)
    print()
    
    all_checks = []
    
    # 1. Structure des dossiers
    print("📁 Vérification de la structure des dossiers...")
    print("-" * 70)
    all_checks.append(check_directory("api", "Dossier API"))
    all_checks.append(check_directory("tests", "Dossier Tests"))
    all_checks.append(check_directory("monitoring", "Dossier Monitoring"))
    all_checks.append(check_directory("models", "Dossier Modèles"))
    all_checks.append(check_directory("data", "Dossier Données"))
    all_checks.append(check_directory("scripts", "Dossier Scripts"))
    all_checks.append(check_directory("notebooks", "Dossier Notebooks"))
    all_checks.append(check_directory(".github/workflows", "Dossier CI/CD"))
    print()
    
    # 2. Fichiers essentiels
    print("📄 Vérification des fichiers essentiels...")
    print("-" * 70)
    all_checks.append(check_file("api/main.py", "API principale"))
    all_checks.append(check_file("api/schemas.py", "Schémas API"))
    all_checks.append(check_file("tests/test_api.py", "Tests unitaires"))
    all_checks.append(check_file("monitoring/app.py", "Dashboard monitoring"))
    all_checks.append(check_file("scripts/train_model.py", "Script d'entraînement"))
    all_checks.append(check_file("Dockerfile", "Dockerfile"))
    all_checks.append(check_file("docker-compose.yml", "Docker Compose"))
    all_checks.append(check_file("requirements.txt", "Requirements"))
    all_checks.append(check_file(".github/workflows/deploy.yml", "Pipeline CI/CD"))
    all_checks.append(check_file("README.md", "README principal"))
    all_checks.append(check_file(".gitignore", "Gitignore"))
    print()
    
    # 3. Documentation
    print("📚 Vérification de la documentation...")
    print("-" * 70)
    all_checks.append(check_file("QUICKSTART.md", "Guide démarrage rapide"))
    all_checks.append(check_file("PRESENTATION.md", "Document présentation"))
    all_checks.append(check_file("CHANGELOG.md", "Changelog"))
    all_checks.append(check_file("LICENSE", "Licence"))
    print()
    
    # 4. Modules Python
    print("🐍 Vérification des modules Python...")
    print("-" * 70)
    all_checks.append(check_module("fastapi"))
    all_checks.append(check_module("uvicorn"))
    all_checks.append(check_module("pydantic"))
    all_checks.append(check_module("pandas"))
    all_checks.append(check_module("sklearn"))
    all_checks.append(check_module("streamlit"))
    all_checks.append(check_module("pytest"))
    print()
    
    # 5. Modèle et données
    print("🎯 Vérification du modèle et des données...")
    print("-" * 70)
    model_exists = check_file("models/model.pkl", "Modèle entraîné")
    if not model_exists:
        print("   ⚠️  Lancez 'python scripts/train_model.py' pour créer un modèle")
    
    ref_data_exists = check_file("data/reference_data.csv", "Données de référence")
    if not ref_data_exists:
        print("   ⚠️  Lancez 'python scripts/train_model.py' pour créer les données")
    print()
    
    # 6. Résumé
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100
    
    print(f"Tests réussis: {passed}/{total} ({percentage:.1f}%)")
    print()
    
    if passed == total:
        print("✅ INSTALLATION COMPLÈTE ET FONCTIONNELLE!")
        print()
        print("🚀 Prochaines étapes:")
        print("   1. Activez l'environnement: source venv/bin/activate")
        print("   2. Lancez l'API: uvicorn api.main:app --reload")
        print("   3. Lancez le monitoring: streamlit run monitoring/app.py")
        print("   4. Testez l'API: python scripts/test_api.py")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  Installation presque complète (quelques éléments manquants)")
        print()
        print("💡 Vérifiez les éléments marqués ❌ ci-dessus")
        return 1
    else:
        print("❌ Installation incomplète")
        print()
        print("💡 Relancez le script de setup: ./setup.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
