"""
Script d'entraînement du modèle de scoring
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def create_synthetic_data(n_samples=5000):
    """
    Crée des données synthétiques pour l'entraînement
    À remplacer par vos vraies données
    """
    np.random.seed(42)
    
    # Générer des features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(45000, 15000, n_samples)
    loan_amount = np.random.normal(20000, 10000, n_samples)
    employment_length = np.random.randint(0, 30, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    
    # Créer une target basée sur une logique simple
    # Score élevé si: bon credit_score, revenu élevé, prêt faible
    target = (
        (credit_score > 600) & 
        (income > loan_amount * 2) & 
        (age > 25)
    ).astype(int)
    
    # Ajouter du bruit
    noise = np.random.rand(n_samples) < 0.1
    target = np.where(noise, 1 - target, target)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'employment_length': employment_length,
        'credit_score': credit_score,
        'approved': target
    })
    
    return df

def train_model():
    """
    Entraîne le modèle de scoring
    """
    print("🚀 Démarrage de l'entraînement du modèle...")
    
    # Créer ou charger les données
    data_file = "data/training_data.csv"
    
    if os.path.exists(data_file):
        print(f"📂 Chargement des données depuis {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("⚠️ Données non trouvées. Création de données synthétiques...")
        df = create_synthetic_data(5000)
        
        # Sauvegarder pour référence
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_file, index=False)
        print(f"💾 Données sauvegardées dans {data_file}")
    
    print(f"📊 Dataset: {len(df)} échantillons")
    
    # Séparer features et target
    X = df[['age', 'income', 'loan_amount', 'employment_length', 'credit_score']]
    y = df['approved'] if 'approved' in df.columns else df['target']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔧 Entraînement: {len(X_train)} échantillons")
    print(f"🧪 Test: {len(X_test)} échantillons")
    
    # Entraîner le modèle
    print("\n🎯 Entraînement du RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Entraînement terminé!")
    
    # Évaluation
    print("\n📊 Évaluation du modèle:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC AUC Score: {auc:.4f}")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"  {feature:20s}: {importance:.4f}")
    
    # Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Modèle sauvegardé dans {model_path}")
    
    # Sauvegarder les données de référence pour le drift
    reference_path = "data/reference_data.csv"
    X_train.to_csv(reference_path, index=False)
    print(f"💾 Données de référence sauvegardées dans {reference_path}")
    
    print("\n✅ Entraînement terminé avec succès!")
    
    return model

if __name__ == "__main__":
    train_model()
