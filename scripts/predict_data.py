"""
Exemple d'utilisation des modèles sauvegardés

Ce script montre comment charger et utiliser les modèles LightGBM 
sauvegardés pour faire des prédictions sur de nouvelles données.
"""

import pandas as pd
import numpy as np
from scripts.train_model import (
    load_model, 
    predict_with_model,
    application_train_test,
    bureau_and_balance,
    previous_applications,
    pos_cash,
    installments_payments,
    credit_card_balance,
    clean_column_names
)

def prepare_features_for_prediction(num_rows=None):
    """
    Prépare les features exactement de la même manière que lors de l'entraînement
    
    Returns:
    --------
    pandas DataFrame : DataFrame avec toutes les features nécessaires
    """
    # Charger et préparer les données de la même façon que pendant l'entraînement
    df = application_train_test()
    
    bureau = bureau_and_balance(num_rows)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    
    prev = previous_applications()
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    
    pos = pos_cash()
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    
    ins = installments_payments()
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    
    cc = credit_card_balance()
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    
    # Nettoyer les noms de colonnes
    df = clean_column_names(df)
    
    return df


def main_predict():
    """
    Exemple d'utilisation pour faire des prédictions
    """
    print("=" * 80)
    print("CHARGEMENT DU MODÈLE")
    print("=" * 80)
    
    # Charger l'ensemble de modèles
    model_data = load_model('models/model.pkl')
    
    print("\n" + "=" * 80)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 80)
    
    # Préparer les données (ici on utilise les données de test)
    df = prepare_features_for_prediction()
    
    # Filtrer uniquement les données de test (TARGET est null)
    test_df = df[df['TARGET'].isnull()].copy()
    print(f"\nNombre d'observations à prédire: {len(test_df)}")
    
    print("\n" + "=" * 80)
    print("PRÉDICTIONS")
    print("=" * 80)
    
    # Faire des prédictions
    predictions = predict_with_model(model_data, test_df)
    
    # Afficher quelques statistiques
    print(f"\nStatistiques des prédictions:")
    print(f"  - Minimum: {predictions.min():.6f}")
    print(f"  - Maximum: {predictions.max():.6f}")
    print(f"  - Moyenne: {predictions.mean():.6f}")
    print(f"  - Médiane: {np.median(predictions):.6f}")
    
    # Créer un DataFrame de soumission
    submission = pd.DataFrame({
        'SK_ID_CURR': test_df['SK_ID_CURR'],
        'TARGET': predictions
    })
    
    # Sauvegarder les prédictions
    output_file = 'output/predictions_from_loaded_model.csv'
    submission.to_csv(output_file, index=False)
    print(f"\nPrédictions sauvegardées dans: {output_file}")
    
    # Afficher les premières prédictions
    print("\n" + "=" * 80)
    print("APERÇU DES PRÉDICTIONS")
    print("=" * 80)
    print(submission.head(10))
    
    return submission


def predict_single_customer(customer_data, model_data):
    """
    Faire une prédiction pour un seul client
    
    Parameters:
    -----------
    customer_data : pandas DataFrame
        DataFrame avec les features d'un seul client (1 ligne)
    model_data : dict
        Données du modèle chargées avec load_model_ensemble()
    
    Returns:
    --------
    float : Probabilité de défaut de paiement
    """
    prediction = predict_with_ensemble(model_data, customer_data)
    return prediction[0]


if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        submission = main_predict()
        
        print("\n" + "=" * 80)
        print("TERMINÉ AVEC SUCCÈS!")
        print("=" * 80)
        
    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("ERREUR: Fichier du modèle non trouvé!")
        print("=" * 80)
        print("\nVeuillez d'abord entraîner le modèle en exécutant:")
        print("  python lightgbm_with_simple_features.py")
        print("\nCela créera le fichier 'lgbm_model_ensemble.pkl'")
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"ERREUR: {type(e).__name__}")
        print("=" * 80)
        print(f"\n{str(e)}")