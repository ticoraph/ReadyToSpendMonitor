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
)


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
    
    # Préparer les données 
    test_df = pd.read_csv('output/dataset_test_top40.csv') 
    
    # Grouper par TARGET et prendre 10% de chaque groupe
    #df_test_10 = test_df.groupby('TARGET', group_keys=False).apply(
    #    lambda x: x.sample(frac=0.10, random_state=42))
    #test_df = df_test_10

    # Filtrer uniquement les données de test (TARGET est null)
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
    
    # Compter les prédictions au-dessus et au-dessous de 0.5
    print(f"\n" + "=" * 80)
    print("COMPTAGE DES PRÉDICTIONS (seuil 0.2)")
    print("=" * 80)
    
    count_below = (predictions <= 0.2).sum()
    count_above = (predictions > 0.2).sum()
    total = len(predictions)
    
    print(f"\nPrédictions ≤ 0.2 (classe 0) : {count_below} ({count_below/total*100:.2f}%)")
    print(f"Prédictions > 0.2  (classe 1) : {count_above} ({count_above/total*100:.2f}%)")
    print(f"Total                         : {total}")
    
    # Tableau récapitulatif
    summary_data = {
        'Classe': ['0 (≤ 0.2)', '1 (> 0.2)'],
        'Count': [count_below, count_above],
        'Percentage (%)': [round(count_below/total*100, 2), round(count_above/total*100, 2)]
    }
    summary_df = pd.DataFrame(summary_data)
    print("\n")
    print(summary_df.to_string(index=False))


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