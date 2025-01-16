# load pandas to deal with the data
import pandas as pd
import numpy as np

# Définir les types de données pour les colonnes problématiques
dtype_dict = {
    11: 'string',
    12: 'string'
}

# Charger les données
print("Chargement des données...")
data = pd.read_csv("flickr_data2.csv", dtype=dtype_dict)
print(f"Dimensions initiales : {data.shape}")

# 1. Supprimer les colonnes vides
data = data.dropna(axis=1, how='all')
print(f"Après suppression des colonnes vides : {data.shape}")

# 2. Supprimer les doublons
data = data.drop_duplicates()
print(f"Après suppression des doublons : {data.shape}")

# 3. Supprimer les lignes avec trop de valeurs manquantes (plus de 50% de NaN)
data = data.dropna(thresh=len(data.columns)//2)
print(f"Après suppression des lignes avec trop de NaN : {data.shape}")

# 4. Remplacer les valeurs manquantes restantes
# Pour les colonnes numériques : remplacer par la médiane
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

# Pour les colonnes textuelles : remplacer par 'Unknown'
text_columns = data.select_dtypes(include=['object']).columns
for col in text_columns:
    data[col] = data[col].fillna('Unknown')

# 5. Afficher les informations sur le dataset nettoyé
print("\nInformations sur le dataset nettoyé:")
print(data.info())

print("\nNombre de valeurs uniques par colonne:")
print(data.nunique())

print("\nAperçu des données nettoyées:")
print(data.head())

# 6. Sauvegarder les données nettoyées
print("\nSauvegarde des données nettoyées...")
data.to_csv("flickr_data_cleaned.csv", index=False)
print("Données sauvegardées dans 'flickr_data_cleaned.csv'")
