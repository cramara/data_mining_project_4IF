# load pandas to deal with the data
import pandas as pd
import numpy as np

# Charger les données
print("Chargement des données...")
data = pd.read_csv("flickr_data2.csv")
print("Données chargées.")
print(f"Dimensions initiales : {data.shape}")

data["Unnamed: 16"].count()
data["Unnamed: 17"].count()
data["Unnamed: 18"].count()
data[' long'].count()
# Seulement 144 lignes sur 420000 ont des valeurs dans ces colonnes - on peut les supprimer, mais on va déja
# essayer de comprendre pourquoi elles sont là : caractère spéciaux comme ; dans le titre qui sont interprétés
# comme des séparateurs de colonnes. On va supprimer ces lignes car elles sont peu nombreuses

print(f"Avant suppression des lignes : {data.shape}")

# Supprimer les lignes ayant des valeurs dans les colonnes "Unnamed: 16", "Unnamed: 17", et "Unnamed: 18"
data = data[data[["Unnamed: 16", "Unnamed: 17", "Unnamed: 18"]].isna().all(axis=1)]

print(f"Après suppression des lignes : {data.shape}")

# Supprimer les colonnes "Unnamed: 16", "Unnamed: 17", et "Unnamed: 18" pour l'intégralité des lignes
data = data.drop(columns=["Unnamed: 16", "Unnamed: 17", "Unnamed: 18"])


# 2. Supprimer les doublons
data = data.drop_duplicates(keep='first')
print(f"Après suppression des doublons : {data.shape}")

# Supprimer les espaces au début et à la fin des noms de colonnes
data.columns = data.columns.str.strip()

#Supprimer les colonnes inutiles pour le projet
data = data.drop(columns=["date_upload_minute", "date_upload_hour", "date_upload_day", "date_upload_month","date_upload_year"])


# Filtrer les lignes avec des valeurs incorrectes
data = data[(data['date_taken_year'] > 2010) &  (data['date_taken_year'] <= 2024) &
            (data['date_taken_month'] > 0) & (data['date_taken_month'] <= 12) & 
            (data['date_taken_day'] > 0) & (data['date_taken_day'] <= 31) & 
            (data['date_taken_hour'] >= 0) & (data['date_taken_hour'] < 24) & 
            (data['date_taken_minute'] >= 0) & (data['date_taken_minute'] < 60)]

# Concaténer les colonnes en une seule colonne de type datetime
try:
    # Supprimer les parties non converties comme ".0"
    data['date_taken_year'] = data['date_taken_year'].astype(str).str.replace('.0', '', regex=False)
    data['date_taken_month'] = data['date_taken_month'].astype(str).str.replace('.0', '', regex=False)
    data['date_taken_day'] = data['date_taken_day'].astype(str).str.replace('.0', '', regex=False)
    data['date_taken_hour'] = data['date_taken_hour'].astype(str).str.replace('.0', '', regex=False)
    data['date_taken_minute'] = data['date_taken_minute'].astype(str).str.replace('.0', '', regex=False)

    data['date_taken'] = pd.to_datetime(data['date_taken_year'] + '-' +
                                        data['date_taken_month'] + '-' +
                                        data['date_taken_day'] + ' ' +
                                        data['date_taken_hour'] + ':' +
                                        data['date_taken_minute'],
                                        format='%Y-%m-%d %H:%M')
    print("Colonne 'date_taken' créée avec succès.")
except KeyError as e:
    print(f"Erreur : La colonne {e} n'existe pas dans le DataFrame.")
except ValueError as e:
    print(f"Erreur de conversion de date : {e}")

#Supprimer les colonnes inutiles pour le projet
data = data.drop(columns=["date_taken_minute", "date_taken_hour", "date_taken_day", "date_taken_month","date_taken_year"])

#Supprimer les lignes qui n'ont ni tags, ni titre
data = data.dropna(subset=['tags', 'title'], how='all')

# Définir les limites du rectangle
lat_min = 45.73  # Exemple : latitude minimale
lat_max = 45.80  # Exemple : latitude maximale
lon_min = 4.79   # Exemple : longitude minimale
lon_max = 4.90   # Exemple : longitude maximale

# Filtrer les lignes pour ne garder que celles à l'intérieur du rectangle
data = data[(data['lat'] >= lat_min) & (data['lat'] <= lat_max) &
            (data['long'] >= lon_min) & (data['long'] <= lon_max)]

print(f"Après suppression des lignes hors du rectangle : {data.shape}")

# 6. Sauvegarder les données nettoyées
print("\nSauvegarde des données nettoyées...")
data.to_csv("flickr_data_cleaned.csv", index=False)
print("Données sauvegardées dans 'flickr_data_cleaned.csv'")