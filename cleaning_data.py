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
data = data[(data['date_taken_year'] > 0) & 
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


# Définir les limites du rectangle
lat_min = 45.70  # Exemple : latitude minimale
lat_max = 45.87  # Exemple : latitude maximale
lon_min = 4.78   # Exemple : longitude minimale
lon_max = 4.95   # Exemple : longitude maximale

# Filtrer les lignes pour ne garder que celles à l'intérieur du rectangle
data = data[(data['lat'] >= lat_min) & (data['lat'] <= lat_max) &
            (data['long'] >= lon_min) & (data['long'] <= lon_max)]

print(f"Après suppression des lignes hors du rectangle : {data.shape}")

# 6. Sauvegarder les données nettoyées
print("\nSauvegarde des données nettoyées...")
data.to_csv("flickr_data_cleaned.csv", index=False)
print("Données sauvegardées dans 'flickr_data_cleaned.csv'")


######################################################################

"""
# Afficher les types de chaque valeur dans la colonne "date_upload_minute"
types_values = data[" title"].apply(type)
print(types_values.value_counts())

# Afficher les lignes dont la colonne "date_upload_minute" est de type string
lignes_string = data[data[" title"].apply(lambda x: isinstance(x, float))]
print(lignes_string[" title"])


print("Types de données des colonnes :")
print(data.dtypes)

print(data.columns)

# Afficher les lignes dont la colonne "date_upload_minute" est de type string
lignes_string = data[data[" date_upload_minute"].apply(lambda x: isinstance(x, str))]
print(lignes_string[" date_upload_minute"])

# Afficher les types de chaque valeur dans la colonne "date_upload_minute"
types_values = data[" tags"].apply(type)
print(types_values.value_counts())

# Afficher les lignes dont la colonne "date_upload_minute" est de type string
lignes_string = data[data[" date_upload_minute"].apply(lambda x: isinstance(x, str))]
print(lignes_string)

# Afficher les lignes dont la colonne "date_upload_minute" est de type string
print(data[data[" date_upload_minute"]>60])


print(f"Avant suppression de la conversion : {data.shape}")

# Convertir les valeurs de la colonne "date_upload_minute" en int
data[" date_upload_minute"] = pd.to_numeric(data[" date_upload_minute"], errors='coerce')

# Supprimer les lignes avec des valeurs NaN résultant de la conversion
data = data.dropna(subset=[" date_upload_minute"])

# Convertir les valeurs en int
data[" date_upload_minute"] = data[" date_upload_minute"].astype(int)

print(f"Après suppression de la conversion : {data.shape}")


"""


"""
#Min en entier etc.




print(data.columns)

data["Unnamed: 16"].count()
data["Unnamed: 18"].count()
data[' long'].count()


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

"""