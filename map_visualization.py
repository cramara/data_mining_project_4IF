import pandas as pd
import folium
from datetime import datetime
import numpy as np
from branca.colormap import LinearColormap

# Charger les données nettoyées
data = pd.read_csv("flickr_data_cleaned.csv")

# Créer une date complète à partir des composants
data['datetime'] = pd.to_datetime(data[['date_taken_year', 'date_taken_month', 'date_taken_day', 
                                      'date_taken_hour', 'date_taken_minute']].assign(second=0))

# Convertir les dates en timestamps pour la colormap
timestamps = data['datetime'].astype(np.int64) // 10**9  # Conversion en secondes Unix
min_time = timestamps.min()
max_time = timestamps.max()

# Créer une carte centrée sur la moyenne des coordonnées
center_lat = data['latitude'].mean()
center_lon = data['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Créer une colormap
colormap = LinearColormap(
    colors=['blue', 'yellow', 'red'],
    vmin=min_time,
    vmax=max_time
)
m.add_child(colormap)

# Ajouter les points avec des couleurs basées sur la date
for idx, row in data.iterrows():
    timestamp = pd.to_datetime(row['datetime']).timestamp()
    color = colormap.rgb_hex_str(timestamp)
    
    # Créer le popup avec les informations de date
    popup_text = f"Date: {row['datetime']}"
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        popup=popup_text,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7
    ).add_to(m)

# Sauvegarder la carte
m.save('carte_photos_temporelle.html')
print("Carte sauvegardée dans 'carte_photos_temporelle.html'") 