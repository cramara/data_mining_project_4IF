import pandas as pd
import folium

# Lire le fichier CSV
df = pd.read_csv('flickr_data_cleaned.csv')

# Ne garder que les 100 premiers points
df = df.head(100)  # Vous pouvez ajuster ce nombre selon vos besoins

# Créer une carte centrée sur la moyenne des coordonnées
carte = folium.Map(
    location=[df['latitude'].mean(), df['longitude'].mean()],
    zoom_start=12
)

# Ajouter chaque point à la carte
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red',
        fill=True,
        popup=f"Photo ID: {row['photo_id']}",
    ).add_to(carte)

# Sauvegarder la carte en HTML
carte.save('carte_photos.html') 