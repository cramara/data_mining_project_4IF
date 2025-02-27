import pandas as pd
import folium
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from scipy.spatial import ConvexHull
import colorsys
from collections import Counter
import webbrowser
import os
import plotly.express as px
import math
from collections import defaultdict
import unicodedata

show_time_plots = True  # Valeur par défaut
time_grouping = "mois"  # Valeur par défaut

def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

def generate_time_distribution_plot(cluster_data, cluster_id, cluster_name):
    """Génère un graphique de distribution temporelle pour un cluster"""
    try:
        # Créer une copie du DataFrame pour éviter les warnings
        cluster_data = cluster_data.copy()
        
        try:
            # Convertir explicitement la colonne date_taken en datetime
            cluster_data['date_taken'] = pd.to_datetime(cluster_data['date_taken'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            # Supprimer les lignes avec des dates invalides
            cluster_data = cluster_data.dropna(subset=['date_taken'])
            
            if len(cluster_data) == 0:
                print(f"Aucune donnée valide pour le cluster {cluster_id}")
                return None
            
            # Regrouper selon le choix de l'utilisateur
            if time_grouping == "mois":
                # Grouper par mois
                daily_counts = (cluster_data.groupby(pd.Grouper(key='date_taken', freq='ME'))
                              .size()
                              .to_frame(name='count')
                              .reset_index())
                daily_counts['date'] = daily_counts['date_taken'].dt.strftime('%Y-%m')
                x_title = "Mois"
            else:  # année
                # Créer directement une colonne année
                daily_counts = (cluster_data.assign(year=lambda x: x['date_taken'].dt.year)
                              .groupby('year')
                              .size()
                              .reset_index(name='count'))
                daily_counts.columns = ['date', 'count']
                daily_counts['date'] = daily_counts['date'].astype(str)
                daily_counts = daily_counts.sort_values('date')
                x_title = "Année"
            
            # Vérifier si les données sont vides après le regroupement
            if daily_counts.empty:
                print(f"Aucune donnée après regroupement pour le cluster {cluster_id}")
                return None

            # Créer le graphique avec plotly
            fig = px.line(daily_counts, x='date', y='count',
                         title=f'Distribution temporelle du cluster {cluster_name} (ID: {cluster_id})')
            
            # Personnaliser le layout
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title="Nombre de photos",
                hovermode='x'
            )
            
            # Si c'est un graphique par année, forcer l'affichage de toutes les années
            if time_grouping == "année":
                fig.update_xaxes(
                    dtick=1,
                    type='category',
                    categoryorder='category ascending'
                )
            
            # Sauvegarder le graphique
            plot_path = os.path.join('cluster_plots', f'cluster_{cluster_id}_distribution.html')
            fig.write_html(plot_path)
            return f'cluster_plots/cluster_{cluster_id}_distribution.html'
            
        except Exception as e:
            print(f"Erreur lors du traitement des données pour le cluster {cluster_id}: {str(e)}")
            print(f"Types des données: {cluster_data['date_taken'].dtypes}")
            print(f"Exemple de dates: {cluster_data['date_taken'].head()}")
            return None
            
    except Exception as e:
        print(f"Erreur lors de la génération du graphique pour le cluster {cluster_id}: {str(e)}")
        return None

def main():
    try:
        global df, clustering_algo, N, show_points, nb_points_cluster, show_time_plots, time_grouping

        df = df.sample(n=min(int(nb_points_cluster), len(df)), random_state=42)

        print(f"Taille du DataFrame après échantillonnage: {df.shape}")
        # Préparer les données pour la clusterisation
        X = df[['lat', 'long']].values
        
        # Appliquer l'algorithme de clustering
        df['cluster'] = clustering_algo.fit_predict(X)

        # Prendre un échantillon aléatoire de nb_points_cluster points
       
        
        # Si K-means est utilisé, les clusters commencent à 0 et sont tous positifs
        # Pour DBSCAN, -1 représente le bruit
        
        all_dataset = []

        # Trouver les tags les plus communs dans tout le dataset
        all_dataset_tags = []
        for tags_str in df['tags'].fillna(''):
            tags_list = tags_str.lower().split(',')
            for tag in tags_list:
                tag = tag.strip()
                tag = remove_accents(tag)  # Normaliser le tag
                mots_exclus = ['unknown', 'lyon', '', 'france', 'europe']
                if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                    subtags = tag.replace('_', ' ').replace('-', ' ').split()
                    all_dataset_tags.extend(subtags)

        # Trouver les titres les plus communs dans tout le dataset
        all_dataset_title = []
        for title_str in df['title'].fillna(''):
            title_list = tags_str.lower().split(',')
            for title in title_list:
                title = title.strip()
                title = remove_accents(title)  # Normaliser le title
                mots_exclus = ['unknown', 'lyon', '', 'france', 'europe']
                if title not in mots_exclus and not any(c.isdigit() for c in title):
                    subtitle = title.replace('_', ' ').replace('-', ' ').split()
                    all_dataset_title.extend(subtitle)

        all_dataset.extend(all_dataset_tags + all_dataset_title)

        # Trouver les N tags les plus communs dans tout le dataset (title + tags)
        common_tags = [tag for tag, _ in Counter(all_dataset_tags).most_common(N)]
        print("Tags les plus communs exclus:", common_tags)

        # Ajouter ces tags communs à la liste des mots exclus
        mots_exclus = ['unknown', 'lyon', '', 'france', 'europe','nuit','streetphotography','french','creative','basilique','wheatpaste']
        
        # Si l'option est activée et qu'un tag de recherche est présent, ne pas l'exclure
        search_term = getattr(df, 'search_term', None)
        keep_search_tag = getattr(df, 'keep_search_tag', False)
        
        if keep_search_tag and search_term:
            # Ne pas ajouter le tag recherché aux mots exclus
            mots_exclus.extend(tag for tag in common_tags if tag != search_term)
        else:
            if keep_search_tag and search_term:
                mots_exclus.extend(tag for tag in common_tags if tag != search_term)
            else:
                pass  # Ne pas exclure les tags communs
        
        print("Mots exclus:", mots_exclus)


        # Trouver les noms de clusters avec TF-IDF
        cluster_tags = {}
        unique_clusters = sorted(df['cluster'].unique())
        
        # Exclure le cluster de bruit (-1) pour le calcul des fréquences
        clusters_for_tfidf = [c for c in unique_clusters if c != -1]
        total_clusters = len(clusters_for_tfidf)
        
        # Étape 1: Collecter la fréquence des documents (nombre de clusters où chaque tag apparaît)
        doc_freq = defaultdict(int)
        for cluster_id in clusters_for_tfidf:
            cluster_data = df[df['cluster'] == cluster_id]
            all_tags = []
            
            for tags_str in cluster_data['tags'].fillna(''):
                tags_list = tags_str.lower().split(',')
                for tag in tags_list:
                    tag = tag.strip()
                    tag = remove_accents(tag)
                    if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                        subtags = [tag.replace('_', ' ').replace('-', ' ').strip()]
                        all_tags.extend(subtags)

            for title_str in cluster_data['title'].fillna(''):
                title_words = remove_accents(title_str).lower().split()
                for word in title_words:
                    word = word.strip()
                    if word not in mots_exclus and not any(c.isdigit() for c in word):
                        all_tags.extend(word.split())
                        
            unique_tags = set(all_tags)
            for tag in unique_tags:
                doc_freq[tag] += 1
        
        # Étape 2: Calculer le score TF-IDF pour chaque tag dans chaque cluster
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_tags[cluster_id] = "Non clustérisé"
                continue
            
            cluster_data = df[df['cluster'] == cluster_id]
            all_tags = []
            for tags_str in cluster_data['tags'].fillna(''):
                tags_list = tags_str.lower().split(',')
                for tag in tags_list:
                    tag = tag.strip()
                    tag = remove_accents(tag)
                    if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                        subtags = tag.replace('_', ' ').replace('-', ' ').split()
                        all_tags.extend(subtags)
            
            tag_counts = Counter(all_tags)
            scores = {}
            total_terms = sum(tag_counts.values())

            for tag, count in tag_counts.items():
                if len(tag) <= 2 or ' ' in tag or tag in mots_exclus:
                    continue
                
                # TF normalisé
                tf = count / total_terms if total_terms > 0 else 0
                
                # IDF ajusté
                df_count = doc_freq.get(tag, 0)
                idf = math.log(total_clusters / (df_count + 1e-6))
                
                scores[tag] = tf * idf
            
            # Sélection adaptative
            if scores:
                best_score = max(scores.values())
                threshold = 0.7 * best_score
                best_tags = [tag.capitalize() for tag, score in scores.items() if score >= threshold]
                
                # Récupérer au moins 1 tag pour les petits clusters
                if not best_tags and scores:
                    best_tags = [max(scores, key=scores.get).capitalize()]
                    
                cluster_tags[cluster_id] = ', '.join(best_tags[:3])
            else:
                cluster_tags[cluster_id] = f"Cluster {cluster_id}"

        # Nombre de clusters trouvés (excluant le bruit qui est -1)
        n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
        print(f"Nombre de clusters trouvés : {n_clusters}")

        # Générer des couleurs aléatoires pour chaque cluster
        colors = []
        for i in range(n_clusters):
            hue = i / n_clusters
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(color)
        # Ajouter une couleur grise pour les points de bruiti
        colors = ['#808080'] + colors

        # Créer une carte centrée sur la moyenne des coordonnées
        carte = folium.Map(
            location=[df['lat'].mean(), df['long'].mean()],
            zoom_start=15
        )

        # Ajouter le rectangle englobant
        bounds = [
            [df['lat'].min(), df['long'].min()],  # coin sud-ouest
            [df['lat'].max(), df['long'].max()]   # coin nord-est
        ]
        print(bounds)

        bounds = [[45.73, 4.79], [45.80, 4.90]]
        
        folium.Rectangle(
            bounds=bounds,
            color='red',
            weight=2,
            fill=False,
            popup='Zone d\'étude',
            opacity=0.7
        ).add_to(carte)

        # Créer un dictionnaire pour stocker les points de chaque cluster
        cluster_points = {}
        for cluster_id in unique_clusters:
            cluster_points[cluster_id] = df[df['cluster'] == cluster_id]
        
        # Créer un dossier pour les graphiques si nécessaire
        plots_dir = 'cluster_plots'
        if show_time_plots:
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Nettoyer le dossier des anciens graphiques
            for file in os.listdir(plots_dir):
                if file.endswith('.html'):
                    os.remove(os.path.join(plots_dir, file))
        
        # Générer les graphiques pour chaque cluster
        plot_paths = {}
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            if len(cluster_data) >= 3:
                try:
                    cluster_points = cluster_data[['lat', 'long']].values
                    jittered_points = cluster_points + np.random.normal(0, 1e-10, cluster_points.shape)
                    hull = ConvexHull(jittered_points)
                    hull_points = jittered_points[hull.vertices]
                    polygon_points = [[point[0], point[1]] for point in hull_points]
                    
                    # Utiliser le tag le plus représentatif dans le popup
                    cluster_name = cluster_tags[cluster_id]
                    nb_points = len(cluster_data)
                    
                    # Générer le popup selon que les graphiques sont activés ou non
                    if show_time_plots:
                        # Générer le graphique de distribution
                        plot_path = generate_time_distribution_plot(
                            cluster_data, 
                            cluster_id,
                            cluster_name
                        )
                        if plot_path: 
                            plot_paths[cluster_id] = plot_path
                            popup_content = f"""
                            <div style="min-width: 200px;">
                            <b>{cluster_name}</b><br>
                            Nombre de points : {nb_points}<br>
                            <button onclick="window.open('./{plot_path}', 
                                'Distribution temporelle', 
                                'width=800,height=600'); return false;">
                                Voir distribution temporelle
                            </button>
                            </div>
                            """
                        else:
                            popup_content = f"""
                            <div style="min-width: 200px;">
                            <b>{cluster_name}</b><br>
                            Nombre de points : {nb_points}<br>
                            (Données temporelles non disponibles)
                            </div>
                            """
                    else:
                        popup_content = f"""
                        <div style="min-width: 200px;">
                        <b>{cluster_name}</b><br>
                        Nombre de points : {nb_points}
                        </div>
                        """
                    
                    folium.Polygon(
                        locations=polygon_points,
                        color=colors[cluster_id + 1],
                        weight=2,
                        fill=True,
                        fill_color=colors[cluster_id + 1],
                        fill_opacity=0.2,
                        popup=popup_content
                    ).add_to(carte)
                    
                except Exception as e:
                    print(f"Erreur lors de la création du polygone pour le cluster {cluster_id}: {str(e)}")
                    continue
        
        # Ajouter les points si l'option est activée
        if show_points:
            # Ajouter chaque point à la carte avec la couleur de son cluster
            for idx, row in df.iterrows():
                color_idx = row['cluster'] + 1 if row['cluster'] >= 0 else 0
                
                # Utiliser le tag le plus représentatif dans le popup
                cluster_name = cluster_tags[row['cluster']] if row['cluster'] >= 0 else "Non clustérisé"
                
                # Créer le lien Flickr
                flickr_link = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
                
                # Créer le popup avec le lien HTML et un style pour une largeur fixe
                popup_content = f"""
                <div style="min-width: 200px;">
                Cluster: {cluster_name}<br>
                <a href="{flickr_link}" target="_blank">Voir la photo sur Flickr</a>
                </div>
                """
                
                # Ajuster la taille et l'opacité selon que le point est dans un cluster ou non
                radius = 5 if row['cluster'] >= 0 else 3
                opacity = 0.7 if row['cluster'] >= 0 else 0.15
                
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=radius,  # Plus petit pour les points non clusterisés
                    color=colors[color_idx],
                    fill=True,
                    popup=popup_content,
                    fill_opacity=opacity  # Plus transparent pour les points non clusterisés
                ).add_to(carte)
        
        # Sauvegarder la carte en HTML
        carte.save('carte_photos.html')
        webbrowser.open('file://' + os.path.realpath('carte_photos.html'))
        

    except Exception as e:
        print(f"Erreur dans map_visualization.main(): {str(e)}")
        raise  # Propager l'erreur vers interface.py

if __name__ == "__main__":
    # Configuration par défaut si exécuté directement
    df = pd.read_csv('flickr_data_cleaned.csv', low_memory=False)
    df = df.head(10000)
    clustering_algo = DBSCAN(eps=0.0003, min_samples=5)
    N = 100
    show_points = True
    nb_points_cluster = 1000
    main() 