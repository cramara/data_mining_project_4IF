import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import map_visualization
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tkcalendar import DateEntry
from datetime import datetime
import threading
import webbrowser
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import os

class DataMiningInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface Data Mining")
        
        # Vérifier que le dossier cluster_plots existe
        if not os.path.exists('cluster_plots'):
            os.makedirs('cluster_plots')
        
        # Valeurs par défaut
        self.default_values = {
            'eps': "0.0003",
            'min_samples': "5",
            'n_clusters': "10",
            'n_points': "10000",
            'n_common_tags': "100",
            'data_file': "flickr_data_cleaned.csv",
            'algo': "DBSCAN",
            'display_points': "2000"
        }
        
        # Initialiser les dates min et max
        self.start_year = 2010
        self.end_year = 2018
        try:
            if Path(self.default_values['data_file']).exists():
                df = pd.read_csv(self.default_values['data_file'], low_memory=False)
                df['date_taken'] = pd.to_datetime(df['date_taken'])
                self.start_year = df['date_taken'].dt.year.min()
                self.end_year = df['date_taken'].dt.year.max()
        except Exception as e:
            print(f"Erreur lors de l'initialisation des dates: {e}")
        
        # Variables pour les paramètres
        self.eps_var = tk.StringVar(value=self.default_values['eps'])
        self.min_samples_var = tk.StringVar(value=self.default_values['min_samples'])
        self.n_clusters_var = tk.StringVar(value=self.default_values['n_clusters'])
        self.n_points_var = tk.StringVar(value=self.default_values['n_points'])
        self.n_common_tags_var = tk.StringVar(value=self.default_values['n_common_tags'])
        self.data_file_path = tk.StringVar(value=self.default_values['data_file'])
        self.algo_var = tk.StringVar(value=self.default_values['algo'])
        self.search_var = tk.StringVar()
        self.keep_search_tag_var = tk.BooleanVar(value=False)
        self.display_points_var = tk.StringVar(value=self.default_values['display_points'])
        
        # Variables pour les labels
        self.n_clusters_label = None
        self.k_range_label = None
        self.n_points_label = None
        self.display_points_label = None
        self.n_common_tags_label = None
        
        # Variables pour les sliders k_min et k_max
        self.k_min_var = tk.StringVar(value="2")
        self.k_max_var = tk.StringVar(value="20")
        
        # Ajouter les variables pour les dates
        self.date_start_var = tk.StringVar()
        self.date_end_var = tk.StringVar()
        
        # Stocker les données des clusters
        self.cluster_data = None
        
        # Ajouter une variable pour l'affichage des points
        self.show_points_var = tk.BooleanVar(value=True)
        
        # Ajouter la variable pour les graphiques temporels
        self.show_time_plots_var = tk.BooleanVar(value=True)
        self.time_grouping_var = tk.StringVar(value="mois")  # Changer la valeur par défaut en "mois"
        
        # Création des widgets dans le bon ordre
        self.create_file_frame()
        self.create_actions_frame()  # Créer d'abord les boutons d'action
        self.create_parameters_frame()  # Puis les paramètres qui utilisent ces boutons
        
    def create_file_frame(self):
        file_frame = ttk.LabelFrame(self.root, text="Sélection des données", padding="10")
        file_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Frame pour la recherche et suggestions
        search_frame = ttk.Frame(file_frame)
        search_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Barre de recherche
        ttk.Label(search_frame, text="Rechercher par tag:").grid(row=0, column=0, sticky="w")
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(search_frame, text="Rechercher", command=self.filter_by_tag).grid(row=0, column=2, padx=5)
        
        # Suggestions de tags
        suggestions_frame = ttk.Frame(search_frame)
        suggestions_frame.grid(row=1, column=0, columnspan=3, pady=2)
        
        ttk.Label(suggestions_frame, text="Suggestions:").grid(row=0, column=0, padx=2)
        
        suggestions = ["lumieres", "insa", "bellecour", "confluence"]
        for i, tag in enumerate(suggestions):
            ttk.Button(suggestions_frame, text=tag, 
                      command=lambda t=tag: self.apply_suggestion(t)).grid(row=0, column=i+1, padx=2)
        
        # Case à cocher pour conserver le tag recherché
        ttk.Checkbutton(file_frame, text="Retirer recherché de la liste des tags exclus", 
                       variable=self.keep_search_tag_var).grid(row=1, column=0, columnspan=3, pady=2)
        
        # Affichage du chemin du fichier
        self.file_label = ttk.Label(file_frame, textvariable=self.data_file_path, 
                                  wraplength=300)
        self.file_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Bouton pour sélectionner le fichier
        ttk.Button(file_frame, text="Choisir un fichier", 
                  command=self.select_file).grid(row=3, column=0, columnspan=3, pady=5)
        
        # Ajouter un frame pour la sélection de dates
        date_frame = ttk.LabelFrame(file_frame, text="Filtrer par période", padding="5")
        date_frame.grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")
        
        # Date de début
        ttk.Label(date_frame, text="Du:").grid(row=0, column=0, padx=5)
        self.date_start = DateEntry(date_frame, width=12, 
                                  background='darkblue', foreground='white', 
                                  borderwidth=2, year=self.start_year,
                                  date_pattern='dd/mm/yyyy',
                                  textvariable=self.date_start_var)
        self.date_start.grid(row=0, column=1, padx=5, pady=5)
        
        # Date de fin
        ttk.Label(date_frame, text="Au:").grid(row=0, column=2, padx=5)
        self.date_end = DateEntry(date_frame, width=12, 
                                background='darkblue', foreground='white', 
                                borderwidth=2, year=self.end_year,
                                date_pattern='dd/mm/yyyy',
                                textvariable=self.date_end_var)
        self.date_end.grid(row=0, column=3, padx=5, pady=5)
        
        # Case à cocher pour activer le filtre temporel
        self.use_date_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(date_frame, text="Activer le filtre temporel", 
                       variable=self.use_date_filter).grid(row=0, column=4, padx=5)
        
    def create_parameters_frame(self):
        # Frame principal pour tous les paramètres
        main_params_frame = ttk.Frame(self.root)
        main_params_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Frame pour les paramètres de clustering
        clustering_frame = ttk.LabelFrame(main_params_frame, text="Paramètres de clustering", padding="10")
        clustering_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Choix de l'algorithme
        ttk.Label(clustering_frame, text="Algorithme:").grid(row=0, column=0, sticky="w")
        algo_combo = ttk.Combobox(clustering_frame, textvariable=self.algo_var, 
                                values=["DBSCAN", "K-means"], state="readonly")
        algo_combo.grid(row=0, column=1, padx=5, columnspan=2)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algo_change)
        
        # Frame pour les paramètres DBSCAN
        self.dbscan_frame = ttk.Frame(clustering_frame)
        self.dbscan_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Epsilon avec champ de texte
        ttk.Label(self.dbscan_frame, text="Epsilon:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.eps_var, width=10).grid(row=0, column=1, padx=5)
        
        # Min Samples avec champ de texte
        ttk.Label(self.dbscan_frame, text="Min Samples:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.min_samples_var, width=10).grid(row=1, column=1, padx=5)
        
        # Frame pour les paramètres K-means
        self.kmeans_frame = ttk.Frame(clustering_frame)
        self.kmeans_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Initialiser les labels avant de créer les scales
        self.n_clusters_label = ttk.Label(self.kmeans_frame, text=self.default_values['n_clusters'])
        self.k_range_label = ttk.Label(self.kmeans_frame, text="2-20")
        self.n_points_label = ttk.Label(clustering_frame, text=self.default_values['n_points'])
        self.n_common_tags_label = ttk.Label(clustering_frame, text=self.default_values['n_common_tags'])
        
        # Nombre de clusters avec slider
        ttk.Label(self.kmeans_frame, text="Nombre de clusters:").grid(row=0, column=0, sticky="w")
        n_clusters_scale = ttk.Scale(self.kmeans_frame,
                                   from_=2,
                                   to=50,
                                   orient="horizontal",
                                   length=200,
                                   command=lambda v: self.update_n_clusters(v))
        n_clusters_scale.grid(row=0, column=1, padx=5, sticky="ew")
        n_clusters_scale.set(int(self.default_values['n_clusters']))
        self.n_clusters_label.grid(row=0, column=2, padx=5)
        
        # Paramètres pour la méthode du coude
        ttk.Label(self.kmeans_frame, text="Plage k:").grid(row=1, column=0, sticky="w")
        k_range_frame = ttk.Frame(self.kmeans_frame)
        k_range_frame.grid(row=1, column=1, columnspan=2, sticky="ew")
        
        # Frame pour les sliders et leurs valeurs
        k_min_frame = ttk.Frame(k_range_frame)
        k_min_frame.grid(row=0, column=0, padx=(0,5))
        k_max_frame = ttk.Frame(k_range_frame)
        k_max_frame.grid(row=0, column=1, padx=(5,0))
        
        # Créer les labels pour afficher les valeurs
        self.k_min_value_label = ttk.Label(k_min_frame, text="2")
        self.k_min_value_label.grid(row=1, column=0)
        
        self.k_max_value_label = ttk.Label(k_max_frame, text="20")
        self.k_max_value_label.grid(row=1, column=0)
        
        # Créer les sliders
        self.k_min_scale = ttk.Scale(k_min_frame,
                                    from_=2,
                                    to=20,
                                    orient="horizontal",
                                    length=95)
        self.k_min_scale.grid(row=0, column=0)
        
        self.k_max_scale = ttk.Scale(k_max_frame,
                                    from_=2,
                                    to=50,
                                    orient="horizontal",
                                    length=95)
        self.k_max_scale.grid(row=0, column=0)
        
        # Configurer les valeurs initiales
        self.k_min_scale.set(2)
        self.k_max_scale.set(20)
        
        # Ajouter les commandes
        self.k_min_scale.configure(command=lambda v: self.update_k_min(v))
        self.k_max_scale.configure(command=lambda v: self.update_k_max(v))
        
        self.k_range_label.grid(row=1, column=2, padx=5)
        
        # Nombre de points pour le clustering avec slider
        ttk.Label(clustering_frame, text="Nombre de points pour clustering:").grid(row=4, column=0, sticky="w")
        n_points_scale = ttk.Scale(clustering_frame,
                                 from_=1000,
                                 to=200000,
                                 orient="horizontal",
                                 length=200,
                                 command=lambda v: self.update_n_points(v))
        n_points_scale.grid(row=4, column=1, padx=5, sticky="ew")
        n_points_scale.set(int(self.default_values['n_points']))
        self.n_points_label.grid(row=4, column=2, padx=5)
        
        # Frame pour les paramètres d'affichage
        display_frame = ttk.LabelFrame(main_params_frame, text="Paramètres d'affichage", padding="10")
        display_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Nombre de points à afficher
        points_display_frame = ttk.Frame(display_frame)
        points_display_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(points_display_frame, text="Nombre de points à afficher:").grid(row=0, column=0, sticky="w")
        display_points_scale = ttk.Scale(points_display_frame,
                                       from_=100,
                                       to=5000,
                                       orient="horizontal",
                                       length=200,
                                       command=lambda v: self.update_display_points(v))
        display_points_scale.grid(row=0, column=1, padx=5, sticky="ew")
        self.display_points_label = ttk.Label(points_display_frame, text=self.default_values['display_points'])
        self.display_points_label.grid(row=0, column=2, padx=5)
        
        # Case à cocher pour afficher/masquer les points
        ttk.Checkbutton(points_display_frame, text="Afficher les points", 
                       variable=self.show_points_var).grid(row=1, column=0, columnspan=3, pady=2)
        
        # Tags communs à exclure
        tags_frame = ttk.Frame(display_frame)
        tags_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(tags_frame, text="Tags communs à exclure:").grid(row=0, column=0, sticky="w")
        n_common_tags_scale = ttk.Scale(tags_frame,
                                      from_=0,
                                      to=200,
                                      orient="horizontal",
                                      length=200,
                                      command=lambda v: self.update_n_common_tags(v))
        n_common_tags_scale.grid(row=0, column=1, padx=5, sticky="ew")
        n_common_tags_scale.set(int(self.default_values['n_common_tags']))
        self.n_common_tags_label.grid(row=0, column=2, padx=5)
        
        # Options des graphiques temporels
        time_plots_frame = ttk.Frame(display_frame)
        time_plots_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Checkbutton(time_plots_frame, text="Activer graphiques temporels", 
                        variable=self.show_time_plots_var).grid(row=0, column=0, padx=5)
        
        ttk.Label(time_plots_frame, text="Regrouper par:").grid(row=0, column=1, padx=5)
        time_grouping = ttk.Combobox(time_plots_frame, 
                                    textvariable=self.time_grouping_var,
                                    values=["mois", "année"],
                                    state="readonly",
                                    width=10)
        time_grouping.grid(row=0, column=2, padx=5)
        
        # Bouton de réinitialisation
        ttk.Button(main_params_frame, text="Réinitialiser les paramètres", 
                  command=self.reset_to_defaults).grid(row=2, column=0, pady=10)
        
        # Afficher les paramètres de l'algorithme sélectionné
        self.on_algo_change(None)
        
    def on_algo_change(self, event):
        """Affiche/cache les paramètres et boutons selon l'algorithme choisi"""
        if self.algo_var.get() == "DBSCAN":
            self.dbscan_frame.grid()
            self.kmeans_frame.grid_remove()
        else:
            self.dbscan_frame.grid_remove()
            self.kmeans_frame.grid()
        
        # Mettre à jour les boutons d'action
        self.update_action_buttons()
        
    def create_actions_frame(self):
        actions_frame = ttk.LabelFrame(self.root, text="Actions", padding="10")
        actions_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        # Frame pour centrer les boutons
        buttons_frame = ttk.Frame(actions_frame)
        buttons_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configurer le gestionnaire de grille pour centrer
        actions_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(0, weight=1)
        
        # Boutons communs
        ttk.Button(buttons_frame, text="Générer la carte", 
                  command=self.generate_map).grid(row=0, column=0, pady=5)
        
        # Boutons spécifiques aux algorithmes
        
        self.elbow_button = ttk.Button(buttons_frame, text="Méthode du Coude (Elbow)", 
                                     command=self.elbow_method)
        self.elbow_button.grid(row=2, column=0, pady=5)
        
        # Afficher/cacher les boutons selon l'algorithme initial
        self.update_action_buttons()
        
    def reset_to_defaults(self):
        """Réinitialise tous les paramètres à leurs valeurs par défaut"""
        self.eps_var.set(self.default_values['eps'])
        self.min_samples_var.set(self.default_values['min_samples'])
        self.n_clusters_var.set(self.default_values['n_clusters'])
        self.n_points_var.set(self.default_values['n_points'])
        self.n_common_tags_var.set(self.default_values['n_common_tags'])
        self.data_file_path.set(self.default_values['data_file'])
        self.algo_var.set(self.default_values['algo'])
        self.display_points_var.set(self.default_values['display_points'])
        self.show_time_plots_var.set(True)
        self.time_grouping_var.set("mois")
        messagebox.showinfo("Réinitialisation", "Les paramètres ont été réinitialisés aux valeurs par défaut.")
        
    def select_file(self):
        filetypes = (
            ('Fichiers CSV', '*.csv'),
            ('Tous les fichiers', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Choisir un fichier de données',
            initialdir='.',
            filetypes=filetypes
        )
        
        if filename:
            self.data_file_path.set(filename)
        

    
    def plot_cluster_frequentation(self, cluster_id):
        """Affiche un graphique de la fréquentation pour un cluster donné"""
        if self.cluster_data is None or cluster_id not in self.cluster_data:
            messagebox.showerror("Erreur", "Données du cluster non disponibles")
            return
        
        # Récupérer les données du cluster
        cluster_df = self.cluster_data[cluster_id]
        
        # Convertir les dates en datetime si nécessaire
        cluster_df['date_taken'] = pd.to_datetime(cluster_df['date_taken'])
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        
        # Compter le nombre de photos par jour
        daily_counts = cluster_df.groupby(cluster_df['date_taken'].dt.date).size()
        
        # Tracer le graphique
        sns.lineplot(data=daily_counts)
        
        plt.title(f'Fréquentation du cluster {cluster_id} au fil du temps')
        plt.xlabel('Date')
        plt.ylabel('Nombre de photos')
        plt.xticks(rotation=45)
        
        # Formater les dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        plt.show()
    
    def generate_map(self):
        try:
            # Créer et afficher la fenêtre de chargement
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Génération en cours...")
            loading_window.geometry("300x150")
            
            # Centrer la fenêtre de chargement
            loading_window.transient(self.root)
            loading_window.grab_set()
            loading_window.geometry("+%d+%d" % (
                self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
                self.root.winfo_rooty() + self.root.winfo_height()//2 - 75))
            
            # Ajouter un message et une barre de progression
            ttk.Label(loading_window, text="Génération de la carte en cours...", 
                     padding=20).pack()
            progress = ttk.Progressbar(loading_window, mode='indeterminate')
            progress.pack(padx=20, pady=10, fill='x')
            progress.start(10)
            
            # Mettre à jour l'interface
            loading_window.update()
            
            try:
                # Vérification et chargement des données
                if not Path(self.data_file_path.get()).exists():
                    loading_window.destroy()
                    error_msg = "Le fichier de données n'existe pas!"
                    print(f"Erreur: {error_msg}")
                    messagebox.showerror("Erreur", error_msg)
                    return
                
                # Chargement de toutes les données
                df = pd.read_csv(self.data_file_path.get(), low_memory=False)
                
                # Appliquer le filtre temporel si activé
                if self.use_date_filter.get():
                    try:
                        # Convertir les dates sélectionnées au format YYYY-MM-DD
                        start_date = datetime.strptime(self.date_start_var.get(), "%d/%m/%Y").strftime("%Y-%m-%d")
                        end_date = datetime.strptime(self.date_end_var.get(), "%d/%m/%Y").strftime("%Y-%m-%d")
                        
                        # Convertir la colonne date_taken en datetime si ce n'est pas déjà fait
                        df['date_taken'] = pd.to_datetime(df['date_taken'])
                        
                        # Filtrer les données selon la période
                        mask = (df['date_taken'].dt.date >= pd.to_datetime(start_date).date()) & \
                              (df['date_taken'].dt.date <= pd.to_datetime(end_date).date())
                        df = df[mask]
                        
                        if len(df) == 0:
                            messagebox.showinfo("Résultat", "Aucun point trouvé dans cette période")
                            return
                    except Exception as e:
                        messagebox.showerror("Erreur", 
                            f"Erreur lors du filtrage par date: {str(e)}\n"
                            "Vérifiez le format des dates.")
                        return
                
                # Appliquer le filtre de tag si un tag est spécifié
                search_term = self.search_var.get().lower().strip()
                if search_term:
                    mask = df['tags'].fillna('').str.lower().str.contains(search_term)
                    df = df[mask]
                    if len(df) == 0:
                        messagebox.showinfo("Résultat", "Aucun point trouvé avec ce tag")
                        return
                    
                    # Ajouter les attributs pour le traitement des tags
                    df.search_term = search_term
                    df.keep_search_tag = self.keep_search_tag_var.get()
                
                # Faire le clustering sur tous les points
                if self.algo_var.get() == "DBSCAN":
                    clustering_algo = DBSCAN(
                        eps=float(self.eps_var.get()),
                        min_samples=int(self.min_samples_var.get())
                    )
                else:
                    clustering_algo = KMeans(
                        n_clusters=min(int(self.n_clusters_var.get()), len(df)),
                        random_state=42
                    )
                
                # Appliquer le clustering sur tous les points
                df['cluster'] = clustering_algo.fit_predict(df[['lat', 'long']].values)
                
                # Sélectionner un échantillon aléatoire pour l'affichage si nécessaire
                max_display_points = int(self.display_points_var.get())
                if len(df) > max_display_points:
                    # Échantillonnage stratifié par cluster pour maintenir la distribution
                    display_df = pd.DataFrame()
                    for cluster in df['cluster'].unique():
                        cluster_data = df[df['cluster'] == cluster]
                        # Calculer le nombre de points à prendre de ce cluster
                        n_points = int(max_display_points * (len(cluster_data) / len(df)))
                        if n_points > 0:  # S'assurer qu'on prend au moins 1 point
                            sampled = cluster_data.sample(n=min(n_points, len(cluster_data)), 
                                                        random_state=42)
                            display_df = pd.concat([display_df, sampled])
                    
                    # S'assurer qu'on a exactement max_display_points
                    if len(display_df) < max_display_points:
                        remaining = max_display_points - len(display_df)
                        additional = df[~df.index.isin(display_df.index)].sample(n=remaining, 
                                                                               random_state=42)
                        display_df = pd.concat([display_df, additional])
                else:
                    display_df = df
                
                # Stocker les données par cluster
                self.cluster_data = {}
                for cluster_id in df['cluster'].unique():
                    self.cluster_data[cluster_id] = df[df['cluster'] == cluster_id].copy()
                
                # Continuer avec la génération de la carte
                map_visualization.df = df
                map_visualization.nb_points_cluster = self.n_points_var.get()
                map_visualization.clustering_algo = clustering_algo
                map_visualization.N = int(self.n_common_tags_var.get())
                map_visualization.show_points = self.show_points_var.get()
                map_visualization.show_time_plots = self.show_time_plots_var.get()
                map_visualization.time_grouping = self.time_grouping_var.get()
                
                try:
                    map_visualization.main()
                except Exception as e:
                    loading_window.destroy()
                    error_msg = f"Erreur lors de la génération de la carte: {str(e)}"
                    print(f"Erreur: {error_msg}")
                    raise
                
                # Mise à jour du message de succès
                total_points = self.n_points_var.get()
                displayed_points = len(display_df)
                message = f"La carte a été générée avec {displayed_points} points maximum affichés sur {total_points} points maximum"
                
                if search_term:
                    message += f" contenant le tag '{search_term}'"
                if self.use_date_filter.get():
                    message += f"\nPériode : du {self.date_start_var.get()} au {self.date_end_var.get()}"
                
                # Fermer la fenêtre de chargement
                loading_window.destroy()
                
                print(f"Succès: {message}")
                messagebox.showinfo("Succès", message + "!")
                
            except Exception as e:
                # S'assurer que la fenêtre de chargement est fermée en cas d'erreur
                loading_window.destroy()
                error_msg = f"Une erreur est survenue: {str(e)}"
                print(f"Erreur: {error_msg}")
                messagebox.showerror("Erreur", error_msg)
            
        except Exception as e:
            # En cas d'erreur lors de la création de la fenêtre de chargement
            error_msg = f"Une erreur est survenue: {str(e)}"
            print(f"Erreur: {error_msg}")
            messagebox.showerror("Erreur", error_msg)

    def elbow_method(self):
        try:
            # Charger et préparer les données
            if not Path(self.data_file_path.get()).exists():
                messagebox.showerror("Erreur", "Le fichier de données n'existe pas!")
                return
            
            # Vérifier les valeurs de k_min et k_max
            k_min = int(self.k_min_scale.get())
            k_max = int(self.k_max_scale.get())
            
            if k_min < 2:
                messagebox.showerror("Erreur", "k_min doit être supérieur ou égal à 2!")
                return
            if k_max <= k_min:
                messagebox.showerror("Erreur", "k_max doit être supérieur à k_min!")
                return
            if k_max > 100:
                if not messagebox.askyesno("Attention", 
                    "Un grand nombre de clusters peut prendre beaucoup de temps à calculer. Continuer?"):
                    return
            
            # Créer une fenêtre de chargement plus grande
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Calcul de la méthode du coude")
            progress_window.geometry("400x200")  # Fenêtre plus grande
            
            # Centrer la fenêtre
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_window.geometry("+%d+%d" % (
                self.root.winfo_rootx() + self.root.winfo_width()//2 - 200,
                self.root.winfo_rooty() + self.root.winfo_height()//2 - 100))
            
            # Frame pour organiser les éléments
            info_frame = ttk.Frame(progress_window, padding="20")
            info_frame.pack(fill='both', expand=True)
            
            # Titre
            ttk.Label(info_frame, 
                     text="Calcul de la méthode du coude en cours...",
                     font=('Helvetica', 12, 'bold')).pack(pady=(0, 20))
            
            # Label pour le statut
            progress_label = ttk.Label(info_frame, text="Initialisation...", font=('Helvetica', 10))
            progress_label.pack(pady=(0, 10))
            
            # Barre de progression
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(info_frame, 
                                         variable=progress_var,
                                         maximum=k_max - k_min + 1,
                                         length=300)
            progress_bar.pack(pady=(0, 10))
            
            # Label pour le nombre de clusters en cours
            cluster_label = ttk.Label(info_frame, text="", font=('Helvetica', 10))
            cluster_label.pack()
            
            progress_window.update()
            
            # Charger les données et faire les calculs
            df = pd.read_csv(self.data_file_path.get(), low_memory=False)
            df = df.head(int(self.n_points_var.get()))
            X = df[['lat', 'long']].values
            
            # Calculer l'inertie et le score silhouette
            k_range = range(k_min, k_max + 1)
            inertias = []
            silhouette_scores = []
            
            for i, k in enumerate(k_range):
                progress_var.set(i)
                progress_label.config(text=f"Calcul des clusters...")
                cluster_label.config(text=f"Nombre de clusters: {k}")
                progress_window.update()
                
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            
            progress_window.destroy()
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Graphique de l'inertie (méthode du coude)
            ax1.plot(k_range, inertias, 'bx-')
            ax1.set_xlabel('k (nombre de clusters)')
            ax1.set_ylabel('Inertie')
            ax1.set_title('Méthode du coude')
            
            # Graphique du score silhouette
            ax2.plot(k_range, silhouette_scores, 'rx-')
            ax2.set_xlabel('k (nombre de clusters)')
            ax2.set_ylabel('Score Silhouette')
            ax2.set_title('Score Silhouette vs. k')
            
            plt.tight_layout()
            plt.show()
            
            # Trouver le meilleur k selon le score silhouette
            best_k = k_range[np.argmax(silhouette_scores)]
            messagebox.showinfo("Résultat", 
                f"Selon le score silhouette, le nombre optimal de clusters est {best_k}.\n"
                f"Vous pouvez aussi utiliser le graphique de la méthode du coude pour "
                f"choisir le nombre de clusters.")
            
            # Mettre à jour automatiquement le nombre de clusters
            self.n_clusters_var.set(str(best_k))
            
        except ValueError as ve:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides pour k_min et k_max")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

    def update_action_buttons(self):
        """Met à jour l'affichage des boutons selon l'algorithme sélectionné"""
        if self.algo_var.get() == "DBSCAN":
            self.elbow_button.grid_remove()
        else:
            self.elbow_button.grid()

    def filter_by_tag(self):
        """Vérifie simplement si le tag existe dans les données"""
        try:
            search_term = self.search_var.get().lower().strip()
            if not search_term:
                messagebox.showwarning("Attention", "Veuillez entrer un tag à rechercher")
                return
                
            # Vérification et chargement des données
            if not Path(self.data_file_path.get()).exists():
                messagebox.showerror("Erreur", "Le fichier de données n'existe pas!")
                return
                
            # Vérification rapide de l'existence du tag
            df = pd.read_csv(self.data_file_path.get(), low_memory=False)
            df = df.head(int(self.n_points_var.get()))
            mask = df['tags'].fillna('').str.lower().str.contains(search_term)
            count = mask.sum()
            
            if count == 0:
                messagebox.showinfo("Résultat", "Aucun point ne contient ce tag")
            else:
                messagebox.showinfo("Résultat", 
                    f"{count} points contiennent le tag '{search_term}'\n"
                    "Cliquez sur 'Générer la carte' pour visualiser ces points.")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

    def apply_suggestion(self, tag):
        """Applique le tag suggéré à la barre de recherche et lance la recherche"""
        self.search_var.set(tag)
        self.filter_by_tag()

    def update_eps(self, value):
        """Met à jour la valeur d'epsilon"""
        val = float(float(value))
        self.eps_var.set(f"{val:.6f}")

    def update_min_samples(self, value):
        """Met à jour la valeur de min_samples"""
        val = int(float(value))
        self.min_samples_var.set(str(val))

    def update_n_clusters(self, value):
        """Met à jour le nombre de clusters"""
        val = int(float(value))
        self.n_clusters_var.set(str(val))
        if self.n_clusters_label:
            self.n_clusters_label.config(text=str(val))

    def update_k_min(self, value):
        """Met à jour k_min"""
        val = int(float(value))
        self.k_min_var.set(str(val))
        self.k_min_value_label.config(text=str(val))
        k_max = int(float(self.k_max_scale.get()))
        
        # Ajuster k_max si nécessaire
        if val >= k_max:
            self.k_max_scale.set(val + 1)
            k_max = val + 1
        
        self.k_range_label.config(text=f"{val}-{k_max}")

    def update_k_max(self, value):
        """Met à jour k_max"""
        val = int(float(value))
        self.k_max_var.set(str(val))
        self.k_max_value_label.config(text=str(val))
        k_min = int(float(self.k_min_scale.get()))
        
        # Ajuster k_min si nécessaire
        if val <= k_min:
            self.k_min_scale.set(val - 1)
            k_min = val - 1
        
        self.k_range_label.config(text=f"{k_min}-{val}")

    def update_n_points(self, value):
        """Met à jour le nombre de points"""
        val = int(float(value))
        self.n_points_var.set(str(val))
        if self.n_points_label:
            self.n_points_label.config(text=str(val))

    def update_n_common_tags(self, value):
        """Met à jour le nombre de tags communs"""
        val = int(float(value))
        self.n_common_tags_var.set(str(val))
        if self.n_common_tags_label:
            self.n_common_tags_label.config(text=str(val))

    def update_display_points(self, value):
        """Met à jour le nombre de points à afficher"""
        val = int(float(value))
        self.display_points_var.set(str(val))
        if self.display_points_label:
            self.display_points_label.config(text=str(val))

def main():
    root = tk.Tk()
    app = DataMiningInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main() 