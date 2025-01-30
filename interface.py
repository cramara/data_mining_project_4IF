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

class DataMiningInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface Data Mining")
        
        # Valeurs par défaut
        self.default_values = {
            'eps': "0.0003",
            'min_samples': "5",
            'n_clusters': "10",
            'n_points': "10000",
            'n_common_tags': "100",
            'data_file': "flickr_data_cleaned.csv",
            'algo': "DBSCAN",
            'display_points': "10000"
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
        
        suggestions = ["fete des lumieres", "insa", "bellecour", "confluence"]
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
        params_frame = ttk.LabelFrame(self.root, text="Paramètres", padding="10")
        params_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Choix de l'algorithme
        ttk.Label(params_frame, text="Algorithme:").grid(row=0, column=0, sticky="w")
        algo_combo = ttk.Combobox(params_frame, textvariable=self.algo_var, 
                                values=["DBSCAN", "K-means"], state="readonly")
        algo_combo.grid(row=0, column=1, padx=5, columnspan=2)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algo_change)
        
        # Frame pour les paramètres DBSCAN
        self.dbscan_frame = ttk.Frame(params_frame)
        self.dbscan_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Epsilon avec champ de texte
        ttk.Label(self.dbscan_frame, text="Epsilon:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.eps_var, width=10).grid(row=0, column=1, padx=5)
        
        # Min Samples avec champ de texte
        ttk.Label(self.dbscan_frame, text="Min Samples:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.min_samples_var, width=10).grid(row=1, column=1, padx=5)
        
        # Frame pour les paramètres K-means
        self.kmeans_frame = ttk.Frame(params_frame)
        self.kmeans_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
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
        
        self.n_clusters_label = ttk.Label(self.kmeans_frame, text=self.default_values['n_clusters'])
        self.n_clusters_label.grid(row=0, column=2, padx=5)
        
        # Paramètres pour la méthode du coude
        ttk.Label(self.kmeans_frame, text="Plage k:").grid(row=1, column=0, sticky="w")
        k_range_frame = ttk.Frame(self.kmeans_frame)
        k_range_frame.grid(row=1, column=1, columnspan=2, sticky="ew")
        
        self.k_min_scale = ttk.Scale(k_range_frame,
                                    from_=2,
                                    to=20,
                                    orient="horizontal",
                                    length=95,
                                    command=lambda v: self.update_k_min(v))
        self.k_min_scale.grid(row=0, column=0, padx=(0,5))
        self.k_min_scale.set(2)
        
        self.k_max_scale = ttk.Scale(k_range_frame,
                                    from_=2,
                                    to=50,
                                    orient="horizontal",
                                    length=95,
                                    command=lambda v: self.update_k_max(v))
        self.k_max_scale.grid(row=0, column=1, padx=(5,0))
        self.k_max_scale.set(20)
        
        self.k_range_label = ttk.Label(self.kmeans_frame, text="2-20")
        self.k_range_label.grid(row=1, column=2, padx=5)
        
        # Nombre de points pour le clustering avec slider
        ttk.Label(params_frame, text="Nombre de points pour clustering:").grid(row=2, column=0, sticky="w")
        n_points_scale = ttk.Scale(params_frame,
                                 from_=1000,
                                 to=200000,
                                 orient="horizontal",
                                 length=200,
                                 command=lambda v: self.update_n_points(v))
        n_points_scale.grid(row=2, column=1, padx=5, sticky="ew")
        n_points_scale.set(int(self.default_values['n_points']))
        
        self.n_points_label = ttk.Label(params_frame, text=self.default_values['n_points'])
        self.n_points_label.grid(row=2, column=2, padx=5)
        
        # Nombre de points à afficher avec slider
        ttk.Label(params_frame, text="Nombre de points à afficher:").grid(row=3, column=0, sticky="w")
        display_points_scale = ttk.Scale(params_frame,
                                       from_=1000,
                                       to=50000,
                                       orient="horizontal",
                                       length=200,
                                       command=lambda v: self.update_display_points(v))
        display_points_scale.grid(row=3, column=1, padx=5, sticky="ew")
        display_points_scale.set(int(self.default_values['display_points']))
        
        self.display_points_label = ttk.Label(params_frame, text=self.default_values['display_points'])
        self.display_points_label.grid(row=3, column=2, padx=5)
        
        # Tags communs avec slider
        ttk.Label(params_frame, text="Tags communs à exclure:").grid(row=4, column=0, sticky="w")
        n_common_tags_scale = ttk.Scale(params_frame,
                                      from_=0,
                                      to=200,
                                      orient="horizontal",
                                      length=200,
                                      command=lambda v: self.update_n_common_tags(v))
        n_common_tags_scale.grid(row=4, column=1, padx=5, sticky="ew")
        n_common_tags_scale.set(int(self.default_values['n_common_tags']))
        
        self.n_common_tags_label = ttk.Label(params_frame, text=self.default_values['n_common_tags'])
        self.n_common_tags_label.grid(row=4, column=2, padx=5)
        
        # Bouton de réinitialisation
        ttk.Button(params_frame, text="Réinitialiser les paramètres", 
                  command=self.reset_to_defaults).grid(row=5, column=0, columnspan=3, pady=10)
        
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
        
    def generate_map(self):
        try:
            # Vérification et chargement des données
            if not Path(self.data_file_path.get()).exists():
                messagebox.showerror("Erreur", "Le fichier de données n'existe pas!")
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
            
            # Mise à jour des variables globales pour l'affichage
            map_visualization.df = display_df
            map_visualization.clustering_algo = clustering_algo
            map_visualization.N = int(self.n_common_tags_var.get())
            
            # Exécution de la génération de carte
            map_visualization.main()
            
            # Mise à jour du message de succès
            total_points = len(df)
            displayed_points = len(display_df)
            message = f"La carte a été générée avec {displayed_points} points affichés sur {total_points} points"
            
            if search_term:
                message += f" contenant le tag '{search_term}'"
            if self.use_date_filter.get():
                message += f"\nPériode : du {self.date_start_var.get()} au {self.date_end_var.get()}"
            
            messagebox.showinfo("Succès", message + "!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

    
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
            
            df = pd.read_csv(self.data_file_path.get(), low_memory=False)
            df = df.head(int(self.n_points_var.get()))
            X = df[['lat', 'long']].values
            
            # Calculer l'inertie et le score silhouette pour la plage de k spécifiée
            k_range = range(k_min, k_max + 1)
            inertias = []
            silhouette_scores = []
            
            # Ajouter une barre de progression
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Calcul en cours...")
            progress_var = tk.DoubleVar()
            ttk.Progressbar(progress_window, variable=progress_var, 
                           maximum=len(k_range)).pack(padx=10, pady=10)
            progress_label = ttk.Label(progress_window, text="Calcul des clusters...")
            progress_label.pack(pady=5)
            
            for i, k in enumerate(k_range):
                progress_var.set(i)
                progress_label.config(text=f"Calcul pour k={k}...")
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
        self.n_clusters_label.config(text=str(val))

    def update_k_min(self, value):
        """Met à jour k_min"""
        val = int(float(value))
        self.k_min_var.set(str(val))
        if val >= int(self.k_max_scale.get()):
            self.k_max_scale.set(val + 1)
        self.k_range_label.config(text=f"{val}-{self.k_max_scale.get()}")

    def update_k_max(self, value):
        """Met à jour k_max"""
        val = int(float(value))
        self.k_max_var.set(str(val))
        if val <= int(self.k_min_scale.get()):
            self.k_min_scale.set(val - 1)
        self.k_range_label.config(text=f"{self.k_min_scale.get()}-{val}")

    def update_n_points(self, value):
        """Met à jour le nombre de points"""
        val = int(float(value))
        self.n_points_var.set(str(val))
        self.n_points_label.config(text=str(val))

    def update_n_common_tags(self, value):
        """Met à jour le nombre de tags communs"""
        val = int(float(value))
        self.n_common_tags_var.set(str(val))
        self.n_common_tags_label.config(text=str(val))

    def update_display_points(self, value):
        """Met à jour le nombre de points à afficher"""
        val = int(float(value))
        self.display_points_var.set(str(val))
        self.display_points_label.config(text=str(val))

def main():
    root = tk.Tk()
    app = DataMiningInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main() 