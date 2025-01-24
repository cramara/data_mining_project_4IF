import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import map_visualization
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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
            'algo': "DBSCAN"
        }
        
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
        
    def create_parameters_frame(self):
        params_frame = ttk.LabelFrame(self.root, text="Paramètres", padding="10")
        params_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Choix de l'algorithme
        ttk.Label(params_frame, text="Algorithme:").grid(row=0, column=0, sticky="w")
        algo_combo = ttk.Combobox(params_frame, textvariable=self.algo_var, 
                                values=["DBSCAN", "K-means"], state="readonly")
        algo_combo.grid(row=0, column=1, padx=5)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algo_change)
        
        # Frame pour les paramètres DBSCAN
        self.dbscan_frame = ttk.Frame(params_frame)
        self.dbscan_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Label(self.dbscan_frame, text="Epsilon:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.eps_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.dbscan_frame, text="Min Samples:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.dbscan_frame, textvariable=self.min_samples_var, width=10).grid(row=1, column=1, padx=5)
        
        # Frame pour les paramètres K-means
        self.kmeans_frame = ttk.Frame(params_frame)
        self.kmeans_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Label(self.kmeans_frame, text="Nombre de clusters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.kmeans_frame, textvariable=self.n_clusters_var, width=10).grid(row=0, column=1, padx=5)
        
        # Ajout des paramètres pour la méthode du coude
        ttk.Label(self.kmeans_frame, text="Plage k min:").grid(row=1, column=0, sticky="w")
        self.k_min_var = tk.StringVar(value="2")
        ttk.Entry(self.kmeans_frame, textvariable=self.k_min_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(self.kmeans_frame, text="Plage k max:").grid(row=2, column=0, sticky="w")
        self.k_max_var = tk.StringVar(value="20")
        ttk.Entry(self.kmeans_frame, textvariable=self.k_max_var, width=10).grid(row=2, column=1, padx=5)
        
        # Paramètres communs
        ttk.Label(params_frame, text="Nombre de points:").grid(row=2, column=0, sticky="w")
        ttk.Entry(params_frame, textvariable=self.n_points_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(params_frame, text="Tags communs à exclure:").grid(row=3, column=0, sticky="w")
        ttk.Entry(params_frame, textvariable=self.n_common_tags_var, width=10).grid(row=3, column=1, padx=5)
        
        # Bouton de réinitialisation
        ttk.Button(params_frame, text="Réinitialiser les paramètres", 
                  command=self.reset_to_defaults).grid(row=4, column=0, columnspan=2, pady=10)
        
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
                
            # Chargement et préparation des données
            df = pd.read_csv(self.data_file_path.get(), low_memory=False)
            df = df.head(int(self.n_points_var.get()))
            
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
            
            # Mise à jour des variables globales
            map_visualization.df = df
            
            # Configuration de l'algorithme choisi
            if self.algo_var.get() == "DBSCAN":
                map_visualization.clustering_algo = DBSCAN(
                    eps=float(self.eps_var.get()),
                    min_samples=int(self.min_samples_var.get())
                )
            else:
                map_visualization.clustering_algo = KMeans(
                    n_clusters=min(int(self.n_clusters_var.get()), len(df)),
                    random_state=42
                )
            
            # Mise à jour du nombre de tags communs
            map_visualization.N = int(self.n_common_tags_var.get())
            
            # Exécution de la génération de carte
            map_visualization.main()
            
            # Message de succès avec info sur le filtrage si un tag était spécifié
            if search_term:
                messagebox.showinfo("Succès", 
                    f"La carte a été générée avec {len(df)} points contenant le tag '{search_term}'!")
            else:
                messagebox.showinfo("Succès", "La carte a été générée avec succès!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

    
    def elbow_method(self):
        try:
            # Charger et préparer les données
            if not Path(self.data_file_path.get()).exists():
                messagebox.showerror("Erreur", "Le fichier de données n'existe pas!")
                return
            
            # Vérifier les valeurs de k_min et k_max
            k_min = int(self.k_min_var.get())
            k_max = int(self.k_max_var.get())
            
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

def main():
    root = tk.Tk()
    app = DataMiningInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main() 