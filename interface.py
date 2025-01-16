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
        
        # Création des widgets dans le bon ordre
        self.create_file_frame()
        self.create_actions_frame()  # Créer d'abord les boutons d'action
        self.create_parameters_frame()  # Puis les paramètres qui utilisent ces boutons
        
    def create_file_frame(self):
        file_frame = ttk.LabelFrame(self.root, text="Sélection des données", padding="10")
        file_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Affichage du chemin du fichier
        self.file_label = ttk.Label(file_frame, textvariable=self.data_file_path, 
                                  wraplength=300)
        self.file_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Bouton pour sélectionner le fichier
        ttk.Button(file_frame, text="Choisir un fichier", 
                  command=self.select_file).grid(row=1, column=0, pady=5)
        
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
        
        # Boutons communs
        ttk.Button(actions_frame, text="Générer la carte", 
                  command=self.generate_map).grid(row=0, column=0, pady=5)
        
        # Boutons spécifiques aux algorithmes
        self.dbscan_button = ttk.Button(actions_frame, text="Analyser paramètres DBSCAN", 
                                       command=self.analyze_dbscan)
        self.dbscan_button.grid(row=1, column=0, pady=5)
        
        self.elbow_button = ttk.Button(actions_frame, text="Méthode du Coude (Elbow)", 
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
                
            # Mise à jour des variables globales
            map_visualization.df = pd.read_csv(self.data_file_path.get(), low_memory=False)
            map_visualization.df = map_visualization.df.head(int(self.n_points_var.get()))
            
            # Vérification des colonnes requises
            required_columns = ['lat', 'long', 'tags', 'user', 'id']
            missing_columns = [col for col in required_columns if col not in map_visualization.df.columns]
            if missing_columns:
                messagebox.showerror("Erreur", 
                    f"Colonnes manquantes dans le fichier: {', '.join(missing_columns)}")
                return
            
            # Configuration de l'algorithme choisi
            if self.algo_var.get() == "DBSCAN":
                map_visualization.clustering_algo = DBSCAN(
                    eps=float(self.eps_var.get()),
                    min_samples=int(self.min_samples_var.get())
                )
            else:
                map_visualization.clustering_algo = KMeans(
                    n_clusters=int(self.n_clusters_var.get()),
                    random_state=42
                )
            
            # Mise à jour du nombre de tags communs
            map_visualization.N = int(self.n_common_tags_var.get())
            
            # Exécution de la génération de carte
            map_visualization.main()
            messagebox.showinfo("Succès", "La carte a été générée avec succès!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")
            
    def analyze_dbscan(self):
        messagebox.showinfo("Info", "Fonctionnalité à implémenter: Analyse des paramètres optimaux pour DBSCAN")
    
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
            self.dbscan_button.grid()
            self.elbow_button.grid_remove()
        else:
            self.dbscan_button.grid_remove()
            self.elbow_button.grid()

def main():
    root = tk.Tk()
    app = DataMiningInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main() 