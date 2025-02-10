# data_mining_project_4IF
Data Mining project: Discover and describe areas of interest and events from geo-located data

package : 
- matplotlib
- tkcalendar 
- pandas
- folium
- sklearn.cluster
- numpy
- scipy.spatial
- colorsys
- collections
- webbrowser
- os
- seaborn

-plotly

pip install pandas, numpy, folium,  scikit-learn, tkcalendar, matplotlib, seaborn, plotly

todo cleaning : 
- justifier le cleaning des données
-effacer les post multiples d'un meme user au meme endroit

totdo carte: 
- faire un lien entre les points et les photos (done)
- faire un algo qui repère les tags les plus utilisés en général et les exclure des tags pour titres (done)
- ajouter un filtre sur une période temporelle 
- sortir pour chaque cluster la période de l'année durant laquelle il y a le plus de monde (générer un graph du nb de photo en fonction du temps)


Présentation :

- Fonctionnalitées : 
     - choix du clustering (Kmeans ou DBScan)
     - choix du nombre de tags à retirer
     - paramètres par défaut sont les meilleurs paramètres
     - possibilité d'effectuer un test Elbow
     - 


- Interprétation des données : 
    - Permet de visualiser les zones d'intêret dans Lyon
    - Possibilité de cibler un événement ou une zone avec le filtre sur les tags
    - Possibilité de filtrer sur une périodes
    - Possibilité de voir pour les cluster les plus importants un graphique de la fréquentation du lieu