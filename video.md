

### Nettoyage des données
- qu'est ce qu'on a remarqué au début qui nous semblait bizzare (bcp de doublons, lignes incomplètes, format de date pas pratique ...)
- en quoi notre solution (tout effacer) est justifiable
- 

### Utilisation de l'interface graphique

- L'interface graphique permet de jouer sur pleins de paramètres : 
    - Choix de l'algo de clustering et choix des paramètres de l'algo (de base les meilleurs)
=> Argumenter sur pourquoi DBSCAN est meilleur

    - Possibilité d'utiliser qu'un échantillon des données en les filtrant par tag et intervalle de temps
- Possibilité d'effectuer la méthode du coude avec l'algo K-means pou identifier le meilleur nombre de clusters
- On peut choisir le nombre de tag à retirer parmi les tags les plus communs 


### Interprétation des cluster
- On a pour chaque cluster un nom et un polygone qui défini sa zone
- On peut aussi cliquer sur un point pour aller voir la photo en question

### Nom des clusters
- Suppression des tags les plus communs
- Au début utilisation du plus fréquent pour sortir un tag logique par rapport au cluster
- Puis amélioration 

### Zoom sur une fonctionnalitée
- On a décider d'offrir la possibilité de savoir la répartition temporelle des photos sur différents cluster : cela pourrait permettre à la ville de savoir quand sont les périodes de forte fréquentation (group by mois) mais aussi de voir si un lieu est de moins en moins visité au cours des années et donc réagir en question 


### Discussion autour des la recherche de point d'intéret par clustering avec les data de Flickr
- Données très lié aux touristes, car app étrangère donc pratique pour voir la fréquentation des lieux touristique mais attention cela ne représente qu'une partie de la population dans Lyon
- Les données brut de Flickr peuvent être trompeuses (20 photos faites par la meme personne au meme endroit peuvent faire grandement varier les valeur / clusters) il faut donc faire attention à ne pas prendre plusieurs fois en compte ce genre de post
- Cependant on arrive quand meme facilement à voir les zones d'intérets dans Lyon, avec des noms cohérent, ce qui est déjà bien !
- Si jamais on veut cibler un événement, cela est possible grace aux tags
- 
