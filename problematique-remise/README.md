# APP2 - S8 - Solution de la problématique
## Auteurs
* Samuel Laperrière - laps2022
* Raphaël Lebrasseur - lebr2112
* Charles Murphy - murc3002

## Exécution du code pour l'algorithme génétique
Exécuter le fichier `main.py` situé dans `scripts/optimize-ga`.

Sous la section `Define hyperparameters here` (vers ligne 55), on peut faire varier des paramètres. La simulation fera plusieurs passes et produira des graphiques pour montrer les performances des paramètres établis. En fournissant plus d'un élément dans les listes (e.g. `mutation_rate_list`), la simulation sera exécutée plusieurs fois, pour évaluer chaque paramètre. Cela produira en sortie les graphiques présentés dans les diapositives de la défense de la solution.

## Exécution du code de logique floue
Exécuter le fichier `main.py` situé dans `scripts/drive-fuzzy`.

Pour visualiser tous les graphiques présentés dans les diapositives de la défense de la solution, exécuter le fichier `visualisation.py` dans ce même dossier.

Le code de logique floue utilise deux classes, contenues dans des fichiers séparés : `FuzzyController` et `SimpleController`. Le `SimpleController` est simplement une copie du code de `drive-simple`, empaqueté dans une classe. Il est donc possible d'utiliser soit le `SimpleController` ou le `FuzzyController` pour contrôler indépendamment l'accélération, la direction ou la transmission (voir la méthode `drive` dans `main.py`).