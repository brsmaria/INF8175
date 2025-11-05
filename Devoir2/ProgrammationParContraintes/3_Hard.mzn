% Maria Bourouisa (2211827)
% Ihsane Majdoubi (2125034)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de villes à visiter + point de départ de la tournée
int: n;
 % Distance entre chaque villes (symmétrique)
array[1..n,1..n] of int: distance;
 % Distance minimale possible entre deux villes différentes
int: min_val = min([distance[i,j] | i,j in 1..n where distance[i,j] > 0]);
% Distance maximale possible entre deux villes différentes
int: max_val = max([distance[i,j] | i,j in 1..n]);

% Fenêtre de temps (1 = début, 2 = fin) durant laquelle il est possible de jouer pour chaque ville
array[1..n,1..2] of int: fenetre_arrivee;
% Temps maximum pour tout le trajet
int: max_trajet = max([fenetre_arrivee[i,2] | i in 1..n]);
% Fenêtre de temps la plus grande du trajet
int: max_fenetre = max([fenetre_arrivee[i,2] - fenetre_arrivee[i,1]| i in 2..n]);

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Ordre des villes visitées par la tournée
array[1..n] of var 1..n: chemin_tournee;
% Circuit de la tournée, circuit_tournee[i] = j => j est le successeur de i
array[1..n] of var 1..n: circuit_tournee;
% Temps cumulé lors de l'arrivé au point i de la tournée
array[1..n] of var 0..max_trajet: cumul_temps;
% Temps d'attente avant le spectacle en ville i
array[1..n] of var 0..max_fenetre: temps_attente;
% Temps d'attente total 
var 0..max_trajet: total_temps_attentes = sum(temps_attente);

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Temps de trajet à minimiser
var int: temps_trajet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
%Contrainte: Le trajet de termine par un retour à la première ville
constraint chemin_tournee[n] = 1;
constraint forall(i in 1..n-1) (circuit_tournee[chemin_tournee[i]] = chemin_tournee[i+1]);

%Contrainte: Visiter toutes les villes une seule fois
constraint all_different(chemin_tournee);

%Contrainte: Arriver dans chaque ville dans la fenêtre de temps permise + ne pas arriver en retard
%Pour cela on doir dabord calculer le temps cumulé avant d'entrer dans chaque ville
constraint cumul_temps[1] = distance[1, chemin_tournee[1]] + temps_attente[chemin_tournee[1]];
constraint forall(i in 2..n)(
  cumul_temps[i] =
    cumul_temps[i-1] + temps_attente[chemin_tournee[i]]
  + distance[chemin_tournee[i-1], chemin_tournee[i]]
);

%Ensuite on impose les contraintes sur les fenêtres d'arrivée
constraint forall(i in 1..n) (cumul_temps[i]>= fenetre_arrivee[chemin_tournee[i], 1] 
                              /\ cumul_temps[i]<= fenetre_arrivee[chemin_tournee[i], 2]);
  
%Contrainte: Calcul du temps total du trajet
constraint temps_trajet = cumul_temps[n];

% Nous avons utilisé une contrainte qui permet d'optimiser la recherche solve::int_search [2]
%Celle ci nous permet de forcer la priorité dans l'ordre de recherche à travers le choix des paramètres
% Paramètre 1: chemin_tournee qui est l'espace de recherche (l'ordre des villes)
% Paramètre 2: first_fail qui permet de choisir la variable avec le domaine le plus petit en premier donc celle qui a le moins de possibilités
% Paramètre 3: indomain_min qui permet de choisir la plus petite valeur possible dans le domaine de la variable choisie puisque le but est 
% de minimiser le temps de trajet

solve :: int_search(chemin_tournee, first_fail, indomain_min) minimize temps_trajet;
%-----------------------------------------------------------------------------%

output [
  "Chemin de la tournée                   : \(chemin_tournee)\n",
  "Circuit de la tournée                  : \(circuit_tournee)\n",
  "Temps cumulés avant d'entrer en ville  : \(cumul_temps)\n",
  "Temps d'attente avant l'entrée en ville: \(temps_attente)\n",
  "Temps d'attente total                  : \(total_temps_attentes)\n",
  "Temps de trajet total                  : \(temps_trajet)\n",
];

%[2] https://github.com/MiniZinc/MiniZincIDE/blob/master/MiniZincIDE/cheat_sheet.mzn