% Maria Bourouisa (2211827)
% Ihsane Majdoubi (2125034)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de tâches
int: n;
% Durée maximal du projet
int: m;
% Limite de ressources pouvant être partagées entre les tâches en cours
int: limite;

% Durée des tâches
array[1..n] of 1..m: duree_tache;
% Ressources requises par les tâches
array[1..n] of 1..m: ressource_tache;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

array[1..n] of var 1..m: debut_tache; % Début des tâches
array[1..n] of var 1..m: fin_tache; % Fin des tâches

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Fin du projet
var int: fin_projet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
%Contrainte: Le projet ne peut pas dépasser la limite de temps m
%Cette constrainte est validé par le fait que la durée des tâches est dans un 
%intervalle de 1 à m, ainsi aucune tâche ne va finir après m

%Contrainte: Besoin d'énergie ne dépasse pas la limite [1] 
constraint cumulative(debut_tache, duree_tache, ressource_tache, limite);

%Contrainte: Pas plus de deux tâches ne démarent en même temps
constraint forall(max_debut in 1..m)(count(debut_tache, max_debut) <= 2);

%Contrainte: Chaque tâche s'effectue au complet (toute sa durée)
constraint forall(i in 1..n) (fin_tache[i] = debut_tache[i] + duree_tache[i]);

%Contrainte: Minimiser le temps de fin du projet
constraint fin_projet = max(fin_tache);
solve minimize fin_projet;
%-----------------------------------------------------------------------------%

output [
  "Envergure du projet  : ", show(fin_projet), "\n",
  "Début des tâches     : ", show(debut_tache), "\n",
  "Fin des tâches       : ", show(fin_tache), "\n",
];

%[1] https://sofdem.github.io/gccat/gccat/Ccumulative.html