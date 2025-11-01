% Maria Bourouisa (2211827)
% Ihsane Majdoubi (2125034)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

array[1..n] of var costumes: choix_final;
%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role; %l'index c'est l'acteur

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
%Contrainte chaque acteur décide du costume qu’il prendra pour chacun des rôles
% L'utilisation du tableau choix_costumes vérifie cette contrainte

%Contrainte: attribuer un seul rôle à chaque acteur
constraint alldifferent(assignation_role);

%Contrainte: Chaque acteur choisit un costume différent
constraint  forall(i in 1..n)(choix_final[i] = choix_costume[assignation_role[i], i]);
constraint alldifferent(choix_final);

%Contrainte: deux acteurs voisins, selon l’ordre lexicographique, ne jouent pas un rôle adjacent.
%Dans la liste d'assignation de rôles, on compare le rôle de chaque acteur avec celui de l'acteur
%suivant, on ne passe pas par le dernier acteur car la validation aura déjà été faite
constraint forall(i in 1..n-1)(assignation_role[i] != assignation_role[i+1] + 1
                            /\ assignation_role[i] != assignation_role[i+1] - 1);


solve satisfy;
%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];