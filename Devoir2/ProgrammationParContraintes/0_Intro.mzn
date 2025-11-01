% Maria Bourouisa (2211827)
% Ihsane Majdoubi (2125034)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%
% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%
% On a mis les contraintes dans l'ordre qui a été donné dans l'énoncé 
% car il permet d'éliminer le maximum de possibiltés au début et permettre 
% d'avoir moins d'entrées pour les opérations les plus couteuses
%-----------------------------------------------------------------------------%

constraint chiffres[4] in {0, 2, 4, 6, 8};
constraint forall(chiffre in chiffres)(chiffre != 0 );
constraint alldifferent(chiffres);
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint chiffres[2] > chiffres[4];
constraint sum(chiffres) > 15;
constraint p = chiffres[4] * chiffres[3] * chiffres[2];

constraint nombre = 1000 * chiffres[1] 
                  + 100  * chiffres[2]
                  + 10   * chiffres[3]
                  + 1    * chiffres[4];

solve minimize p;
%-----------------------------------------------------------------------------%

output [
  show(nombre)
];

