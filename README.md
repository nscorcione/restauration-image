# Restauration d'image par gradient conjugué

Bureau d'études Ma316 « Généralisation de l'algorithme du gradient conjugué »,
fait en groupe à l'IPSA (décembre 2025).

**Équipe :** Elyn Piot, Eva Ramos, Etienne Roumiguier, Noa Scorcione,
Lucas Verleyen, Adam Zerhouni.

## Principe

On sélectionne une zone de l'image à la souris (le masque), puis on la remplit à
partir des pixels qui l'entourent. On cherche le remplissage le plus lisse
possible, c'est-à-dire avec un laplacien nul à l'intérieur de la zone.

Sous forme matricielle, ça donne l'équation

```
(Lm X + X Ln) * M = R
```

où `Lm` et `Ln` sont des matrices laplaciennes 1D et `M` le masque. On la résout
avec la méthode du gradient conjugué, un canal de couleur à la fois.

La fonction `image_restauration_couleur_opti` ne fait le calcul que sur un
rectangle autour du masque (avec 3 pixels de marge), ce qui rend le calcul
beaucoup plus rapide que sur l'image entière.

Le même solveur sert aussi dans le BE à calculer un potentiel électrostatique à
partir de charges dessinées à la souris, et à faire du transfert d'une zone d'une
image vers une autre.

## Lancer le projet

```
pip install -r requirements.txt
```

`Ma316_BE_main.py` est découpé en cellules (`#%%`), à lancer une par une, par
exemple dans Spyder ou VS Code.

Commandes dans les fenêtres OpenCV :
- tracé du masque : clic gauche maintenu pour peindre, **Entrée** pour valider ;
- masque en polygone : clic pour placer les points, **Entrée** pour remplir,
  **Retour arrière** pour enlever le dernier point, **Espace** pour terminer ;
- recadrage : tracer un rectangle, puis **Entrée**.

Les résultats sont enregistrés à côté du script (`result_*.jpg`).

## Données

Le dossier `data/` contient les images utilisées pour la restauration :

- `graffiti.jpg` : suppression des graffitis (noir et blanc, couleur, version
  optimisée) ;
- `PAF.png` : image 1920 × 1080 de la Patrouille de France, dont on efface un
  avion ;
- `orig1.jpg` : vieille photo abîmée, utilisée pour la restauration et pour
  montrer la limite de la méthode (floutage du visage) ;
- `piscine.jpg` et `ours.jpg` : transfert de l'ours dans la piscine
  (`transfert_zone_couleur`) ;
- `mur.png`, `grad_F.png` et `ipsa.png` : insertion d'une note manuscrite et du
  logo IPSA sur un mur en briques avec la méthode du gradient dominant
  (`transfert_zone_couleur_mixed`) ;
- `R.npy` : distribution de charges (512 × 512, valeurs -1, 0 et 1) pour le calcul
  du potentiel électrostatique ;
- `aigle.jpg` : autre image à insérer, utilisée dans le rapport avec un tableau
  de Magritte (le tableau n'est pas inclus).

La cellule « Potentiel électrique avec tracé manuel » n'a besoin d'aucun fichier :
on dessine les charges à la souris.

## Fichiers

- `Ma316_BE_lib.py` : gradient conjugué, matrices laplaciennes, tracé du masque
  et fonctions de restauration.
- `Ma316_BE_main.py` : les différentes applications du BE.
- `rapport.pdf` : le rapport complet, avec la partie théorique (équation
  d'Euler-Lagrange à plusieurs variables, convexité, démonstrations sur le
  gradient conjugué) et les applications.

Une version interactive de la restauration est disponible sur mon site.
