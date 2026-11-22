# PIOT Elyn
# RAMOS Eva
# ROUMIGUIER Etienne
# SCORCIONE Noa
# VERLEYEN Lucas 
# ZERHOUNI Adam

#%% === Importations ===

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Ma316_BE_lib as ma
from matplotlib.colors import ListedColormap

#%% === Utilisation du gradient conjugue ===

# Création de la matrice R
R = np.array([[0, 0],
              [0, 0]])

# Création des matrices P et Q 
P = np.array([[2, 1],
              [1, 2]])
Q = P

# Création des conditions initiales/tolérance de convergence/nombre max d'itérations
x0 = np.zeros((2,2))
epsilon = 1e-8
nbiter = 1000

# Définition de l'application linéaire utilisé
def A(X):
    """Application linéaire : A(X) = LM@X + X@LN"""
    return P@X + X@Q

# Calcul du gradient conjugué
X, cpt = ma.grad_conj(ma.ps, A, R, x0, nbiter, epsilon)

# Affichage du résultat et du nombre d'itérations
print(X, cpt)

#%% ==== Partie électrostatique ====


# Chargement du fichier contenant le résidu R 
R = np.load("data/R.npy")

# On pose R_sylv = -R car R_sylv = -1/eps * R
R_sylv = -R

# Récupération des dimensions de R
M, N = R.shape

# Construction des matrices Laplaciennes
LM = ma.matrice_laplacienne_1D(M)
LN = ma.matrice_laplacienne_1D(N)

# Création des conditions initiales/tolérance de convergence/nombre max d'itérations
x0 = np.ones((M,N))
epsilon = 1
nbiter = 500

# Définition de l'application linéaire utilisée
def A(X):
    """Application linéaire : A(X) = LM@X + X@LN"""
    return LM@X + X@LN

# L'équation est: LM @ V + V @ LN = R_sylv
V, cpt = ma.grad_conj(ma.ps, A, R_sylv, x0, nbiter, epsilon)

# Création des matrices de dérivation D
DM = ma.create_derivation_matrice(M)
DN = ma.create_derivation_matrice(N)

# Calcul des composantes du champ électrique : E = -∇V
Ex = DM @ V
Ey = V @ DN.T


#====AFFICHAGE=====
# Initialisation des axes
x = np.arange(N)
y = np.arange(M)
X,Y = np.meshgrid(x,y)

# Création de la figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Fond: Potentiel électrostatique V
im = ax.contourf(X, Y, V, levels=30, cmap='RdBu', alpha=0.7)

# # 3. Lignes de champ électrique
stream = ax.streamplot(X, Y, Ey, Ex,
                        color='black', density=1.2, 
                        linewidth=1.3, arrowsize=1.5)

# Définition des couleurs (de -1 à +1)
colors = ['blue', 'white', 'red'] 

# Créer la colormap
cmap_custom = ListedColormap(colors)

# Définir les limites de l'échelle de couleurs
# vmin = -1 (mappé à la première couleur : bleu)
# vmax = 1 (mappé à la dernière couleur : rouge)
vmin_val = -1
vmax_val = 1

# Affichage du dessin original
im2 = ax.imshow(R_sylv, cmap=cmap_custom, vmin=vmin_val, vmax=vmax_val)

# Titre et labels
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_aspect('equal')
plt.show()

#%% ==== Potentiel électrique avec tracé manuel ====

# Créer une image noire de 512x512 pixels avec 3 canaux de couleur (BGR)
img = np.zeros((512, 512, 3), np.uint8)

# Récupération des dimensions
M, N, P = img.shape

# Création de la matrice de charge à modifier en fonction du tracé
R = np.zeros((M, N), np.int8)

# Créer une fenêtre nommée "Curve Window"
cv2.namedWindow("Curve Window")

# Définition des paramètres à passer au callback
state = {
    "img": img,
    "drawing_p": False,
    "drawing_m": False,
    "R": R
}

# Connecter la fonction callback draw_curve aux événements souris de la fenêtre
cv2.setMouseCallback("Curve Window", ma.draw_curve, param=state)

# Boucle principale d'affichage
while True:
    # Afficher l'image dans la fenêtre
    cv2.imshow("Curve Window", img)
    
    # Attendre 10ms pour une touche
    # Si la touche ESC (code ASCII 27) est pressée, sortir de la boucle
    if cv2.waitKey(1) == 27:
        break

# Fermer toutes les fenêtres OpenCV
cv2.destroyAllWindows()

# Construction des matrices Laplaciennes
LM = ma.matrice_laplacienne_1D(M)
LN = ma.matrice_laplacienne_1D(N)

# Création des conditions initiales/tolérance de convergence/nombre max d'itérations
x0 = np.ones((M,N))
epsilon = 10**(-1)
nbiter = 500

def A(X):
    """Application linéaire : A(X) = LM@X + X@LN"""
    return LM@X + X@LN

# L'équation est: LM @ V + V @ LN = R
V, cpt = ma.grad_conj(ma.ps, A, R, x0, nbiter, epsilon)

# Création des matrices de dérivation pour calculer le gradient de V
DM = ma.create_derivation_matrice(M)
DN = ma.create_derivation_matrice(N)

# Caclul des composantes de champ électrique E = -∇V
Ex = DM @ V
Ey = V @ DN.T

#====AFFICHAGE=====
# Initialisation des axes
x = np.arange(N)
y = np.arange(M)
X,Y = np.meshgrid(x,y)

# Création de la figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Fond: Potentiel électrostatique V
im = ax.contourf(X, Y, V, levels=30, cmap='RdBu', alpha=0.5)

# 3. Lignes de champ électrique
stream = ax.streamplot(X, Y, Ey, Ex,
                        color='black', density=1.2, 
                        linewidth=1.3, arrowsize=1.5)

# Les couleurs sont définies dans l'ordre où elles seront mappées (de -1 à +1)
colors = ['blue', 'white', 'red'] 

# Créer la colormap
cmap_custom = ListedColormap(colors)

# Définir les limites de l'échelle de couleurs
# vmin = -1 (mappé à la première couleur : bleu)
# vmax = 1 (mappé à la dernière couleur : rouge)
vmin_val = -1
vmax_val = 1

# Affichage du dessin original
im = ax.imshow(R, cmap=cmap_custom, vmin=vmin_val, vmax=vmax_val)

# Titre et labels
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_aspect('equal')
plt.show()


#%% === Application draw_mask ===
Masque = ma.draw_mask("data/graffiti.jpg", 5).astype("float32")

#Cette etape nous permet de faire ressortir visuellement les pixels du masque (qui ont la valeur 1) en les mettant en blanc
(i,j) = np.where(Masque == 1)
Masque[i,j] = 255
#Enregistre le masque dans un fichier puis l'affiche
cv2.imwrite("masque.jpg", Masque)
cv2.imshow("Masque", Masque.astype("uint8"))

#%% === Application image_restauration_nb à graffiti.jpg ===

# On transforme l ’ image originale en noir et blanc afin de la comparer au resultat et on l ’ enregistre
Img_nb = cv2.cvtColor(cv2.imread("data/graffiti.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite("data/graffiti_nb.jpg",Img_nb)
# On applique la fonction de restauration en noir et blanc sur l ’ image
Restauration_nb = ma.image_restauration_nb("data/graffiti.jpg")


#%% === Application image_retauration_couleur à graffiti.jpg ===

Restauration_couleur = ma.image_restauration_couleur("data/graffiti.jpg")

#%% === Application image_restauration_couleur_opti à graffiti.jpg ===

Restauration_couleur = ma.image_restauration_couleur_opti("data/graffiti.jpg")

#%% === Application image_restauration_couleur_opti à PAF.png ===

Escadrille_modifiee = ma.image_restauration_couleur_opti("data/PAF.png", None, 1, 500)

#%% === Restauration d'image ancienne et mise en avant des limites avec une bonne précision pour le gradient ===

Restauration_vielle_image = ma.image_restauration_couleur_opti("data/orig1.jpg", None, 1e-8, 5000)

#%% === Floutage visage, pas besoin de beaucoup de precision ===

Floutage = ma.image_restauration_couleur_opti("data/orig1.jpg", None, 1, 5000)

#%% === Transposition d'image ===

m = ma.transfert_zone_couleur("data/piscine.jpg", 'data/ours.jpg', 1, 3000, 1e-3)

#%% === Transposition d'image par gradient dominant ===

m = ma.transfert_zone_couleur_mixed("data/mur.png", "data/grad_F.png", 3000, 1e-3)

#%% === Limite : Transposition d'image par gradient dominant ===

m = ma.transfert_zone_couleur_mixed("data/mur.png", "data/ipsa.png", 3000, 1e-3)
