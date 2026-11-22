# PIOT Elyn
# RAMOS Eva
# ROUMIGUIER Etienne
# SCORCIONE Noa
# VERLEYEN Lucas 
# ZERHOUNI Adam

import numpy as np
import cv2

def grad(X, A, b):
    """Gradient: A(X) - b où A(X) = PX + XQ"""
    return A(X) - b

def ps(u, v):
    """Produit scalaire de Frobenius"""
    return np.trace(u.T @ v)


def grad_conj(ps, A, b, X0, nbiter, epsilon):
    """
    Résout A(X) = b par gradient conjugué.
    Paramètres:
    - ps : fonction produit scalaire ps(U,V) = trace(U^T @ V)
    - A : application linéaire à définir
    - b : second membre (résidu)
    - X0 : solution initiale
    - nbiter : int, nombre max d'itérations
    - epsilon : float, tolérance de convergence
    Return:
        X : solution approchée
        cpt : nombre d'itérations effectuées
    """
    X = X0
    d = - grad(X, A, b)
    cpt=0
    while np.linalg.norm(d) > epsilon and cpt < nbiter :
        t = - ps(d, grad(X, A, b)) / ps(d, A(d))
        X = X + t*d
        beta = ps(d, A(grad(X, A, b))) / ps(d, A(d))
        d = - (grad(X, A, b)) + beta*d
        cpt += 1
    return X, cpt

def matrice_laplacienne_1D(n):
    """
    Crée la matrice de dérivation seconde de taille n x n
    """
    L = -2 * np.eye(n)  # Diagonale principale
    L += np.diag(np.ones(n-1), k=1)   # k représente le décalage par rapport à la diagonale principale.
    L += np.diag(np.ones(n-1), k=-1)  # Sur-diagonale inférieure
    return L

def create_derivation_matrice(n):
    """
    Crée la matrice de dérivation de taille n x n
    """
    D = np.zeros((n,n))  # Matrice nulle
    D += np.diag(0.5*np.ones(n-1), k=1)   # k représente le décalage par rapport à la diagonale principale.
    D += np.diag(-0.5*np.ones(n-1), k=-1)
    return D

        
def draw_curve(event, x, y, flags, param):
    """
    Fonction callback appelée à chaque événement de souris.
    
    Paramètres:
    - event: Type d'événement souris (clic, mouvement, relâchement)
    - x, y: Coordonnées actuelles du curseur
    - flags: Flags additionnels (non utilisés ici)
    - param: Paramètres additionnels
    """
    
    # L'image original est passée en paramètre pour être modifier
    img = param["img"]
    
    # Le résidu est passé en paramètre de la fonction
    R = param["R"]
    
    # Gestion du bouton GAUCHE de la souris (dessin rouge)
    if event == cv2.EVENT_LBUTTONDOWN:
        # Début du dessin lorsque le bouton gauche est pressé
        param["drawing_p"] = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        # Si on est en train de dessiner avec le bouton gauche
        if param["drawing_p"] == True:
            # Dessiner un cercle rouge (BGR: 0, 0, 255) de rayon 3 et épaisseur 4
            cv2.circle(img, (x, y), 3, (0, 0, 255), 4)
            cv2.circle(R, (x, y), 3, 1, 4)   # Modification charge +1
            
    elif event == cv2.EVENT_LBUTTONUP:
        # Arrêter le dessin quand le bouton gauche est relâché
        param["drawing_p"] = False
    
    # Gestion du bouton DROIT de la souris (dessin bleu)
    if event == cv2.EVENT_RBUTTONDOWN:
        # Début du dessin lorsque le bouton droit est pressé
        param["drawing_m"] = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        # Si on est en train de dessiner avec le bouton droit
        if param["drawing_m"] == True:
            # Dessiner un cercle bleu (BGR: 255, 0, 0) de rayon 3 et épaisseur 4
            cv2.circle(img, (x, y), 3, (255, 0, 0), 4)
            cv2.circle(R, (x, y), 3, -1, 4)   # Modification charge -1
            
    elif event == cv2.EVENT_RBUTTONUP:
        # Arrêter le dessin quand le bouton droit est relâché
        param["drawing_m"] = False


def draw_mask(chemin_image, epaisseur_trait = 5):
    """
    Fonction draw_mask permettant de tracer le masque sur une image pour supprimer ou restaurer des éléments.
    
    Paramètres:
    - chemin_image: chemin d'accès de l'image sur laquelle on veut tracer le masque
    - epaisseur_trait : (facultatif) permet de choisir la largeur du tracé
    """
    #Il est possible , grace a l'argument epaisseur_trait (facultatif), de choisir l'epaisseur du trait pour l'application du masque
    
    #Variable pour le dessin
    clic = False
    
    #Charger l'image et l'avoir en matrice
    img = cv2.imread(chemin_image)
    #Image sur laquelle on va tracer le masque
    img_et_mask = img.copy()
    
    #Création du masque
    m = img.shape[0]
    n = img.shape[1]
    
    mask = np.zeros((m,n))
    mask = mask.astype("uint8")
    
    #Fonction pour faire le lien avec les commandes de la souris
    def souris_callback(event, x, y, flags, param):
        nonlocal clic, img_et_mask, mask
        #Si on realise un clic gauche alors on active le dessin
        if event == cv2.EVENT_LBUTTONDOWN:
            clic = True
        #Si en plus on se deplace alors on marque l'image avec le masque et on met a jour le masque
        elif event == cv2.EVENT_MOUSEMOVE:
            if clic:
                #Dessiner sur l'image d'affichage un cercle vert plein de rayon 4
                cv2.circle(img_et_mask, (x, y), epaisseur_trait, (0, 255, 0), -1)
                # On met les coordonnees correspondantes a 1 dans le masque
                cv2.circle(mask, (x, y), epaisseur_trait, 1, -1)
        #Si on arrete de cliquer alors le dessin doit se stopper
        elif event == cv2.EVENT_LBUTTONUP:
            # Fin du dessin
            clic = False
    
    #Creation de la fenetre qui va s'ouvrir pour le dessin du masque
    fenetre = "Tracez le masque"
    cv2.namedWindow(fenetre)
    #On associe la fonction callback pour la commande du trace par la souris a la fenetre
    cv2.setMouseCallback(fenetre, souris_callback)
    
    #Boucle infini qui va faire tourner le code continuellement jusqu'a ce que la touche d'arret soit saisit
    while True:
        #On affiche le trace au fur et a mesure de sa creation
        cv2.imshow(fenetre, img_et_mask)
        #Delai necessaire afin de permettre l'actualisation du trace
        key = cv2.waitKey(1)
        if key == 13 :
            break
    
    cv2.destroyAllWindows()
    return mask


def image_restauration_nb(chemin, masque = None):
    """
    Fonction permettant de retirer des éléments d'une image en les sélectionnant ou grâce à un masque prédéfini. Le résultat est en noir et blanc.
    
    Paramètres:
    - chemin : chemin d'accès de l'image à modifier
    - masque : (facultatif) permet de travailler avec un masque prédéfini
    """

    if masque is None:
        M = draw_mask(chemin).astype("float32")
    else :
        M = masque
        
    img_org = cv2.imread(chemin).astype("float32")
    F = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY).astype("float32")
    
    # Créer ones avec la même forme que le masque
    ones = np.ones_like(M)
    
    (m,n) = np.shape(ones)
    
    Lm = matrice_laplacienne_1D(m)
    Ln = matrice_laplacienne_1D(n)
    K = ones - M
    
    R = -((Lm @(F*K) + (F*(K))@Ln))*M
    
    def A(X):
        #Application linéaire : A(X) = (PX + XQ)*M
        return (Lm@X + X@Ln) * M
    
    X0 = np.zeros((m,n))
    
    #Application du gradient conjugué
    X, cpt = grad_conj(ps, A, R, X0, 100, 1)
    
    #Cette ligne nous permet de supprimer les valeur aberrantes, c'est a dire celles inferieures a 0 ou alors superieure a 255. Il est en effet possible que le gradient conjugue produise de telles valeurs, d'où l'interet de l'appliquer juste apres. 
    X = np.maximum(0, np.minimum(X, 255))
    
    Final = X + F*K
    # Sauvegarder l'image
    cv2.imwrite("./result_restauration_nb.jpg", Final.astype("uint8"))
    
    return Final.astype("uint8")



#Meme programme pour un resultat en couleur
def image_restauration_couleur(chemin, masque = None):
    """
    Fonction permettant de retirer des éléments d'une image en les sélectionnant ou grâce à un masque prédéfini. Le résultat est en couleur.
    
    Paramètres:
    - chemin : chemin d'accès de l'image à modifier
    - masque : (facultatif) permet de travailler avec un masque prédéfini
    """

    if masque is None:
        M = draw_mask(chemin).astype("float32")
    else :
        M = masque
    img_org = cv2.imread(chemin).astype("float32")
    
    # Créer ones avec la même forme que le masque
    ones = np.ones_like(M)
    
    (m, n) = np.shape(ones)
    
    Lm = matrice_laplacienne_1D(m)
    Ln = matrice_laplacienne_1D(n)
    K = ones - M
    
    # Traiter chaque canal de couleur séparément
    Final = np.zeros_like(img_org)
    
    #On traite couleur par couleur
    
    for canal in range(3):  
        #On sélectionne avec F chaque canal de couleur que l'on va modifier
        #img_org est composé de trois éléments [m,n,canaux] : m nb de ligne, n nb colonnes, canaux = 0,1,2 pour Blue Green Red
        F = img_org[:, :, canal]
        #Même opération qu'en noir et blanc
        R = -((Lm @ (F * K) + (F * K) @ Ln)) * M
        
        def A(X):
            """Application linéaire : A(X) = (Lm@X + X@Ln)*M"""
            return (Lm @ X + X @ Ln) * M
        
        X0 = np.zeros((m,n))
        #Application du gradient conjugué
        X, cpt = grad_conj(ps, A, R, X0, 100, 1)
        
        #Cette ligne nous permet de supprimer les valeur aberrantes, c'est a dire celles inferieures a 0 ou alors superieure a 255. Il est en effet possible que le gradient conjugue produise de telles valeurs, d'où l'interet de l'appliquer juste apres.
        X = np.maximum(0, np.minimum(X, 255))
        
        Final[:, :, canal] = X + F * K
    
    # Sauvegarder l'image
    cv2.imwrite("./result_restauration_couleur.jpg", Final.astype("uint8"))
    
    return Final.astype("uint8")

#Programme utile pour l'optmisation et donc pour le code d'apres
def relimiter_masque(M):
    """
    Détermine les coordonnées limites d'un masque

    paramètre :
        M : matrice masque (valeurs 0 ou 1)
    
    return :
        bas, haut, gauche, droite :
            - bas : indice de la ligne la plus basse où M=1
            - haut : indice de la ligne la plus haute où M=1
            - gauche : indice de la colonne la plus à gauche où M=1
            - droite : indice de la colonne la plus à droite où M=1
    """

    # Récupère les indices (ligne, colonne) où le masque vaut 1
    un = np.where(M == 1)
    # un[0] contient les indices de lignes
    # un[1] contient les indices de colonnes

    # Ligne la plus haute (minimum avec OpenCV) et la plus basse (maximum)
    haut = np.max(un[0])   # ligne la plus basse où M=1
    bas = np.min(un[0])   # ligne la plus haute où M=1

    # Colonne la plus à gauche (min) et la plus à droite (max)
    gauche = np.min(un[1])
    droite  = np.max(un[1])

    # Renvoie les limites définie
    return bas, haut, gauche, droite

def image_restauration_couleur_opti(chemin, masque = None, epsilon = 1, nb_iterations = 100):
    """
    Fonction permettant de retirer des éléments d'une image en les sélectionnant ou grâce à un masque prédéfini. Le résultat est en couleur et le temps de calcul réduit.
    
    Paramètres:
    - chemin : chemin d'accès de l'image à modifier
    - masque : (facultatif) permet de travailler avec un masque prédéfini
    - epsilon : float, tolérance de convergence
    - nb_iterations : int, nombre max d'itérations
    """
    if masque is None:
        M = draw_mask(chemin).astype("float32")
    else :
        M = masque

    img_org = cv2.imread(chemin).astype("float32")
    m,n = np.shape(M)
    
    # Déterminer les limites du masque
    limites = relimiter_masque(M)
    
    bas, haut, gauche, droite = limites
    
    #On rajoute des pixels de marge pour avoir de bonnes valeurs pour le gradient
    marge = 3
    bas_m = max(0, bas - marge)
    haut_m = min(M.shape[0], haut + marge + 1)
    gauche_m = max(0, gauche - marge)
    droite_m = min(M.shape[1], droite + marge + 1)
    
    #On relimite les matrices aux dimensions optimales
    M_relim = M[bas_m:haut_m, gauche_m:droite_m]
    img_relim = img_org[bas_m:haut_m, gauche_m:droite_m, :]
    
    # La matrice de un avec la bonne taille relimitee
    ones_relim = np.ones_like(M_relim)
    
    (m_relim, n_relim) = np.shape(ones_relim)
    
    # Matrices laplaciennes relimitees
    Lm_relim = matrice_laplacienne_1D(m_relim)
    Ln_relim = matrice_laplacienne_1D(n_relim)
    K_relim = ones_relim - M_relim
    
    # Traiter chaque canal de couleur séparément
    Final_relim = np.zeros_like(img_relim)
    
    for canal in range(3):  
        F_relim = img_relim[:, :, canal]
        
        # Même opération qu'avant mais avec les dimensions optimales pour que ce soit moins lourd en calcul
        R_relim = -((Lm_relim @ (F_relim * K_relim) + (F_relim * K_relim) @ Ln_relim)) * M_relim
        
        def A_relim(X):
            """Application linéaire : A(X) = (Lm@X + X@Ln)*M"""
            return (Lm_relim @ X + X @ Ln_relim) * M_relim
        
        X0_relim = np.zeros((m_relim, n_relim))
        
        #Application du gradient conjugué
        X_relim, cpt = grad_conj(ps, A_relim, R_relim, X0_relim, nb_iterations, epsilon)
        
        #Cette ligne nous permet de supprimer les valeur aberrantes, c'est a dire celles inferieures a 0 ou alors superieure a 255. Il est en effet possible que le gradient conjugue produise de telles valeurs, d'où l'interet de l'appliquer juste apres.
        X_relim = np.maximum(0, np.minimum(X_relim, 255))
        
        Final_relim[:, :, canal] = X_relim + F_relim * K_relim
    
    # Reconstruire l'image complète
    Final = img_org.copy()
    #On reforme d'abord la partie redimensionnee finale
    Final[bas_m:haut_m, gauche_m:droite_m, :] = Final_relim
    
    # Sauvegarder l'image
    cv2.imwrite("./result_restauration_couleur_opti.jpg", Final.astype("uint8")) 
    return Final.astype("uint8")


def crop_image(chemin_image):
    """
    Ouvre une image et permet de tracer un rectangle avec la souris
    pour recadrer une zone. 

    paramètre :
        chemin_image : chemin de l'image à charger

    return :
        crop_result : image recadrée
    """

    # Variables utilisées pour gérer le rectangle et la souris
    drawing = False       # True tant que l'utilisateur maintient le clic
    ix, iy = -1, -1       # coordonnées du point de départ du rectangle
    rx1 = ry1 = rx2 = ry2 = 0  # coordonnées finales du rectangle sélectionné

    # Chargement de l'image 
    img = cv2.imread(chemin_image)

    # Copie de l'image pour dessiner dessus sans modifier l’original
    img_copy = img.copy()

    # Fonction callback souris
    def draw_rectangle(event, x, y, flags, param):
        """
        Fonction appelée automatiquement par OpenCV à chaque événement souris.
        """
        nonlocal drawing, ix, iy, img_copy, rx1, ry1, rx2, ry2

        # Clic gauche pressé : début du rectangle 
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y  # on mémorise le point de départ

        # Dessin du rectangle en cours
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # On réinitialise l'image temporaire puis on redessine le rectangle
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

        # Clic gauche relâché : fin du rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rx1, ry1, rx2, ry2 = ix, iy, x, y  # on enregistre la zone
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    # Boucle d'affichage
    cv2.namedWindow("Image")
    # on associe le callback
    cv2.setMouseCallback("Image", draw_rectangle)
    
    # Initialisation de l'image recadrée
    crop_result = None  
    while True:
        # Affiche l'image
        cv2.imshow("Image", img_copy)

        # Lecture des touches
        key = cv2.waitKey(1)

        # Appuyer sur ENTER pour recadrer
        if key == 13:
            # Les coordonnées doivent être triées pour que le rectangle 
            # puisse être dessiner dans n'importe quel sens)
            x1, x2 = sorted([rx1, rx2])
            y1, y2 = sorted([ry1, ry2])
            
            # Récupération de la partie de l'image sélectionnée
            crop_result = img[y1:y2, x1:x2]
            break

    # Nettoyage des fenêtres OpenCV
    cv2.destroyAllWindows()

    # On renvoie l'image recadrée 
    return crop_result


def transfert_zone_couleur(img_orig, img_modif, alpha, iterations, nb_iterations,epaisseur_trait = 5):
    """
    Fonction permettant de retirer des éléments d'une image en les sélectionnant puis d'effectuer le remplissage avec une partie d'une autre image.
    
    Paramètres:
    - img_orig : chemin d'accès de l'image sur laquelle on veut retirer puis remplacer un éléments
    - img_modif : chemin d'accès de l'image où on va chercher l'élément de remplissage
    - alpha : transparence du remplacement entre 0 et 1
    - iterations : int, nombre max d'itérations
    - nb_iterations : int, nombre max d'itérations
    - epaisseur_trait : (facultatif) permet de choisir la largeur du tracé

    return :
        Final : image modifiée
    """

    # Tracé et récupération de la matrice masque
    M = draw_mask_poly(img_orig).astype("float32")
    img_org = cv2.imread(img_orig).astype("float32")
    
    # Rélimitation de l'image à transposer
    G = crop_image(img_modif)
    m, n = np.shape(M)
    
    # Déterminer les limites du masque
    bas, haut, gauche, droite = relimiter_masque(M)

    # Marge de pixels pour le gradient
    marge = 6
    bas_m = max(0, bas - marge)
    haut_m = min(M.shape[0], haut + marge + 1)
    gauche_m = max(0, gauche - marge)
    droite_m = min(M.shape[1], droite + marge + 1)

    # Relimitation des matrices à la taille du masque
    M_relim = M[bas_m:haut_m, gauche_m:droite_m]
    img_relim = img_org[bas_m:haut_m, gauche_m:droite_m, :]
    ones_relim = np.ones_like(M_relim)
    m_relim, n_relim = np.shape(ones_relim)

    # Laplaciennes
    Lm_relim = matrice_laplacienne_1D(m_relim)
    Ln_relim = matrice_laplacienne_1D(n_relim)
    
    # Différence Im,n - M, utilisée pour le calcul du résidu matriciel
    K_relim = ones_relim - M_relim
    Final_relim = np.zeros_like(img_relim)

    # Redimensionner G pour correspondre à la zone relimitée
    hauteur = haut_m - bas_m
    largeur = droite_m - gauche_m
    G_relim = cv2.resize(G, (largeur, hauteur))

    # On traite couleur par couleur
    for canal in range(3):
        # On sélectionne avec F chaque canal de couleur que l'on va modifier
        # img_org est composé de trois éléments [m,n,canaux] : m nb de ligne, n nb colonnes, canaux = 0,1,2 pour Blue Green Red
        F_relim = img_relim[:, :, canal]

        # Définition de l'application linéaire A
        def A_relim(X):
            return (Lm_relim @ X + X @ Ln_relim) * M_relim
            
        # Calcul de la matrice résiduelle
        R_relim = -A_relim(F_relim * K_relim) + alpha*A_relim(G_relim[:, :, canal]) + (1-alpha) * A_relim(F_relim)

        # Définition des paramètres pour appliquer le gradient conjugué
        X0_relim = np.ones((m_relim, n_relim))
        iterations = int((m*n)/(m_relim*n_relim) * 1000)
        epsilon = 1 * (m_relim * n_relim) / (m * n)

        # Calcul du gradient conjugué
        X_relim, cpt = grad_conj(ps, A_relim, R_relim, X0_relim, iterations, epsilon)
        
        # Nnormalisation des valeurs entre 0 et 255
        X = np.maximum(0, np.minimum(X_relim, 255))
        
        # Récupération de l'image complète relimitée
        Final_relim[:, :, canal] = X + F_relim * K_relim

    # Reconstruction de l'image complète
    Final = img_org.copy()
    Final[bas_m:haut_m, gauche_m:droite_m, :] = Final_relim
    
    # Enregistrement du résultat dans result.jpg
    cv2.imwrite("./result_transfert_zone_couleur.jpg", Final.astype("uint8"))
    return Final

def draw_mask_poly(chemin_image):
    """
    Fonction draw_mask_poly permettant de tracer le masque sur une image pour supprimer ou restaurer des éléments. Le tracé se fait en plaçant des points qui génèrent un polygone.
    
    Paramètres:
    - chemin_image: chemin d'accès de l'image sur laquelle on veut tracer le masque
    """
    #Variable pour le dessin
    clic = False
    
    #Charger l'image et l'avoir en matrice
    img = cv2.imread(chemin_image)
    #Image sur laquelle on va tracer le masque
    img_et_mask = img.copy()
    
    #Création du masque
    m, n, p = np.shape(img)
    mask = np.zeros((m,n))
    mask = mask.astype("uint8")
    
    # Création de la liste pour contenir les coordonnées des points sélectionnés
    points = []
    
    #Fonction pour faire le lien avec les commandes de la souris
    def souris_callback(event, x, y, flags, param):
        nonlocal clic, points, img_et_mask
        #Si on realise un clic gauche alors on trace un croix
        if event == cv2.EVENT_LBUTTONDOWN:
            clic = True
            # Dessine un croix rouge d'épaisseur 2, à l'endroit du clic
            cv2.line(img_et_mask, (x - 5, y), (x + 5, y), (0, 0, 255), 2)  # ligne horizontale
            cv2.line(img_et_mask, (x, y - 5), (x, y + 5), (0, 0, 255), 2)  # ligne verticale
            
            # On enregistre les coordonnées correspondantes dans la liste
            points.append((x, y))
    
    #Creation de la fenetre qui va s'ouvrir pour le dessin du masque
    fenetre = "Tracez le masque"
    cv2.namedWindow(fenetre)
    #On associe la fonction callback pour la commande du trace par la souris a la fenetre
    cv2.setMouseCallback(fenetre, souris_callback)
    
    #Boucle infini qui va faire tourner le code continuellement jusqu'a ce que la touche d'arret soit saisit
    while True:
        #On affiche le trace des croix au fur et a mesure de sa creation
        cv2.imshow(fenetre, img_et_mask)
        #Delai necessaire afin de permettre l'actualisation du trace
        key = cv2.waitKey(1)
        # touche ENTER cliqué
        if key == 13 :
            if len(points) >= 3:
                # Conversion de la liste de points en tableau numPy int32
                poly = np.array(points, dtype=np.int32)
                # Trace le polygone rouge avec les points enregistrés
                cv2.fillPoly(img_et_mask, [poly], (0, 0, 255))
                # Affiche le polygone sur la fenêtre
                cv2.imshow(fenetre, img_et_mask)
                # Place des 1 sur le masque à l'endroit du polygone
                cv2.fillPoly(mask, [poly], 1)
        
        # touche ENTER cliqué
        if key == 8 :
            points = points[:-1]
                
            # Re-dessine l'image pour effacer l'ancienne croix
            img_et_mask = img.copy()
            
            # 3. Re-trace toutes les croix restantes
            for x, y in points:
                # Dessine un croix rouge d'épaisseur 2, aux coordonnées du point
                cv2.line(img_et_mask, (x - 5, y), (x + 5, y), (0, 0, 255), 2)  # ligne horizontale
                cv2.line(img_et_mask, (x, y - 5), (x, y + 5), (0, 0, 255), 2)  # ligne verticale
            
        # Touche ESPACE cliqué : fin de la boucle
        if key == 32:
            break
        
    cv2.destroyAllWindows()
    return mask


def transfert_zone_couleur_mixed(img_orig, img_modif, iterations, epsilon, epaisseur_trait = 5):
    """
    Fonction permettant de retirer des éléments d'une image en les sélectionnant puis d'effectuer le remplissage avec une partie d'une autre image avec la méthode du gradient dominant.
    
    Paramètres:
    - img_orig : chemin d'accès de l'image sur laquelle on veut retirer puis remplacer un éléments
    - img_modif : chemin d'accès de l'image où on va chercher l'élément de remplissage
    - iterations : int, nombre max d'itérations
    - epsilon : float, tolérance de convergence
    - epaisseur_trait : (facultatif) permet de choisir la largeur du tracé

    return :
        Final : image modifiée
    """
    # Tracé et récupération de la matrice masque
    M = draw_mask_poly(img_orig).astype("float32")
    img_org = cv2.imread(img_orig).astype("float32")
    
    # Rélimitation de l'image à transposer
    G = crop_image(img_modif).astype("float32")
    m, n = np.shape(M)
    
    # Déterminer les limites du masque
    bas, haut, gauche, droite = relimiter_masque(M)

    # Marge de pixels pour le gradient
    marge = 6
    bas_m = max(0, bas - marge)
    haut_m = min(M.shape[0], haut + marge + 1)
    gauche_m = max(0, gauche - marge)
    droite_m = min(M.shape[1], droite + marge + 1)

    # Relimitation des matrices à la taille du masque
    M_relim = M[bas_m:haut_m, gauche_m:droite_m]
    img_relim = img_org[bas_m:haut_m, gauche_m:droite_m, :]
    ones_relim = np.ones_like(M_relim)
    m_relim, n_relim = np.shape(ones_relim)

    # Laplaciennes
    Lm_relim = matrice_laplacienne_1D(m_relim)
    Ln_relim = matrice_laplacienne_1D(n_relim)
    
    # Différence Im,n - M, utilisée pour le calcul du résidu matriciel
    K_relim = ones_relim - M_relim
    Final_relim = np.zeros_like(img_relim)

    # Redimensionner G pour correspondre à la zone relimitée
    hauteur = haut_m - bas_m
    largeur = droite_m - gauche_m
    G_relim = cv2.resize(G, (largeur, hauteur), interpolation=cv2.INTER_LANCZOS4)
    
    # Matrice dérivée
    Dm_relim = create_derivation_matrice(m_relim)
    Dn_relim = create_derivation_matrice(n_relim)
    

    # On traite couleur par couleur
    for canal in range(3):
        # On sélectionne avec F chaque canal de couleur que l'on va modifier
        # img_org est composé de trois éléments [m,n,canaux] : m nb de ligne, n nb colonnes, canaux = 0,1,2 pour Blue Green Red
        F_relim = img_relim[:, :, canal]
        G2_relim = G_relim[:, :, canal]

        # Calcil des gradients des images
        Fx_relim = (-Dm_relim) @ F_relim
        Fy_relim = F_relim @ (-Dn_relim.T)
        
        Gx_relim = (-Dm_relim) @ G2_relim
        Gy_relim = G2_relim @ (-Dn_relim.T)
        
        # Calcul des normes
        normF = np.sqrt(Fx_relim**2 + Fy_relim**2)
        normG = np.sqrt(Gx_relim**2 + Gy_relim**2)
        
        # Calcul de V en mixant les gradients
        # Si la condition est vérifiée, alors on remplace par les coefficients de G
        # Sinon on remplace par les coefficients de F
        V = np.where(normG > normF, G2_relim, F_relim)
        
        # Définition de l'application linéaire A
        def A_relim(X):
            return (Lm_relim @ X + X @ Ln_relim) * M_relim
            
        # Calcul de la matrice résiduelle
        R_relim = -A_relim(F_relim * K_relim) + A_relim(V)

        # Définition des paramètres pour appliquer le gradient conjugué
        X0_relim = np.ones((m_relim, n_relim))

        # Calcul du gradient conjugué
        X_relim, cpt = grad_conj(ps, A_relim, R_relim, X0_relim, iterations, epsilon)
        
        # Nnormalisation des valeurs entre 0 et 255
        X = np.maximum(0, np.minimum(X_relim, 255))
        
        # Récupération de l'image complète relimitée
        Final_relim[:, :, canal] = X + F_relim * K_relim

    # Reconstruction de l'image complète
    Final = img_org.copy()
    Final[bas_m:haut_m, gauche_m:droite_m, :] = Final_relim
    
    # Enregistrement du résultat dans result.jpg
    cv2.imwrite("./result_transfert_zone_couleur_mixed.jpg", Final.astype("uint8"))
    return Final

