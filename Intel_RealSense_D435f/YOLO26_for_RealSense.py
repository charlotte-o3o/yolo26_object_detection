import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import random
import time
import os


""" ----------------------------------------------------------------------------------------------- """
""" ----------- YOLO26 for Intel RealSense with Movement Detection and Object Detection ----------- """
""" ----------------------------------------------------------------------------------------------- """


###################################################################
#                         CONFIGURATION                           # 
###################################################################


# --- Fonts config
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

# --- Configuration de la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
fps_cam = 30

# --- Stream couleur
# --- format.bgr8 => Format d'image classique : BGR = RGB inversé ; 8 bits par canal de couleur (B, G, R) => 24
# --- Chaque pixel est codé sur 24 bits
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps_cam)

# --- Stream profondeur = carte qui mesure les distances
# --- format.z16 => 16 bits par pixel (unsigned int 16) + de précision que seulement 8 bits
# --- Valeur de l'int = dist. en mm
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps_cam)

print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !")
print("Appuier sur la touche ECHAP dans la fenêtre vidéo pour quitter.")

# --- Config de la détection de mouvement - Soustraction de fond avec OpenCV MOG2
# --- Détection demouvement par changement de couleur des pixels
"""
    - history : algorithm memory in number of frames ; at 30 fps => x frames represent x/30 sec
    - varThreshold : tolerance threshold to decide whether a pixel has moved or not ; the lower the more sensitive
    - detectShadows : the algorithm will try to guess whether a change of color is due to shadows ; /!\ quite heavy computation
"""
history = 15
varThreshold = 12
hist_time = history/fps_cam
backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=False)

# --- Alignement du flux de profondeur sur le flux couleur
# --- Le flux couleur sert de base à l'alignement
align = rs.align(rs.stream.color)

print("Lancement de l'enregistrement vidéo...")

# --- Config enregistrement vidéo
video_directory = "captures_realsense"

if not os.path.exists(video_directory):
    os.makedirs(video_directory)
    print(f"Dossier '{video_directory}' créé avec succès !")
    
timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
video_file = f"capture_{timestamp}.avi"
video_path = os.path.join(video_directory, video_file)

# --- Init du VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps_enregistrement = 18.0 # Baisser si la vidéo est en accéléré 
video_writer = cv2.VideoWriter(video_path, fourcc, fps_enregistrement, (1280, 720))

print("Enregistrement vidéo démarré !")
print(f"La vidéo sera enregistrée dans : {video_path}")

# --- Chargement du modèle YOLO26 nano
#model = YOLO('yolo26n.pt')
model = YOLO('best.pt')

# --- Assignation d'une couleur pour toutes les classes du dataset
couleurs_classes = {}
for class_id in model.names.keys():
    # --- Génère 3 entiers aléatoires entre 0 et 255
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    couleurs_classes[class_id] = (b, g, r)

try:
    while True:

        ###################################################################
        #                    RECUPERATION DE L'IMAGE                      # 
        ###################################################################

        # --- Récupération des frames brutes de la caméra (non alignées)
        frames = pipeline.wait_for_frames()

        # --- Alignement des frames => alginement des frames depth sur frames color
        aligned_frames = align.process(frames)

        # --- Récupération des flux une fois superposés
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Alerte : Pas de flux vidéo couleur et/ou profondeur reçu !")
            continue

        # --- Convertir en tableaux numpy car YOLO ne comprend pas le language natif de la caméra
        img = np.asanyarray(color_frame.get_data())


        ###################################################################
        #                     DETECTION DE MOUVEMENT                      #
        ###################################################################


        # --- Calcul du masque de mouvement
        # --- Application d'un flou gaussien (lissage des pixels avec leurs voisins directs) => lissage
        # --- Evite que le bruit numérique de la cam soit vu comme du mouvement
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # --- Calcul du mouvement (soustraction de fond) de l'image floutée
        # --- Comparation de l'image avec image mémorisée de la pièce immobile
        fg_mask = backSub.apply(blur)
        
        # --- Nettoyage du masque (enlève les petits pixels isolés détectés)
        # --- Création du kernel : matrice carrée de taille 5 remplie de 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # --- Ouverture : érosion des formes blanches (détectées comme en mouvement) puis dilatation de celles restantes
        # --- Supprime les parasites restants
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # --- Vérification de la présence ou non de mouvement dans l'image
        # --- Crée un tableau de bool : pixel blanc en mouvement = True ; pixel noir immobile = False
        # --- Calcul du nombre total de True (1) dans le tableau = nombre de pixels blancs du masque
        total_motion_pixels = np.sum(fg_mask == 255)

        if total_motion_pixels < 500: # Seuil global
            # --- Aucun mouvement détecté
            # --- On n'applique pas YOLO => inference = 0.0ms
            inference_time = 0.0
            results = []
        else:

        ###################################################################
        #                     DETECTION DES OBJETS                        #
        #                     SI MOUVEMENT DETECTE                        #
        ###################################################################

            # --- Mouvement(s) détecté(s)
            """
                Filtrage de classes : 
                Passer dans la méthode model() un paramètre classes=[x, y, z, ...] ; x, y et z sont les id associés aux classes dans names
                Afin de ne tester les objets détectés que sur ces classes
            """
            results = model(img, stream=True, verbose=False)
        
        """ 
            Méthode model() : détection des objets avec YOLO26
            Renvoie une liste d'objets Results ; ne contient toujours qu'un seul objet
            Objet Result = image analysée
            Attributs principaux :
                - orig_img : input image as a numpy array
                - boxes : detected bounding boxes (Boxes object)
                - masks : segmentation masks (Masks object)
                - probs : classification probabilities (Probs object)
                - obb : oriented bounding boxes (OBB object)
                - speed: dictionary containing preprocess, inference, and postprocess times
                - names: dictionary mapping class indices to class names
        """

        ###################################################################
        #                         TRAITEMENT DES                          #
        #                         BOUNDING BOXES                          #
        ###################################################################
        
        for r in results:
            
            # --- Récupération de bounding boxes détectées (objets Boxes)
            boxes = r.boxes

            # --- Récupération du temps d'inférence en ms
            if 'inference' in r.speed:
                inference_time = r.speed['inference']

            for box in boxes:
                
                # --- Récupération de l'ID de la classe
                class_id = int(box.cls[0])
                couleur = couleurs_classes[class_id]

                # --- Coordonnées du centre de la boîte
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # --- Filtrage des faux positifs
                # --- Extrait la ROI => la bounding box de l'objetc détecté
                roi_motion = fg_mask[y1:y2, x1:x2]
                
                # --- Si le nombre de pixels en mouvement < fp_threshold, on ignore l'objet
                fp_threshold = 150
                if np.sum(roi_motion == 255) < fp_threshold:
                    continue

                # --- Centre de la boîte pour calculer la distance
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                
                # --- Distance en m (grâce à l'alignement des frames)
                distance = depth_frame.get_distance(x_center, y_center)

                # --- Nommage de la classe de l'objet détecté grâce au mapping de names
                label = model.names[int(class_id)]

                # --- Probabilité de classe (confiance)
                confiance = float(box.conf[0]) * 100

                # --- Dessin sur l'image RGB des BB, nom de classe, centre des BB, distance...
                text = f"{label} ({confiance:.1f}%) : {distance:.2f}m"

                cv2.rectangle(img, (x1, y1), (x2, y2), couleur, 2)
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
                cv2.circle(img, (x_center, y_center), 4, (0, 0, 255), -1)

                print(f"Object: {model.names[int(box.cls[0])]} | Distance: {distance:.2f}m")


        ###################################################################
        #                  AFFICHAGE DE L'IMAGE RGB                       #
        #               AVEC OBJETS EN MOUVEMENT DETECTES                 #           
        #                  ET DU MASQUE DE MOUVEMENT                      #
        ###################################################################        


        # --- Calcul du fps théorique de l'IA et affichage sur l'image RGB
        fps_ia = 1000 / inference_time if inference_time > 0 else 0
        text_speed = f"Inference: {inference_time:.1f} ms ({fps_ia:.0f} analyzed images per sec) | Press ECHAP to quit"
        cv2.putText(img, text_speed, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Affichage sur le masque
        text_mask = f"History time: {hist_time:.1f} sec - Threshold: {varThreshold:.1f} | Press ECHAP to quit"
        cv2.putText(fg_mask, text_mask, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

        # --- Enregistrement vidéo
        video_writer.write(img)
                
        cv2.imshow("RGB + Calcul de la distance avec depth", img)        
        cv2.imshow("Masque de Mouvement MOG2", fg_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break


except Exception as e:
    print(f"Une erreur est survenue pendant la capture : {e}")

finally:

    if 'video_writer' in locals():
        video_writer.release()
        print("Fichier vidéo enregistré et fermé avec succès.")

    pipeline.stop()
    print("Arrêt de la caméra !")
    cv2.destroyAllWindows()