import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import random

# --- Configuration de la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()

# --- Stream couleur
# --- format.bgr8 => Format d'image classique : BGR = RGB inversé ; 8 bits par canal de couleur (B, G, R) => 24
# --- Chaque pixel est codé sur 24 bits
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# --- Stream profondeur = carte qui mesure les distances
# --- format.z16 => 16 bits par pixel (unsigned int 16) + de précision que seulement 8 bits
# --- Valeur de l'int = dist. en mm
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !")
print("Appuier sur la touche ECHAP dans la fenêtre vidéo pour quitter.")

# --- Alignement du flux de profondeur sur le flux couleur
# --- Le flux couleur sert de base à l'alignement
align = rs.align(rs.stream.color)

# --- Chargement du modèle YOLO26 nano
model = YOLO('yolo26n.pt')
"""
# --- Assignation d'une couleur pour toutes les classes du dataset
couleurs_classes = {}
for class_id in model.names.keys():
    # --- Génère 3 entiers aléatoires entre 0 et 255
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    couleurs_classes[class_id] = (b, g, r)"""

try:
    while True:

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
        results = model(img, stream=True, verbose=False)

        inference_time = 0.0
        
        for r in results:
            
            # --- Récupération de bounding boxes détectées (objets Boxes)
            boxes = r.boxes

            # --- Récupération du temps d'inférence en ms
            if 'inference' in r.speed:
                inference_time = r.speed['inference']

            for box in boxes:
                
                # --- Récupération de l'ID de la classe
                class_id = int(box.cls[0])
                # couleur = couleurs_classes[class_id]

                # --- Coordonnées du centre de la boîte
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # --- Centre de la boîte pour calculer la distance
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                
                # --- Distance en m (grâce à l'alignement)
                distance = depth_frame.get_distance(x_center, y_center)

                # --- Nommage de la classe de l'objet détecté grâce au mapping de names
                label = model.names[int(class_id)]

                # --- Probabilité de classe (confiance)
                confiance = float(box.conf[0]) * 100

                # --- Dessin sur l'image RGB des BB, nom de classe, centre des BB, distance...
                text = f"{label} ({confiance:.1f}%) : {distance:.2f}m"

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Changer (255, 0, 0) par couleur pour les classes colorées
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Changer (255, 0, 0) par couleur pour les classes colorées
                cv2.circle(img, (x_center, y_center), 4, (0, 0, 255), -1)

                # print(f"Objet: {model.names[int(box.cls[0])]} | Distance: {distance:.2f}m")


        # --- Calcul du fps théorique de l'IA et affichage sur l'image RGB
        fps_ia = 1000 / inference_time if inference_time > 0 else 0

        text_speed = f"Inference: {inference_time:.1f} ms ({fps_ia:.1f} analyzed images per sec)"

        cv2.putText(img, text_speed, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
        cv2.imshow("RGB + Calcul de la distance avec depth", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break


except Exception as e:
    print(f"Une erreur est survenue pendant la capture : {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()