import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2

# 1. Configurer la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
# Attention : 1280x720 demande pas mal de ressources, assure-toi que ton PC suit !
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !")

colorizer = rs.colorizer()

# --- INITIALISATION DE LA DÉTECTION DE MOUVEMENT ---
# history=500 frames, varThreshold=16 (plus c'est bas, plus c'est sensible au mouvement)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

print("Appuyer sur la touche ECHAP dans la fenêtre vidéo pour quitter.")

align = rs.align(rs.stream.color)
model = YOLO('yolo26n.pt') 

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Alerte : Pas de flux vidéo couleur et/ou profondeur reçu !")
            continue

        # Convertir en tableaux numpy
        img = np.asanyarray(color_frame.get_data())
        
        # --- CALCUL DU MASQUE DE MOUVEMENT ---
        # On applique un léger flou pour éviter que le bruit numérique de la cam soit vu comme du mouvement
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        fg_mask = backSub.apply(blur)
        
        # Nettoyage du masque (enlève les petits pixels isolés qui "clignotent")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 2. Prediction avec YOLO26
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # --- FILTRE DE MOUVEMENT ---
                # On regarde la zone de la boîte de détection dans le masque de mouvement
                roi_motion = fg_mask[y1:y2, x1:x2]
                
                # S'il n'y a pas assez de pixels blancs (mouvement) dans cette boîte, on ignore l'objet
                # Ici, il faut au moins 150 pixels en mouvement dans la boîte pour l'activer
                if np.sum(roi_motion == 255) < 150:
                    continue # L'objet est immobile, on passe au suivant sans le dessiner !
                
                # Centre de la boîte pour calculer la distance
                x_center, y_center = int(box.xywh[0][0]), int(box.xywh[0][1])
                
                # Distance via la RealSense (en mètres)
                distance = depth_frame.get_distance(x_center, y_center)
                label = model.names[int(box.cls[0])]
                text = f"{label} [MOUVEMENT]: {distance:.2f}m"

                # --- Dessin uniquement sur les objets en mouvement ---
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Rouge pour le mouvement !
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(img, (x_center, y_center), 4, (0, 0, 255), -1)
                
                print(f"Objet en mouvement: {label} | Distance: {distance:.2f}m")

        # Affichage des fenêtres
        cv2.imshow("Objets en mouvement uniquement (YOLO26)", img)
        # Optionnel : affiche le masque noir et blanc pour voir ce que la caméra considère "en mouvement"
        #cv2.imshow("Masque de mouvement (Témoin)", fg_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"Une erreur est survenue pendant la capture : {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()