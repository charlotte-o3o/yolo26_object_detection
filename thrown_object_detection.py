import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import time

# --- CONFIGURATION DE LA COULEUR DE L'OBJET (HSV) ---
# Exemple pour du ROUGE VIF (Teinte proche de 0 ou 180)
SEUIL_BAS_COULEUR = np.array([35, 50, 50])
SEUIL_HAUT_COULEUR = np.array([85, 255, 255])
# Astuce : Pour du ROUGE, utilisez : [0, 120, 70] à [10, 255, 255]
# Astuce : Pour du VERT, utilisez : [35, 50, 50] à [85, 255, 255]
# Astuce : Pour du BLEU, utilisez : [90, 50, 50] à [130, 255, 255]

# --- VARIABLES DE SÉCURITÉ POUR L'AFFICHAGE DU LANCER ---
derniere_detection_lancer = 0
DELAI_ANTI_REBOND = 2.0  # Temps en secondes avant de pouvoir ré-afficher un nouveau lancer
alerte_lancer_active = False
temps_debut_alerte = 0

# --- CONFIGURATION CAMÉRA REALSENSE ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !")

# --- INITIALISATION DE LA DÉTECTION DE MOUVEMENT ---
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
align = rs.align(rs.stream.color)

# Chargement de YOLO26 - Filtré UNIQUEMENT sur l'humain (classe 0)
model = YOLO('yolo26n.pt')

try:
    while True:
        frames = pipeline.poll_for_frames()
        if not frames: 
            continue

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame: 
            continue

        img = np.asanyarray(color_frame.get_data())
        temps_actuel = time.time()

        # 1. Masque de mouvement global (votre optimisateur de performance)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        fg_mask = backSub.apply(blur)
        
        if np.sum(fg_mask == 255) < 500:
            # Si aucun mouvement, on affiche l'image brute et on passe à la suite
            cv2.imshow("RGB + Tracking Hybride (Sans Musique)", img)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        # 2. APPEL YOLO - UNIQUEMENT SUR L'HUMAIN (classe 0)
        results = model(img, stream=True, verbose=False, classes=[0])
        
        humain_trouve = False
        hx1, hy1, hx2, hy2 = 0, 0, 0, 0

        for r in results:
            for box in r.boxes:
                # On mémorise la boîte de la première personne détectée
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                humain_trouve = True
                
                # Dessin du rectangle autour de l'humain
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                cv2.putText(img, "Humain", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break # On ne traite qu'un seul humain principal pour le lancer

        # 3. SI HUMAIN TROUVÉ -> RECHERCHE DE L'OBJET PAR SA COULEUR
        if humain_trouve:
            # Conversion en HSV pour isoler proprement la couleur
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            masque_couleur = cv2.inRange(hsv_img, SEUIL_BAS_COULEUR, SEUIL_HAUT_COULEUR)
            
            # L'objet doit être de la bonne couleur ET en mouvement
            masque_objet = cv2.bitwise_and(masque_couleur, fg_mask)

            # Recherche des paquets de pixels correspondants
            contours, _ = cv2.findContours(masque_objet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                aire = cv2.contourArea(contour)
                if aire < 100: 
                    continue # Ignore les petits parasites isolés

                # Boîte englobante de l'objet trouvé par sa couleur
                ox, oy, ow, oh = cv2.boundingRect(contour)
                ox_center = int(ox + ow/2)
                oy_center = int(oy + oh/2)
                
                # Récupération de la distance RealSense au centre de l'objet
                distance_objet = depth_frame.get_distance(ox_center, oy_center)

                # --- LOGIQUE GEOMETRIQUE DU LANCER ---
                # Si le centre de l'objet franchit la limite gauche ou droite de la boîte de l'humain
                if ox_center < hx1 or ox_center > hx2:
                    if (temps_actuel - derniere_detection_lancer) > DELAI_ANTI_REBOND:
                        print(f"⚠️ OBJET LANCÉ DÉTECTÉ à {distance_objet:.2f}m !")
                        derniere_detection_lancer = temps_actuel
                        alerte_lancer_active = True
                        temps_debut_alerte = temps_actuel

                # Dessin de la boîte de l'objet en VERT
                cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                cv2.putText(img, f"Objet ({distance_objet:.2f}m)", (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. GESTION DE L'AFFICHAGE DE L'ALERTE SUR L'ÉCRAN (Pendant 1.5 seconde)
        if alerte_lancer_active:
            if (temps_actuel - temps_debut_alerte) < 1.5:
                cv2.putText(img, "OBJET LANCE !", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            else:
                alerte_lancer_active = False

        # Affichage du flux final
        cv2.imshow("RGB + Tracking Hybride (Sans Musique)", img)
        
        # Touche ÉCHAP pour quitter
        if cv2.waitKey(1) & 0xFF == 27: 
            break

except Exception as e:
    print(f"Une erreur est survenue : {e}")
finally:
    # Fermeture propre de la caméra uniquement
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Caméra coupée et fenêtres fermées.")