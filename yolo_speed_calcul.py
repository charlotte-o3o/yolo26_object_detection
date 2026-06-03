import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import time
import math

# 1. Configurer la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Tentative d'ouverture de la RealSense...")
profile = pipeline.start(config)
print("Caméra démarrée avec succès !")

# Récupérer les paramètres intrinsèques pour la projection 3D
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# Dictionnaire de suivi basé sur l'ID unique du Tracker : { id_unique: (X, Y, Z, timestamp) }
suivi_objets = {}

SEUIL_ALERTE_KMH = 3.0

print("Appuyer sur la touche ECHAP dans la fenêtre vidéo pour quitter.")

align = rs.align(rs.stream.color)
model = YOLO('yolo26n.pt') 

try:
    while True:
        t_debut = time.time()
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        
        # 2. APPLICATION DU TRACKER DE YOLO
        # persist=True permet de garder en mémoire les objets entre chaque frame
        # tracker="bytetrack.yaml" ou "botsort.yaml"
        results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        nouveau_suivi = {}
        
        for r in results:
            # On vérifie que YOLO a bien détecté des boîtes ET qu'il a réussi à leur attribuer un ID
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                boxes_wh = r.boxes.xywh.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy() # Récupération des IDs uniques de suivi
                clss = r.boxes.cls.cpu().numpy()
                
                # On parcourt les objets via leur index
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    x_center, y_center = int(boxes_wh[i][0]), int(boxes_wh[i][1])
                    track_id = int(track_ids[i]) # Ex: 1, 2, 3...
                    label = model.names[int(clss[i])]
                    
                    # Récupérer la profondeur Z
                    distance_z = depth_frame.get_distance(x_center, y_center)
                    if distance_z == 0:
                        continue
                    
                    # 3. Calcul de la position 3D exacte
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_center, y_center], distance_z)
                    x_reel, y_reel, z_reel = point_3d[0], point_3d[1], point_3d[2]
                    
                    vitesse_kmh = 0.0
                    
                    # UNIQUE CLÉ DE SUIVI : On combine le nom et son ID unique (ex: "chair_1")
                    cle_suivi = f"{label}_{track_id}"
                    
                    # 4. CALCUL DE LA VITESSE AVEC LES 3 SÉCURITÉS COMBINÉES
                    if cle_suivi in suivi_objets:
                        x_prec, y_prec, z_prec, t_prec = suivi_objets[cle_suivi]
                        
                        # Distance 3D parcourue
                        distance_parcourue = math.sqrt((x_reel - x_prec)**2 + (y_reel - y_prec)**2 + (z_reel - z_prec)**2)
                        dt = t_debut - t_prec
                        
                        # SÉCURITÉ 1 : Filtre de distance maximale (Anti-téléportation / Changement brusque de shape)
                        if distance_parcourue < 0.4 and dt > 0:
                            vitesse_ms = distance_parcourue / dt
                            vitesse_kmh = vitesse_ms * 3.6
                            
                            # SÉCURITÉ 2 : Filtre de vitesse minimale (Anti-scintillement des infrarouges)
                            if vitesse_kmh < 0.8:
                                vitesse_kmh = 0.0
                        else:
                            vitesse_kmh = 0.0
                    
                    # Enregistrer la position actuelle pour la frame suivante
                    nouveau_suivi[cle_suivi] = (x_reel, y_reel, z_reel, t_debut)
                    
                    # 5. CODE COULEUR DYNAMIQUE
                    if vitesse_kmh >= SEUIL_ALERTE_KMH:
                        couleur = (0, 0, 255) # Rouge
                        statut = "[ALERTE]"
                    else:
                        couleur = (0, 255, 0) # Vert
                        statut = ""

                    # Texte affiché à l'écran incluant l'ID de l'objet
                    text_vitesse = f"{statut} ID:{track_id} {label}: {vitesse_kmh:.1f} km/h | Dist: {distance_z:.2f}m"
                    
                    # Dessiner
                    cv2.rectangle(img, (x1, y1), (x2, y2), couleur, 2)
                    cv2.putText(img, text_vitesse, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
                    cv2.circle(img, (x_center, y_center), 4, couleur, -1)

        # Mettre à jour l'historique global
        suivi_objets = nouveau_suivi

        # Affichage
        cv2.imshow("Tracking Multi-Objets + Vitesse Réelle - YOLO26", img)

        if cv2.waitKey(25) & 0xFF == 27: # Échap pour quitter
            break

except Exception as e:
    print(f"Une erreur est survenue : {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()