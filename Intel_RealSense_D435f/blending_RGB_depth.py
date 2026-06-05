import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 

# 1. Configurer la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !")

# --- Filtre de colorisation pour les données de profondeur
# --- Convertit la matrice des données brutes en une image couleur
colorizer = rs.colorizer()

print("Appuier sur la touche ECHAP dans l'une des fenêtres vidéo pour quitter.")

# Aligner le flux de profondeur sur le flux couleur
align = rs.align(rs.stream.color)
model = YOLO('yolo26n.pt') # Chargement du modèle

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Alerte : Pas de flux vidéo couleur et/ou profondeur reçu !")
            continue
        
         # Convertir les flux bruts en tableaux numpy
        img_rgb = np.asanyarray(color_frame.get_data())
        colorized_depth = colorizer.colorize(depth_frame)
        img_depth = np.asanyarray(colorized_depth.get_data())

        # 2. Créer l'image superposée (blending) AVANT de dessiner les boîtes
        # Comme ça, le fond mélangé est prêt
        img_superposee = cv2.addWeighted(img_rgb, 0.6, img_depth, 0.4, 0)
        
        # 2. Inférence YOLO26 (uniquement sur le flux RGB)
        results = model(img_rgb)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center, y_center = int(box.xywh[0][0]), int(box.xywh[0][1])
                
                # Récupérer la distance réelle (3D)
                distance = depth_frame.get_distance(x_center, y_center)
                label = model.names[int(box.cls[0])]
                text = f"{label}: {distance:.2f}m"
                
                # --- Fenêtre 1 : Dessin sur l'image RGB classique ---
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img_rgb, (x_center, y_center), 4, (0, 0, 255), -1)
                
               # --- Fenêtre 2 : Dessin sur l'image Superposée ---
                # On remet les mêmes boîtes pour voir la correspondance en transparence
                cv2.rectangle(img_superposee, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_superposee, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img_superposee, (x_center, y_center), 4, (0, 0, 255), -1)

                print(f"Objet: {model.names[int(box.cls[0])]} | Distance: {distance:.2f}m")
        
        cv2.imshow("Fenetre 1 - RGB + YOLO26", img_rgb)
        cv2.imshow("Fenetre 2 - Superposition Blending (RGB+Depth)", img_superposee)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()