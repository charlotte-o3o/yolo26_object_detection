import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Charger le modèle YOLO26 sur le GPU (device=0)
model = YOLO("yolo26n.pt")

# 2. Configurer le flux de la caméra Intel RealSense
pipeline = rs.pipeline()
config = rs.config()

# On active le flux couleur (RGB) - Résolution standard et rapide : 640x480 à 30 FPS
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# On active aussi le flux de profondeur (Depth) pour le futur de ton projet
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Démarrer la caméra
print("Tentative d'ouverture de la RealSense...")
profile = pipeline.start(config)
print("Caméra démarrée avec succès !")

align_to = rs.stream.color
align = rs.align(align_to)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

print("RealSense D435f démarrée. YOLO26 est prêt sur le GPU. Appuyez sur 'q' pour quitter.")

try:
    while True:
        # Attendre le prochain groupe d'images (couleur + profondeur)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        # if not color_frame:
        #    print("Alerte : Pas de flux vidéo couleur reçu !")
        #    continue

        if not depth_frame or not color_frame:
            continue

        # Convertir l'image RealSense en tableau Numpy lisible par OpenCV et YOLO
        frame = np.asanyarray(color_frame.get_data())

        # 3. Lancer la détection YOLO26 (NMS-Free, ultra rapide sur GPU)
        results = model(frame, stream=True, device=0, verbose=False)

        for r in results:
            for box in r.boxes:
                # Récupérer le centre de la boîte de détection (pixel X, pixel Y)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Obtenir le nom de l'objet (ex: "sports ball")
                cls = int(box.cls[0])
                label = model.names[cls]

                # 4. Récupérer la distance au pixel central (en mètres)
                distance = depth_frame.get_distance(cx, cy)

                if distance > 0:
                    # 5. MATHS TRIDIMENSIONNELLES : Convertir le pixel + la distance en coordonnées (X,Y,Z) réelles
                    # Cette fonction utilise la trigonométrie et les coordonnées intrinsèques de la caméra
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)
                    
                    # point_3d contient maintenant : [X_gauche_droite, Y_haut_bas, Z_profondeur] en mètres !
                    X_metres, Y_metres, Z_metres = point_3d

                    # Afficher les coordonnées sur l'écran pour debug
                    text = f"{label}: X={X_metres:.2f}m, Y={Y_metres:.2f}m, Z={Z_metres:.2f}m"
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO26 - Vision 3D Temps Reel", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"Une erreur est survenue pendant la capture : {e}")

finally:
    # Arrêter proprement la caméra et fermer les fenêtres en quittant
    pipeline.stop()
    cv2.destroyAllWindows()