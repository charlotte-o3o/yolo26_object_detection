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
print("Tentative d'ouverture de la RealSense...")
pipeline.start(config)
print("Caméra démarrée avec succès !") # Si ça n'affiche pas ça, ça bloque au démarrage

print("RealSense D435f démarrée. YOLO26 est prêt sur le GPU. Appuyez sur 'q' pour quitter.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("Alerte : Pas de flux vidéo couleur reçu !")
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Lancer YOLO
        results = model(frame, stream=True, device=0, verbose=False)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("YOLO26 - Flux Intel RealSense D435f", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
except Exception as e:
    print(f"Une erreur est survenue pendant la capture : {e}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()