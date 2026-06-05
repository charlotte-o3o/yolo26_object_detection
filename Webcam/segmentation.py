import cv2
import numpy as np
from ultralytics import YOLO

# 1. Charger le modèle YOLO26 de SEGMENTATION sur le GPU
model = YOLO("yolo26n-seg.pt")

# 2. Initialiser la webcam intégrée de l'ordinateur
# '0' est l'index de la caméra par défaut de ton PC
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam de l'ordinateur.")
    exit()

print("Webcam PC démarrée. YOLO26 Segmentation actif sur GPU.")
print("Appuyez sur 'q' pour quitter.")

try:
    while True:
        # Capturer image par image
        success, frame = cap.read()
        
        if not success:
            print("Erreur : Impossible de lire le flux de la webcam.")
            break

        # 3. Lancer la détection et la segmentation (NMS-Free natif)
        results = model(frame, stream=True, device=0, verbose=False)

        for r in results:
            # Si YOLO26 trouve des objets, on récupère l'image annotée avec les masques
            if r.masks is not None:
                annotated_frame = r.plot(boxes=False)
                cv2.imshow("YOLO26 - Segmentation Webcam PC", annotated_frame)
            else:
                # Si rien n'est détecté, on affiche l'image brute de la webcam
                cv2.imshow("YOLO26 - Segmentation Webcam PC", frame)

        # Quitter si on appuie sur la touche échap
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Libérer proprement la caméra et fermer les fenêtres d'affichage
    cap.release()
    cv2.destroyAllWindows()