import cv2
from ultralytics import YOLO

# 1. Charger le modèle YOLO26 (version 'n' pour nano, ultra-rapide sur CPU)
# Si le fichier .pt n'est pas sur votre ordi, Ultralytics le télécharge tout seul
model = YOLO("yolo26n.pt") 

# 2. Ouvrir la webcam (0 est généralement la webcam intégrée)
cap = cv2.VideoCapture(4)

print("Lancement de YOLO26. Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Impossible d'accéder à la webcam.")
        break

    # 3. Lancer la détection (NMS-Free, traitement direct)
    # stream=True permet de traiter le flux vidéo de manière ultra-fluide
    # results = model(frame, stream=True)
    results = model(frame, stream=True, device=0)

    # 4. Afficher les résultats sur l'image
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLO26 - Detection Temps Reel", annotated_frame)

    # Quitter si on appuie sur la touche échap
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()