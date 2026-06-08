import os
import sys
import contextlib
import threading
import queue
import cv2
import random
import time
import numpy as np
from ultralytics import YOLO

# Désactivation des logs graphiques d'arrière-plan
os.environ["DISPLAY"] = ":1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

# Import propre d'Orbbec SDK
with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

# Configuration globale
SAVE_MODE = False
RECORD_MODE = True
CONFIDENCE_THRESHOLD = 0.50
MIN_SAVING_INTERVAL = 1.0

OUTPUT_DIR = "captures_img"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

VIDEO_DIR = "captures_videos"
if RECORD_MODE and not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

last_save_time = 0
video_writer = None

# Chargement de ton modèle personnalisé
model = YOLO("alien_plushie.pt")

pipe = Pipeline()
config = Config()

# Détermination d'une couleur d'affichage (Bleu pour le texte/boîte)
b = random.randint(0, 255)
g = random.randint(0, 255)
r = random.randint(0, 255)
box_color = (b, g, r)

# File : transfert sécurisé (Frame BGR, Frame Profondeur, Largeur, Hauteur) vers YOLO
inference_queue = queue.Queue(maxsize=1)
# File : transfert des images traitées vers l'affichage (Frame annotée, Nombre d'objets)
display_queue = queue.Queue(maxsize=1)

def inference_worker():
    while True:
        item = inference_queue.get()
        if item is None:
            break

        # Récupération sécurisée des données de la caméra
        frame_bgr, current_frame_depth, width, height = item

        start_time = time.perf_counter()
        results = model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
        results = list(results)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000.0
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        # On génère l'affichage de base de YOLO sans labels par défaut
        annotated_frame = frame_bgr.copy()

        boxes = results[0].boxes
        num_objects = len(boxes) if boxes is not None else 0

        if boxes is not None and current_frame_depth is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confie = float(box.conf[0]) * 100  # Confiance en %
                label = model.names[class_id]       # Nom de la classe

                # Coordonnées de la Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calcul du centre de la boîte pour la mesure de distance
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # Sécurisation des coordonnées par rapport à la taille de l'image
                x_center = max(0, min(x_center, width - 1))
                y_center = max(0, min(y_center, height - 1))

                # Lecture de la distance en mètres (Z16 brute / 1000)
                distance = current_frame_depth[y_center, x_center] / 1000.0

                if distance > 0:
                    text_dist = f"{distance:.2f}m"
                else:
                    text_dist = "dist. inconnue"

                # Construction de la chaîne d'affichage personnalisée requis : Classe, Confiance et Distance
                custom_label = f"{label} ({confie:.1f}%) : {text_dist}"

                # Dessiner la Bounding Box et l'étiquette
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, custom_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Dessiner un petit point rouge au centre de la cible de calcul de profondeur
                cv2.circle(annotated_frame, (x_center, y_center), 5, (0, 0, 255), -1)

                #print(f"Détection - Object: {label} | Confiance: {confie:.1f}% | Distance: {text_dist}")

        # Statistiques en haut à gauche
        cv2.putText(annotated_frame, f"Object(s): {num_objects}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_time_ms:.1f} ms ({fps:.0f} FPS) | Press ECHAP to quit", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not display_queue.full():
            display_queue.put((annotated_frame, num_objects))

# Démarrage du thread d'arrière-plan
infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()

print()
print("----- YOLO26 - Détection de Peluches (Orbbec Femto Bolt) -----")
print()
if SAVE_MODE:
    print("Mode sauvegarde d'images activé ! Les images des objets détectés seront enregistrées dans 'captures_img'.")
else:
    print("Mode sauvegarde désactivé.")
print()

if RECORD_MODE:
    print("Mode enregistrement vidéo activé ! Le flux sera enregistré dans 'captures_videos/'.")
else:
    print("Mode enregistrement vidéo désactivé.")
print()

try:
    # Démarrage des capteurs de la caméra
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    try:
        config.set_align_mode(OBAlignMode.SW_MODE)
    except Exception as e:
        print(f"[REMARQUE] Alignement SW automatique ou non supporté : {e}")

    pipe.start(config)

    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            continue

        # Décodage de l'image couleur brute
        data = color_frame.get_data()
        enc_img = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        height, width, _ = frame_bgr.shape
        local_frame_depth = None
        
        if RECORD_MODE and video_writer is None:
            timestamp_vid = time.strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(VIDEO_DIR, f"capture_{timestamp_vid}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 20.0 FPS est une bonne valeur standard pour éviter l'effet accéléré à la relecture
            video_writer = cv2.VideoWriter(video_path, fourcc, 25 .0, (width, height))
            print(f"[INFO] Fichier vidéo créé !")

        # Traitement et structuration de la matrice de profondeur brute
        if depth_frame is not None:
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width())
            )
            local_frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

        # Envoi asynchrone sécurisé des frames vers le worker YOLO
        if not inference_queue.full():
            inference_queue.put((frame_bgr.copy(), local_frame_depth.copy() if local_frame_depth is not None else None, width, height))

        # Récupération et affichage graphique (Thread Principal)
        if not display_queue.empty():
            annotated, num_objects = display_queue.get_nowait()

            current_datetime_str = time.strftime("%d/%m/%Y  %H:%M:%S")
            cv2.putText(annotated, current_datetime_str, (width - 225, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if RECORD_MODE and video_writer is not None:
                video_writer.write(annotated)
            
            cv2.imshow("YOLO26 - Alien Plushie Detection", annotated)

            # Gestion des captures d'images en cas de détection
            if num_objects > 0:
                if SAVE_MODE:
                    current_time = time.time()
                    if current_time - last_save_time >= MIN_SAVING_INTERVAL:
                        last_save_time = current_time
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{OUTPUT_DIR}/detection_{timestamp}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"[INFO] {num_objects} objets(s) détecté(s) à {time.strftime('%H:%M:%S')} (image enregistrée)")
                else:
                    print(f"[INFO] {num_objects} objets(s) détecté(s) à {time.strftime('%H:%M:%S')}")

        # waitKey(1) s'exécute à chaque itération pour empêcher le freeze sous Ubuntu
        if cv2.waitKey(1) & 0xFF == 27:
            break


except Exception as e:
    print(f"Une erreur est survenue pendant l'exécution : {e}")


finally:
    # Extinction propre des flux et de l'interface
    inference_queue.put(None)
    if video_writer is not None:
        video_writer.release()
        print("[INFO] Enregistrement vidéo finalisé et sauvegardé avec succès.")
    try:
        pipe.stop()
    except:
        pass
    print("\nArrêt de la caméra !")
    cv2.destroyAllWindows()