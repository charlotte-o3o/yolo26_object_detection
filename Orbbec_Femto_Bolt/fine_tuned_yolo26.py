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


""" ----------------------------------------------------------------------------------------------- """
""" ---------------------- YOLO26 for Orbbec Femto Bolt for Object Detection ---------------------- """
""" ----------------------------------------------------------------------------------------------- """

"""RAJOUTER TRACKING (=> sachant qu'il n'y aura qu'un seul objet 
=> a chaque fois qu'il y en a un détecté ça sera le même)
model.track()"""

###################################################################
#                         CONFIGURATION                           # 
###################################################################

# --- Logs config
os.environ["DISPLAY"] = ":1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

# --- Import propre d'Orbbec SDK
with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

# --- Configuration globale (modes, seuils)
SAVE_MODE = False
RECORD_MODE = False
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

# --- Chargement du modèle fine tuned
#model = YOLO("alien_plushie.pt")
model = YOLO("alien_plushie_v2.pt")

# --- Config de la caméra
pipe = Pipeline()
config = Config()

# --- Assignation d'une couleur pour la/les classe(s)
b = random.randint(0, 255)
g = random.randint(0, 255)
r = random.randint(0, 255)
box_color = (b, g, r)

""" 
    Mise en place d'une architecture mutli-thread
    - thread principal : lecture des images de la caméra et affichage écran
    - thread YOLO
"""
# --- Queue : transfert sécurisé (Frame BGR, Frame Profondeur, Largeur, Hauteur) vers YOLO
# --- 1 frame max dans la queue => si une nouvelle image arrive, on écrase la précédente
# --- Traitement par YOLO de l'image la plus récente
# --- Vers le thread YOLO
inference_queue = queue.Queue(maxsize=1)
# --- Queue : transfert des images traitées vers l'affichage (Frame annotée, Nombre d'objets)
# --- 1 frame par max dans la queue => affichage de la détection la plus récente
# --- Vers le thread principal
display_queue = queue.Queue(maxsize=1)

def inference_worker():

    while True:

        ###################################################################
        #                    RECUPERATION DE L'IMAGE                      # 
        ###################################################################

        # --- Récupération de la frame dans la queue
        item = inference_queue.get()

        if item is None:
            break

        # --- Récupération sécurisée des données de la caméra
        frame_bgr, current_frame_depth, width, height = item

        ###################################################################
        #                        DETECTION D'OBJETS                       # 
        ###################################################################

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
            Filtrage de classes : 
            Passer dans la méthode model() un paramètre classes=[x, y, z, ...] ; x, y et z sont les id associés aux classes dans names
            Afin de ne tester les objets détectés que sur ces classes
        """

        # --- Application du modèle YOLO - détection d'objets        
        start_time = time.perf_counter()
        # --- Renvoie un générateur
        results = model(frame_bgr, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        # --- Conversion en liste : utiliser indices, connaître la taille, lire les données plusieurs fois
        results = list(results)
        end_time = time.perf_counter()

        # --- Calcul du temps d'inférence de YOLO
        inference_time_ms = (end_time - start_time) * 1000.0
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        # --- Affichage de base de YOLO sans labels par défaut
        annotated_frame = frame_bgr.copy()
        # --- Batch = 1 seule image donc le résultat de l'analyse est toujours dans results[0]
        boxes = results[0].boxes
        num_objects = len(boxes) if boxes is not None else 0

        ###################################################################
        #                         TRAITEMENT DES                          #
        #                         BOUNDING BOXES                          #
        #                          ET AFFICHAGE                           #
        ###################################################################

        if boxes is not None and current_frame_depth is not None:

            for box in boxes:
                
                # --- ID de la classe de l'objet identifié
                class_id = int(box.cls[0])
                # --- Nom de la classe
                label = model.names[class_id]
                # --- Confiance associée à la détection
                confie = float(box.conf[0]) * 100 
                # --- Coordonnées de la Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # --- Calcul du centre de la boîte pour la mesure de distance
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # --- Sécurisation des coordonnées par rapport à la taille de l'image
                x_center = max(0, min(x_center, width - 1))
                y_center = max(0, min(y_center, height - 1))

                # --- Lecture de la distance en mètres (Z16 brute / 1000)
                distance = current_frame_depth[y_center, x_center] / 1000.0

                if distance > 0:
                    text_dist = f"{distance:.2f}m"
                else:
                    text_dist = "dist. inconnue"

                # --- Construction de la chaîne d'affichage (classe, confiance et distance)
                custom_label = f"{label} ({confie:.1f}%) : {text_dist}"

                # --- Dessin de la Bounding Box et l'étiquette
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, custom_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # --- Dessin d'un petit point rouge au centre de la cible de calcul de profondeur
                cv2.circle(annotated_frame, (x_center, y_center), 5, (0, 0, 255), -1)

                #print(f"Détection - Object: {label} | Confiance: {confie:.1f}% | Distance: {text_dist}")

        # --- Infos sur la détection
        cv2.putText(annotated_frame, f"Object(s): {num_objects}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_time_ms:.1f} ms ({fps:.0f} FPS) | Press ECHAP to quit", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not display_queue.full():
            display_queue.put((annotated_frame, num_objects))

# --- Démarrage du thread d'arrière-plan
# --- Tâche parallèle séparée du flux principal
# --- Execute en boucle inference_worker() qui s'occupe de YOLO
# --- daemon=True : thread esclave
infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()

print()
print("----- YOLO26 - Détection d'objets (Orbbec Femto Bolt) -----")
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

###################################################################
#                        RÉCUPÉRATION ET                          #
#                         TRAITEMENT DES                          #
#                       FRAMES DE LA CAMÉRA                       #
###################################################################

try:
    # --- Démarrage des capteurs de la caméra
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    # --- Alignement des frames couleur et profondeur
    try:
        config.set_align_mode(OBAlignMode.SW_MODE)
    except Exception as e:
        print(f"[REMARQUE] Alignement SW automatique ou non supporté : {e}")

    pipe.start(config)

    # --- while True = thread principal de la caméra
    while True:
        
        # --- Attente des frames de la caméra
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            continue

        # --- Décodage de l'image couleur brute
        data = color_frame.get_data()
        enc_img = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        height, width, _ = frame_bgr.shape
        local_frame_depth = None
        
        # --- Config enregistrement vidéo
        if RECORD_MODE and video_writer is None:
            timestamp_vid = time.strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(VIDEO_DIR, f"capture_{timestamp_vid}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps_enregistrement = 25.0 # Augmenter fps_enresgistrement si la vidéo est en accéléré
            video_writer = cv2.VideoWriter(video_path, fourcc, fps_enregistrement, (width, height))
            print(f"[INFO] Fichier vidéo créé !")

        # --- Traitement et structuration de la matrice de profondeur brute
        if depth_frame is not None:
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width())
            )
            local_frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

        # --- Envoi asynchrone sécurisé des frames vers le worker YOLO
        if not inference_queue.full():
            inference_queue.put((frame_bgr.copy(), local_frame_depth.copy() if local_frame_depth is not None else None, width, height))

        # --- Récupération et affichage graphique (Thread Principal)
        if not display_queue.empty():
            annotated, num_objects = display_queue.get_nowait()

            current_datetime_str = time.strftime("%d/%m/%Y  %H:%M:%S")
            cv2.putText(annotated, current_datetime_str, (width - 225, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if RECORD_MODE and video_writer is not None:
                video_writer.write(annotated)
            
            # --- Fenêtre d'affichage
            cv2.imshow("YOLO26 - Alien Plushie Detection", annotated)

            # --- Gestion des captures d'images en cas de détection (si mode enregistrement photos activé)
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

        # --- waitKey(1) s'exécute à chaque itération => empêche le freeze sous Ubuntu
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