import os
import time
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode

# ==========================================
#               CONFIGURATION
# ==========================================
RECORD_MODE = True  # True pour enregistrer la vidéo, False pour regarder uniquement le flux
VIDEO_DIR = "captures_videos"

if RECORD_MODE and not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
    print(f"Dossier '{VIDEO_DIR}' créé avec succès !")

video_writer = None

# 1. Initialiser le pipeline Orbbec
pipe = Pipeline()
config = Config()

try:
    # Configurer le flux Couleur (RGB)
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    
    # Configurer le flux Profondeur (Depth)
    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    # CRUCIAL : Aligner le flux de profondeur sur le flux couleur
    config.set_align_mode(OBAlignMode.SW_MODE)

    # Démarrer la caméra
    pipe.start(config)
    print("Caméra Orbbec Femto Bolt connectée avec succès !")
    if RECORD_MODE:
        print("Mode enregistrement vidéo ACTIVÉ.")

    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        # 2. Récupérer l'image couleur et profondeur
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is not None and depth_frame is not None:
            # Convertir la frame couleur en tableau NumPy (BGR pour OpenCV)
            data = color_frame.get_data()
            enc_img = np.frombuffer(data, dtype=np.uint8)
            frame_rgb = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
            
            if frame_rgb is None:
                continue

            # Récupérer les dimensions réelles après décodage
            height, width, _ = frame_rgb.shape

            # INITIALISATION DU VIDEOWRITER (dès qu'on connaît la taille de l'image)
            if RECORD_MODE and video_writer is None:
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                video_path = os.path.join(VIDEO_DIR, f"capture_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # 25.0 ou 30.0 FPS. Ajuste cette valeur si ta vidéo finale est accélérée ou ralentie.
                video_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))
                print(f"[INFO] Enregistrement démarré dans : {video_path}")

            # Convertir la profondeur en tableau NumPy (valeurs en mm)
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Redimensionner la profondeur à la taille exacte du RGB
            frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

            # Exemple d'utilisation : prendre la distance au centre de l'image
            cx, cy = width // 2, height // 2
            distance_mm = frame_depth[cy, cx]
            distance_m = distance_mm / 1000.0

            # Afficher les infos sur l'image
            #cv2.circle(frame_rgb, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame_rgb, f"Z au centre: {distance_m:.2f} m", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ENREGISTREMENT DE LA FRAME (Écriture dans le fichier vidéo)
            if RECORD_MODE and video_writer is not None:
                video_writer.write(frame_rgb)

            # Afficher le flux vidéo à l'écran
            cv2.imshow("Femto Bolt - RGB", frame_rgb)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # FERMETURE DU FICHIER VIDEO
    if video_writer is not None:
        video_writer.release()
        print("[INFO] Fichier vidéo enregistré et fermé avec succès.")

    pipe.stop()
    cv2.destroyAllWindows()
    print("Caméra arrêtée proprement.")