import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode

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

    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        # 2. Récupérer l'image couleur
        color_frame = frames.get_color_frame()
        # Récupérer l'image de profondeur alignée
        depth_frame = frames.get_depth_frame()

        if color_frame is not None and depth_frame is not None:
            # Convertir la frame couleur en tableau NumPy (BGR pour OpenCV/YOLO)
            # 1. Récupérer les données brutes (compressées en MJPEG)
            data = color_frame.get_data()
            # 2. Convertir en tableau 1D NumPy
            enc_img = np.frombuffer(data, dtype=np.uint8)
            # 3. Utiliser OpenCV pour décoder le JPEG en image BGR (parfait pour YOLO et l'affichage)
            frame_rgb = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
            # Au cas où le décodage échoue sur une frame corrompue
            if frame_rgb is None:
                continue

            # Récupérer les dimensions réelles après décodage
            height, width, _ = frame_rgb.shape

            # Convertir la profondeur en tableau NumPy (valeurs en mm)
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))
            # 3. CRUCIAL : On redimensionne la profondeur à la taille exacte du RGB
            # On utilise INTER_NEAREST pour ne pas fausser les valeurs de distance en millimètres
            frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

            # 3. Exemple d'utilisation : prendre la distance au centre de l'image
            cx, cy = width // 2, height // 2
            distance_mm = frame_depth[cy, cx]
            distance_m = distance_mm / 1000.0

            # Afficher les infos sur l'image
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame_rgb, f"Z au centre: {distance_m:.2f} m", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Afficher le flux vidéo
            cv2.imshow("Femto Bolt - RGB", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()