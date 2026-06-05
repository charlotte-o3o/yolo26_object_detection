import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
from ultralytics import YOLO

# 1. Charger le modèle YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")  # téléchargement automatique au premier lancement

# 2. Initialiser le pipeline Orbbec
pipe = Pipeline()
config = Config()

try:
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    config.set_align_mode(OBAlignMode.SW_MODE)
    pipe.start(config)
    print("Caméra Orbbec Femto Bolt connectée avec succès !")

    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # On affiche quand même si depth manque, mais on a besoin du color
        if color_frame is None:
            continue

        # Décoder l'image couleur MJPEG → BGR
        data = color_frame.get_data()
        enc_img = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        height, width, _ = frame_bgr.shape

        # Traitement profondeur (optionnel si la frame est disponible)
        distance_m = None
        if depth_frame is not None:
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width())
            )
            frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
            cx, cy = width // 2, height // 2
            distance_m = frame_depth[cy, cx] / 1000.0

        # 3. Inférence YOLO Pose (sur l'image BGR directement)
        results = model(frame_bgr, verbose=False)

        # 4. Dessiner les résultats YOLO sur une copie de l'image
        annotated_frame = results[0].plot()  # dessine squelettes + bounding boxes

        # 5. Afficher le nombre de personnes détectées
        num_persons = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Personnes: {num_persons}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 6. Afficher la distance au centre si disponible
        if distance_m is not None:
            cx, cy = width // 2, height // 2
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"Z centre: {distance_m:.2f} m", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. Toujours afficher la fenêtre
        cv2.imshow("Femto Bolt - RGB + Pose", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()