import os
os.environ["DISPLAY"] = ":1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import contextlib

with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

import threading
import queue
import cv2
import time
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode
from ultralytics import YOLO

SAVE_MODE = False
CONFIDENCE_THRESHOLD = 0.50
MIN_SAVING_INTERVAL = 0.50

OUTPUT_DIR = "captures_img"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

last_save_time = 0

model = YOLO("yolo26n-pose.pt")

pipe = Pipeline()
config = Config()

# File : frames brutes vers YOLO
inference_queue = queue.Queue(maxsize=1)
# File : frames annotées vers affichage
display_queue = queue.Queue(maxsize=1)

def inference_worker():

    while True:

        item = inference_queue.get()

        if item is None:
            break

        frame_bgr, distance_m, width, height = item

        start_time = time.perf_counter()
        results = model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
        results = list(results)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000.0
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        annotated_frame = results[0].plot(labels=False)

        boxes = results[0].boxes
        keypoints_object = results[0].keypoints
        num_persons = len(boxes) if boxes is not None else 0


        if boxes is not None and keypoints_object is not None and frame_depth is not None:

            kpts = keypoints_object.data.cpu().numpy()

            for i, box in enumerate(boxes):
                # S'assurer que les keypoints existent pour cette personne
                if i >= len(kpts):
                    continue

                person_kpts = kpts[i]

                x_l_shoulder, y_l_shoulder, conf_l = person_kpts[5] # Épaule gauche
                x_r_shoulder, y_r_shoulder, conf_r = person_kpts[6] # Épaule droite

                cx = int((x_l_shoulder + x_r_shoulder) / 2)
                cy = int((y_l_shoulder + y_r_shoulder) / 2)

                if cx == 0 and cy == 0:
                    continue

                h, w = frame_depth.shape
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))

                distance_box_m = frame_depth[cy, cx] / 1000.0

                if distance_box_m > 0:
                    text_dist = f"{distance_box_m:.2f}m"
                else:
                    text_dist = "Dist. inconnue"

                # Dessiner un petit point au centre de la boîte
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.putText(annotated_frame, text_dist, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(annotated_frame, f"Person(s): {num_persons}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_time_ms:.1f} ms ({fps:.0f} analyzed images per sec) | Press ECHAP to quit", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not display_queue.full():
            display_queue.put((annotated_frame, frame_bgr, num_persons))

infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()

print()
print("----- YOLO26 Pose - détection de personnes et calcul de la pose -----")
print()
if SAVE_MODE:
    print("Mode sauvegarde d'images activé ! Les images des personnes détectées seront enregistrées.")
else:
    print("Mode sauvegarde désactivé.")
print()

try:
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)


    try:
        config.set_align_mode(OBAlignMode.SW_MODE)
    except Exception as e:
        print(f"[REMARQUE] Alignement SW non supporté ou automatique : {e}")

    pipe.start(config)

    while True:
        frames = pipe.wait_for_frames(100)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            continue

        data = color_frame.get_data()
        enc_img = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            continue

        height, width, _ = frame_bgr.shape

        distance_m = None
        
        if depth_frame is not None:
            depth_data = depth_frame.get_data()
            raw_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width())
            )
            frame_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
            cx, cy = width // 2, height // 2
            distance_m = frame_depth[cy, cx] / 1000.0

        # Envoi à YOLO (non bloquant)
        if not inference_queue.full():
            inference_queue.put((frame_bgr.copy(), distance_m, width, height))

        # Affichage dans le thread principal
        if not display_queue.empty():

            annotated, original, num_persons = display_queue.get_nowait()

            current_datetime_str = time.strftime("%d/%m/%Y  %H:%M:%S")

            cv2.putText(annotated, current_datetime_str, (width - 225, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            

            cv2.imshow("YOLO26-Pose", annotated)


            if num_persons > 0:
                    if SAVE_MODE:
                        current_time = time.time()
                        if current_time - last_save_time >= MIN_SAVING_INTERVAL:
                            last_save_time = current_time
                            timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
                            filename = f"{OUTPUT_DIR}/detection_{timestamp}.jpg"
                            cv2.imwrite(filename, annotated)
                            print(f"[INFO] {num_persons} personne(s) détectée(s) à {time.strftime('%H:%M:%S')} (image enregistrée)")
                    else: 
                        print(f"[INFO] {num_persons} personne(s) détectée(s) à {time.strftime('%H:%M:%S')}")

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    inference_queue.put(None)
    pipe.stop()
    print()
    print("Arrêt de la caméra !")
    cv2.destroyAllWindows()