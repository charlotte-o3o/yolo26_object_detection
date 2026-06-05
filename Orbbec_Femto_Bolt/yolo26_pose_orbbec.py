import os
os.environ["DISPLAY"] = ":1"

import threading
import queue
import cv2
import time
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode
from ultralytics import YOLO

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
        results = model(frame_bgr, verbose=False)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000.0
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        annotated_frame = results[0].plot(labels=False)

        boxes = results[0].boxes
        keypoints_object = results[0].keypoints
        num_persons = len(boxes) if boxes is not None else 0



        if boxes is not None and keypoints_object is not None and frame_depth is not None:
            for box in boxes:

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
                    target_name = "Torse"

                    if cx == 0 and cy == 0:
                        continue

                    h, w = frame_depth.shape
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))

                    distance_box_m = frame_depth[cy, cx] / 1000.0
 
                    if distance_box_m > 0:
                        text_dist = f"{distance_box_m:.2f}m"
                    else:
                        text_dist = "Inconnue"

                    # Dessiner un petit point au centre de la boîte
                    cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                    cv2.putText(annotated_frame, text_dist, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(annotated_frame, f"Person(s): {num_persons}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_time_ms:.1f} ms ({fps:.0f} analyzed images per sec) | Press ECHAP to quit", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not display_queue.full():
            display_queue.put(annotated_frame)

infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()

try:
    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    config.set_align_mode(OBAlignMode.SW_MODE)
    pipe.start(config)

    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None:
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
            annotated = display_queue.get_nowait()
            cv2.imshow("YOLO26-Pose", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    inference_queue.put(None)
    pipe.stop()
    cv2.destroyAllWindows()