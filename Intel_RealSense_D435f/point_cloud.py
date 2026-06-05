import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# 1. Configurer la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)
model = YOLO('yolo26n.pt')

# Outil de calcul RealSense pour extraire les coordonnées XYZ
pointcloud = rs.pointcloud()

# 2. Initialiser la fenêtre de visualisation Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Nuage de Points 3D (Open3D) - YOLO26", width=800, height=600)

# Créer un objet nuage de points vide que l'on va mettre à jour en continu
pcd = o3d.geometry.PointCloud()
is_first_frame = True

print("Utilise la souris sur la fenetre Open3D pour tourner/zoomer.")
print("Appuie sur ECHAP dans la fenetre OpenCV pour quitter.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # Convertir le flux couleur pour OpenCV et Open3D
        img_rgb = np.asanyarray(color_frame.get_data())
        
        # 3. CALCUL DU NUAGE DE POINTS (XYZ + Couleurs)
        pointcloud.map_to(color_frame)
        points = pointcloud.calculate(depth_frame)
        
        # Extraire les coordonnées géométriques (X, Y, Z)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        # Extraire et normaliser les couleurs (Open3D attend des valeurs entre 0 et 1, et en RGB, pas BGR)
        img_rgb_normalized = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) / 255.0
        tex = img_rgb_normalized.reshape(-1, 3)
        
        # Assigner les géométries et couleurs à Open3D
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(tex)
        
        # Inversion de l'axe Y et Z pour que le nuage ne soit pas à l'envers
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # 4. MISE À JOUR DE LA FENÊTRE 3D
        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()

        # 5. Inférence YOLO26 pour la fenêtre de contrôle 2D
        results = model(img_rgb)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center, y_center = int(box.xywh[0][0]), int(box.xywh[0][1])
                
                distance = depth_frame.get_distance(x_center, y_center)
                label = model.names[int(box.cls[0])]
                
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"{label}: {distance:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Affichage du retour 2D
        cv2.imshow("Controle 2D - YOLO26", img_rgb)
        
        # Quitter proprement avec Échap
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Nettoyage complet
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()