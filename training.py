from ultralytics import YOLO

model = YOLO('yolo26n.pt')

# 2. Lancer le fine-tuning sur ton alien vert
results = model.train(
    data=f"Alien-Plushie-1/data.yaml", 
    epochs=50,                            # 50 => suffisant pour commencer
    imgsz=640,                            # Mettre la même size que dans config Roboflow
    batch=16,                             
    device=0,                             
    name='yolo26_alien_plushie'           # Dossier de sortie
)