# YOLO26 Object Detection & Pose Estimation

Ce dépôt contient un code développé pour la **détection d'objets lancés** ainsi que la **détection de personnes** en temps réel, en utilisant des modèles YOLO avancés.

## 🤖 Fonctionnalités principales

* **Détection d'objets lancés :** Utilisation d'un modèle **YOLO26 personnalisé (fine-tuné par mes soins)** spécialement entraîné pour suivre et détecter des objets en mouvement rapide.
* **Estimation de pose (Pose Estimation) :** Intégration de **YOLO Pose** pour la détection et le suivi des personnes/structures corporelles.
* **Multi-caméras :** Support et scripts adaptés pour différents types de capteurs et caméras.

---

## 📁 Structure du projet

Le projet est divisé en dossiers spécifiques selon le matériel (caméra) utilisé. 

> ⚠️ **Important :** Chaque type de caméra possède ses propres contraintes de dépendances. Un fichier `requirements.txt` dédié est donc présent dans chaque sous-dossier car **les versions des bibliothèques à utiliser diffèrent d'un matériel à l'autre**.

```text
yolo26_object_detection/
│
├── 📁 Intel_RealSense_D435f/      # Scripts spécifiques à la caméra Intel RealSense
│   ├── requirements.txt           # Dépendances pour RealSense
│   └── ...                        # YOLO26_for_RealSense.py, point_cloud.py, etc.
│
├── 📁 Orbbec_Femto_Bolt/          # Scripts spécifiques à la caméra Orbbec
│   ├── requirements.txt           # Dépendances pour Orbbec
│   └── ...                        # launch_orbbec.py, yolo26_pose_orbbec.py, etc.
│
├── 📁 Webcam/                     # Scripts pour l'utilisation d'une webcam standard
│   └── ...                        # segmentation.py, test_yolo.py
│
└── .gitignore
```

## 🧠 Modèle YOLO26 Fine-Tuné

Le modèle de détection d'objets inclus dans ce dépôt a été **fine-tuné** sur un jeu de données personnalisé afin d'optimiser la précision de la détection sur des objets lancés **spécifiques**, non présents dans le dataset COCO sur lequel les modèles YOLO sont usuellement entraînés.

## 🚧 Work In Progress (WIP)

* **Guide de téléchargement à venir :** Les fichiers du modèle étant volumineux, un guide détaillé expliquant comment et où télécharger les poids du modèle fine-tuné sera très prochainement ajouté à cette section.