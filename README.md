# YOLO26 Object Detection & Pose Estimation

This repository contains code developed for real-time **thrown object detection** and **human detection**, using advanced YOLO models.

## 🤖 Key Features

* **Thrown Object Detection:** Utilizes a **custom YOLO26 model (fine-tuned by myself)** specifically trained to track and detect fast-moving objects.
* **Pose Estimation:** Integrates **YOLO Pose** for detecting and tracking people/body structures.
* **Multi-camera Support:** Includes dedicated scripts tailored for different types of sensors and cameras.

---

## 📁 Project Structure

The project is organized into specific directories based on the hardware (camera) used. 

> ⚠️ **Important:** Each camera type has its own dependency constraints. A dedicated `requirements.txt` file is provided within each subfolder because **the library versions required differ from one hardware setup to another**.

```text
yolo26_object_detection/
│
├── 📁 Intel_RealSense_D435f/      # Specific scripts for the Intel RealSense camera
│   ├── requirements.txt           # Dependencies for RealSense
│   └── ...                     
│
├── 📁 Orbbec_Femto_Bolt/          # Specific scripts for the Orbbec camera
│   ├── requirements.txt           # Dependencies for Orbbec
│   └── ...                      
│
├── 📁 Webcam/                     # Scripts for standard webcam usage
│   └── ...            
|
├── 📁 weights/                    # Fine-tuned models
│   └── alien_plushie_v3.pt  
│
└── .gitignore
```

## 🧠 Fine-Tuned YOLO26 Model

The object detection model included in this repository (located in the **weights/** folder) has been fine-tuned on a custom dataset to optimize detection accuracy for specific thrown objects that are not natively available in the COCO dataset (on which YOLO models are usually trained).
