🚀 Object Detection Using YOLOv8 — Hackathon Project

This repository contains the complete implementation of a multi-class indoor object detection system built using YOLOv8.
The model is trained on a custom dataset containing images with different lighting conditions, clutter levels, and indoor environments such as hallways and rooms.

📁 Project Structure
.
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── runs/
│   └── detect/
│       ├── train/
│       └── predict/
├── notebooks/
│   └── object_detection_training.ipynb
└── README.md

🎯 Project Goal

To build a robust YOLO-based detection model capable of identifying multiple object classes in indoor environments such as:

Cluttered rooms

Hallways

Images with bright or low lighting

Mixed-object scenes

🚀 Features

✔ Custom dataset preparation
✔ YOLOv8 training pipeline on Google Colab
✔ Evaluation using mAP, F1 curve, PR curve
✔ Confusion matrix analysis
✔ Inference/testing on unseen images
✔ Final predictions with bounding boxes
✔ Export-ready results for Hackathon submission

📦 Technologies Used

Python

YOLOv8 (Ultralytics)

Google Colab

PyTorch

OpenCV

NumPy

PIL

📊 Model Performance
🔹 Confusion Matrix

Your model shows strong prediction accuracy for major classes, with deep-blue diagonals indicating correct predictions.

🔹 F1-Confidence Curve

Best F1 score: 0.66 at confidence 0.302
Model performs best around 0.30 threshold.

🔹 mAP Scores

•	mAP50: 0.6417306477399879
•	mAP50-95: 0.49801668974648133

🔹 Loss Curves

Box, class, and objectness loss all decrease steadily

No signs of overfitting

Validation loss is stable

🖼 Example Prediction

The model successfully detects objects even in complex scenes:

Bright light

Cluttered background

Multiple objects

Rotated camera view

Sample output image from runs/detect/predict/.

📚 Training Instructions
1️⃣ Clone Repository

2️⃣ Install Dependencies
pip install ultralytics

3️⃣ Train Model
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)

🧪 Inference / Testing
Run prediction on test images
model.predict(
    source="dataset/test",
    save=True
)


Results saved in:

runs/detect/predict/

📈 Evaluation Visuals

YOLO auto-generates the following:

confusion_matrix.png

BoxF1_curve.png

PR_curve.png

results.png

labels.jpg

val_batch0.jpg

All these are located in:

/runs/detect/train/

📝 Conclusion

This project successfully builds a smart detection system for real-life indoor scenarios using YOLOv8.
The model performs efficiently across different lighting conditions and clutter levels, achieving strong mAP and F1 scores.

Team Nmae- SuperVisionAI
