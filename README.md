# 🚧 Pothole & Defect Detector

An AI-powered computer vision application that detects potholes in road images using YOLOv8. The application displays bounding boxes, confidence scores, defect coverage, and an estimated severity level.

## ✨ Features

- Upload road images in JPG, JPEG, PNG, or WebP format.
- Detect potholes using a trained YOLOv8 model.
- Display annotated images with detection boxes.
- Show confidence scores for detected potholes.
- Estimate severity based on the percentage of image area covered by predicted pothole boxes.

## 🛠️ Tech Stack

- Python
- YOLOv8 and Ultralytics
- Streamlit
- Pillow
- Hugging Face Hub

## ⚙️ Severity Estimation

| Image area covered | Estimated severity |
|---|---|
| No potholes detected | NONE |
| Up to 3% | LOW |
| Above 3% to 10% | MEDIUM |
| Above 10% to 20% | HIGH |
| Above 20% | CRITICAL |

*Severity is an image-based estimate, not a professional road safety assessment.*

## 🚀 Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/AchuriNeelima/Pothole-Defect-Detector.git
   cd Pothole-Defect-Detector
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

The application downloads the trained `best.pt` model from the Hugging Face model repository when needed.

## 🤖 Trained Model

[View the Pothole Defect Detector model on Hugging Face](https://huggingface.co/neelima03/pothole-defect-detector)

## 📊 Model Evaluation

The trained model achieved the following validation metrics:

- Precision: 79.0%
- Recall: 67.3%
- mAP@50: 77
