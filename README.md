# Pothole & Defect Detector

## Project Overview
Pothole & Defect Detector is an AI-powered computer vision application that detects potholes in road images. It displays bounding boxes, confidence scores, estimated defect coverage, and severity levels.

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- Gradio
- Pillow

## Model Training
The model was fine-tuned from the pretrained YOLOv8n model using a labeled pothole dataset.

### Validation Metrics
| Metric | Score |
|---|---:|
| Precision | 79.0% |
| Recall | 67.3% |
| mAP@50 | 77.1% |
| mAP@50–95 | 48.8% |

These metrics are from validation on 133 images containing 330 labeled instances.

## Severity Classification
- **NONE:** No potholes detected
- **LOW:** Coverage up to 3%
- **MEDIUM:** Coverage above 3% and up to 10%
- **HIGH:** Coverage above 10% and up to 20%
- **CRITICAL:** Coverage above 20%

## How It Works
1. Upload a road image.
2. The YOLOv8 model detects potholes.
3. Bounding boxes and confidence scores are displayed.
4. The application estimates defect coverage and assigns a severity level.

## Limitations
- Detection accuracy depends on image quality, lighting, and pothole appearance.
- Some potholes may be missed or incorrectly detected.
- Bounding-box coverage is an estimate and is not a direct measurement of damaged road surface.
- This application is a prototype and should not replace professional road inspections.

## How to Run Locally
Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Ensure that `best.pt` is present in the same folder as `app.py`.

## Future Improvements
- Improve detection using more diverse road images.
- Add more test examples and failure-case analysis.
- Deploy the application using Hugging Face Spaces.