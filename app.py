
import gradio as gr
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)


def detect_potholes(image):
    if image is None:
        return None, "Please upload a road image."

    result = model.predict(image, conf=0.25, verbose=False)[0]
    annotated_image = Image.fromarray(result.plot()[..., ::-1])

    height, width = result.orig_shape
    total_image_area = height * width
    total_defect_area = 0

    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        total_defect_area += (x2 - x1) * (y2 - y1)

    coverage = total_defect_area / total_image_area * 100
    count = len(result.boxes)

    if count == 0:
        severity = "NONE"
    elif coverage <= 3:
        severity = "LOW"
    elif coverage <= 10:
        severity = "MEDIUM"
    elif coverage <= 20:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    report = (
        f"Detected potholes: {count}\n"
        f"Defect coverage: {coverage:.2f}%\n"
        f"Severity: {severity}\n\n"
    )

    if count:
        report += "Confidence scores:\n"
        for i, box in enumerate(result.boxes, start=1):
            confidence = float(box.conf[0].cpu()) * 100
            report += f"Pothole {i}: {confidence:.2f}%\n"
    else:
        report += "No potholes detected."

    return annotated_image, report


demo = gr.Interface(
    fn=detect_potholes,
    inputs=gr.Image(type="pil", label="Upload a road image"),
    outputs=[
        gr.Image(type="pil", label="Detection result"),
        gr.Textbox(label="Pothole analysis", lines=8),
    ],
    title="Pothole & Defect Detector",
    description=(
        "Upload a road image to detect potholes, "
        "view confidence scores, and estimate severity."
    ),
)

if __name__ == "__main__":
    demo.launch()
