
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image

st.set_page_config(
    page_title="Pothole & Defect Detector",
    page_icon="🚧",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="neelima03/pothole-defect-detector",
        filename="best.pt",
        repo_type="model"
    )
    return YOLO(model_path)


st.title("🚧 Pothole & Defect Detector")
st.write(
    "Upload a road image to detect potholes, "
    "view confidence scores, and estimate severity."
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting potholes..."):
        model = load_model()
        result = model.predict(
            image, conf=0.25, verbose=False
        )[0]
        annotated = Image.fromarray(
            result.plot()[..., ::-1]
        )

    height, width = result.orig_shape
    total_area = height * width
    defect_area = 0

    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        defect_area += max(0, x2 - x1) * max(0, y2 - y1)

    coverage = defect_area / total_area * 100
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Result")
        st.image(annotated, use_container_width=True)

    with col2:
        st.subheader("Analysis Report")
        st.metric("Potholes detected", count)
        st.metric("Defect coverage", f"{coverage:.2f}%")
        st.metric("Estimated severity", severity)

        if count:
            st.write("**Confidence scores**")
            for i, box in enumerate(result.boxes, start=1):
                confidence = float(box.conf[0].cpu()) * 100
                st.write(f"Pothole {i}: {confidence:.2f}%")
        else:
            st.info("No potholes detected.")

st.caption(
    "Severity is an image-area estimate, not a professional "
    "road safety assessment. Predictions may miss defects."
)
