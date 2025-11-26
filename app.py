# app.py
import os
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="Traffic Detection App", layout="wide")

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def read_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def parse_results(results):
    dets = []
    if not results:
        return dets

    r = results[0]
    boxes = r.boxes

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
    except:
        return dets

    names = r.names
    for i, b in enumerate(xyxy):
        dets.append({
            "xmin": float(b[0]), "ymin": float(b[1]),
            "xmax": float(b[2]), "ymax": float(b[3]),
            "conf": float(conf[i]),
            "cls": int(clss[i]),
            "name": names[int(clss[i])]
        })
    return dets


def draw_boxes(img, detections):
    out = img.copy()
    for d in detections:
        x1, y1 = int(d["xmin"]), int(d["ymin"])
        x2, y2 = int(d["xmax"]), int(d["ymax"])
        label = f"{d['name']} {d['conf']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return out


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.title("Model Settings")

model_path = st.sidebar.text_input("Model path (.pt)", "yolov8n.pt")
uploaded_model = st.sidebar.file_uploader("Upload YOLO model", type=["pt"])

if uploaded_model:
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = tmp_path

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
iou = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45)
max_det = st.sidebar.number_input("Max Detections", 1, 1000, 300)

st.sidebar.write("----")


# -------------------------
# Main UI
# -------------------------
st.title("🚦 Traffic Detection using YOLO")

input_type = st.selectbox("Input Type", ["Image", "Video"])
file_upload = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
run_btn = st.button("Run Detection")


# -------------------------
# Run Detection
# -------------------------
if run_btn:
    model = load_model(model_path)

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    if file_upload is None:
        st.error("Please upload a file.")
        st.stop()

    ext = file_upload.name.split(".")[-1].lower()

    # -------------------------
    # IMAGE PROCESSING
    # -------------------------
    if input_type == "Image" and ext in ("jpg", "jpeg", "png"):
        img_bytes = file_upload.read()
        img = read_image(img_bytes)

        results = model(img, conf=confidence, iou=iou, max_det=max_det)
        detections = parse_results(results)
        annotated = draw_boxes(img, detections)

        st.image(annotated, channels="BGR", caption="Annotated Image")

        if detections:
            st.dataframe(pd.DataFrame(detections))

        _, buf = cv2.imencode(".jpg", annotated)
        st.download_button("Download Annotated Image", buf.tobytes(), "annotated.jpg", "image/jpeg")

    # -------------------------
    # VIDEO PROCESSING (FFmpeg replacement)
    # -------------------------
    if input_type == "Video" and ext in ("mp4", "avi", "mov"):
        # Save uploaded video to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tfile.write(file_upload.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        out_path = os.path.join(tempfile.gettempdir(), f"annotated_{time.time()}.mp4")

        # FFmpeg-friendly OpenCV writer (Streamlit Cloud compatible)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        st.write("Processing video...")
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou, max_det=max_det)
            detections = parse_results(results)
            annotated = draw_boxes(frame, detections)
            writer.write(annotated)

            frame_count += 1
            if total_frames > 0:
                progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        writer.release()

        st.success("Video processed successfully.")
        with open(out_path, "rb") as f:
            st.video(f.read())
            st.download_button("Download Processed Video", f.read(), "annotated_video.mp4", "video/mp4")
