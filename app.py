import os
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import ffmpeg  # moviepy removed — ffmpeg works on Streamlit Cloud

st.set_page_config(page_title="Traffic Detection", layout="wide")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def read_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def draw_boxes(img, detections):
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, [d["xmin"], d["ymin"], d["xmax"], d["ymax"]])
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return out

def parse_results(res):
    dets = []
    r = res[0]

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    for i, b in enumerate(boxes):
        dets.append({
            "xmin": float(b[0]),
            "ymin": float(b[1]),
            "xmax": float(b[2]),
            "ymax": float(b[3]),
            "conf": float(confs[i]),
            "cls": int(clss[i]),
            "name": names[int(clss[i])]
        })
    return dets

# ---------------- Sidebar ----------------
st.sidebar.title("YOLO Settings")
model_file = st.sidebar.text_input("Model path", "yolov8n.pt")
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)
iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.45)

st.title("Traffic Detection App")

uploaded = st.file_uploader("Upload image/video", type=["jpg","jpeg","png","mp4"])
run_btn = st.button("Run Detection")

if uploaded and run_btn:

    model = load_model(model_file)
    ext = uploaded.name.split(".")[-1].lower()

    # ------------- IMAGE -------------
    if ext in ["jpg", "jpeg", "png"]:
        img = read_image(uploaded.read())
        results = model.predict(img, conf=confidence, iou=iou)
        detections = parse_results(results)
        annotated = draw_boxes(img, detections)

        st.image(annotated, channels="BGR", use_column_width=True)

        df = pd.DataFrame(detections)
        st.dataframe(df)

        _, buffer = cv2.imencode(".jpg", annotated)
        st.download_button("Download Annotated Image",
                           buffer.tobytes(), "annotated.jpg")

    # ------------- VIDEO -------------
    elif ext == "mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{w}x{h}", r=fps)
            .output(output_path, pix_fmt='yuv420p', vcodec='libx264')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence, iou=iou)
            detections = parse_results(results)
            annotated = draw_boxes(frame, detections)

            process.stdin.write(annotated.tobytes())

            idx += 1
            progress.progress(idx / frame_count)

        cap.release()
        process.stdin.close()
        process.wait()

        st.video(open(output_path, "rb").read())
        st.download_button("Download Annotated Video",
                           open(output_path, "rb").read(),
                           "annotated.mp4",
                           mime="video/mp4")
