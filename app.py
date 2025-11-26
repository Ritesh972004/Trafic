# app.py
"""
Streamlit YOLO app (image + video). Uses OpenCV VideoWriter only (no moviepy/ffmpeg-python).
Put your YOLO .pt model in the repo (e.g., yolov8n.pt) or upload via the sidebar.
"""

import os
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Detection", layout="wide")

@st.cache_resource
def load_model(model_path):
    """Load and cache a YOLO model."""
    return YOLO(model_path)

def read_image_bytes(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def draw_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1,y1,x2,y2 = map(int, [d["xmin"], d["ymin"], d["xmax"], d["ymax"]])
        label = f"{d.get('name', d.get('cls', ''))} {d.get('conf',0):.2f}"
        color = tuple(int(c) for c in np.random.RandomState(d.get("cls",0)).randint(0,255,3))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def parse_yolo_results(results) -> list:
    dets = []
    if not results:
        return dets
    r = results[0]
    # Most ultralytics results expose r.boxes.xyxy, r.boxes.conf, r.boxes.cls
    try:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = getattr(r, "names", {})
        for i, b in enumerate(xyxy):
            dets.append({
                "xmin": float(b[0]), "ymin": float(b[1]), "xmax": float(b[2]), "ymax": float(b[3]),
                "conf": float(confs[i]), "cls": int(clss[i]), "name": names.get(int(clss[i]), str(int(clss[i])))
            })
    except Exception:
        # Fallback: try boxes.data
        try:
            data = getattr(r.boxes, "data", None)
            if data is not None:
                arr = data.cpu().numpy()
                names = getattr(r, "names", {})
                for row in arr:
                    xmin,ymin,xmax,ymax,conf,cls = map(float, row[:6])
                    dets.append({"xmin": xmin,"ymin": ymin,"xmax": xmax,"ymax": ymax,"conf": conf,"cls": int(cls),"name": names.get(int(cls), str(int(cls)))})
        except Exception:
            pass
    return dets

def try_create_writer(path, fourcc_str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        return writer
    # else release and return None
    try:
        writer.release()
    except Exception:
        pass
    return None

# UI
st.sidebar.title("Model & Settings")
model_path_input = st.sidebar.text_input("Model path (in repo)", value="yolov8n.pt")
uploaded_model = st.sidebar.file_uploader("Or upload model (.pt)", type=["pt"])
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IOU threshold", 0.0, 1.0, 0.45, 0.01)
max_det = st.sidebar.number_input("Max detections", min_value=1, max_value=1000, value=300)
st.sidebar.write("Video upload max size is controlled by Streamlit settings.")

# load model path
if uploaded_model is not None:
    tmp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = tmp_model_path
else:
    model_path = model_path_input

# attempt to load model (cached)
try:
    model = load_model(model_path)
except Exception as e:
    model = None
    st.sidebar.error(f"Could not load model: {e}")

st.title("Traffic Detection (YOLO) — Image & Video")

col1, col2 = st.columns([1,2])
with col1:
    input_type = st.selectbox("Input type", ("Image", "Video"))
    uploaded_file = st.file_uploader("Upload file", type=["jpg","jpeg","png","mp4","mov","avi"])
    run_btn = st.button("Run")
with col2:
    out_placeholder = st.empty()

if run_btn:
    if model is None:
        st.error("Model not loaded. Provide a valid .pt model in the sidebar.")
    elif uploaded_file is None:
        st.error("Upload an image or video first.")
    else:
        ext = uploaded_file.name.split(".")[-1].lower()

        # IMAGE branch
        if input_type == "Image" and ext in ("jpg","jpeg","png"):
            raw = uploaded_file.read()
            img = read_image_bytes(raw)
            results = model.predict(source=img, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
            detections = parse_yolo_results(results)
            annotated = draw_boxes(img, detections)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            if detections:
                st.dataframe(pd.DataFrame(detections))
                _, buf = cv2.imencode(".jpg", annotated)
                st.download_button("Download Annotated Image", data=buf.tobytes(), file_name="annotated.jpg", mime="image/jpeg")
            else:
                st.info("No detections above threshold.")

        # VIDEO branch
        elif input_type == "Video" and ext in ("mp4","mov","avi"):
            # Save uploaded video to temp file
            tpath = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}.{ext}")
            with open(tpath, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(tpath)
            if not cap.isOpened():
                st.error("Failed to open uploaded video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

                # prepare output path and try common codecs:
                out_path_mp4 = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
                writer = try_create_writer(out_path_mp4, "mp4v", fps, (w,h))

                if writer is None:
                    # try XVID AVI fallback
                    out_path_avi = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.avi")
                    writer = try_create_writer(out_path_avi, "XVID", fps, (w,h))
                    out_is_avi = True
                    out_path = out_path_avi
                else:
                    out_is_avi = False
                    out_path = out_path_mp4

                if writer is None:
                    st.error("Could not create a video writer with available codecs. Try uploading a different video or run locally with codecs installed.")
                else:
                    progress = st.progress(0)
                    idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # run detection (frame is BGR)
                        try:
                            results = model.predict(source=frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
                        except TypeError:
                            results = model(frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
                        detections = parse_yolo_results(results)
                        annotated = draw_boxes(frame, detections)
                        writer.write(annotated)
                        idx += 1
                        if total_frames:
                            progress.progress(min(idx/total_frames, 1.0))
                    writer.release()
                    cap.release()
                    progress.empty()

                    # show video
                    try:
                        with open(out_path, "rb") as f:
                            vbytes = f.read()
                        st.video(vbytes)
                        st.download_button("Download Annotated Video", data=vbytes, file_name=os.path.basename(out_path), mime="video/mp4" if not out_is_avi else "video/x-msvideo")
                    except Exception as e:
                        st.error(f"Could not show/download output video: {e}")

        else:
            st.error("Selected input type and uploaded file type do not match. Make sure to choose Image or Video and upload the correct file format.")
