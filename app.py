# app.py
"""
Streamlit app for YOLO-based traffic detection + simple speed estimation.

Usage:
- Place a YOLO model file (e.g., yolov8n.pt) in the repo or upload it in the app.
- Run: streamlit run app.py

Requirements (example):
streamlit
ultralytics
opencv-python-headless
numpy
pandas
pillow
moviepy
"""

import os
import io
import tempfile
import time
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
import moviepy.editor as mpy

# ---------------------------
# Helpers: caching & parsing
# ---------------------------
st.set_page_config(layout="wide", page_title="Traffic Detection (YOLO + Streamlit)")

@st.cache_resource
def load_model(model_path: str):
    """Load and cache YOLO model. model_path can be local file path or a remote model string."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        raise

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def parse_yolo_results(results) -> List[Dict]:
    """
    Given ultralytics Results (possibly list), produce list of detections for first result:
    Each detection: {xmin,ymin,xmax,ymax,conf,cls,name}
    """
    detections = []
    if not results:
        return detections
    res = results[0]  # results object for single image/frame

    # Try known shapes (ultralytics v8.x)
    try:
        boxes = res.boxes  # Boxes object
        # boxes.xyxyn or boxes.xyxy
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
        names = res.names if hasattr(res, "names") else {}
        if xyxy is not None:
            for i, b in enumerate(xyxy):
                xmin, ymin, xmax, ymax = map(float, b[:4])
                conf = float(confs[i]) if confs is not None else float(b[4]) if b.shape[0] > 4 else 1.0
                cls = int(clss[i]) if clss is not None else int(b[5]) if b.shape[0] > 5 else -1
                name = names.get(cls, str(cls))
                detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
            return detections
    except Exception:
        pass

    # Fallback: try converting to pandas or .boxes.data
    try:
        if hasattr(res, "boxes") and hasattr(res.boxes, "data"):
            data = res.boxes.data.cpu().numpy()  # each row: [x1,y1,x2,y2,confidence,class]
            names = res.names if hasattr(res, "names") else {}
            for row in data:
                xmin, ymin, xmax, ymax, conf, cls = map(float, row[:6])
                cls = int(cls)
                name = names.get(cls, str(cls))
                detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
            return detections
    except Exception:
        pass

    # As last resort: attempt attribute access
    try:
        for box in getattr(res, "boxes", []):
            coords = getattr(box, "xyxy", None)
            conf = float(getattr(box, "conf", 0.0))
            cls = int(getattr(box, "cls", -1))
            if coords is not None:
                if isinstance(coords, (list, tuple, np.ndarray)):
                    xmin, ymin, xmax, ymax = coords[0] if isinstance(coords[0], (list, tuple, np.ndarray)) else coords
                else:
                    xmin, ymin, xmax, ymax = coords
                name = getattr(res, "names", {}).get(cls, str(cls))
                detections.append({"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax), "conf": conf, "cls": cls, "name": name})
    except Exception:
        pass

    return detections

def draw_boxes_on_image(img: np.ndarray, detections: List[Dict], show_conf: bool=True) -> np.ndarray:
    """Draw rectangles and labels on BGR image and return annotated BGR image."""
    out = img.copy()
    h, w = out.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        label = f"{det.get('name',det.get('cls',''))} {det.get('conf',0):.2f}" if show_conf else f"{det.get('name',det.get('cls',''))}"
        # color by class id
        color = tuple(int(c) for c in np.random.RandomState(det.get("cls",0)).randint(0,255,3))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

# ---------------------------
# Simple tracker for speed
# ---------------------------
class SimpleTracker:
    """Very simple centroid-based tracker used for per-frame matching. Not for production."""
    def __init__(self, max_lost=3):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.lost = {}     # object_id -> lost frames
        self.max_lost = max_lost
        self.hist_positions = {}  # object_id -> list of (frame_idx, centroid)

    @staticmethod
    def centroid_from_box(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[Dict], frame_idx: int):
        centroids = [self.centroid_from_box((d["xmin"], d["ymin"], d["xmax"], d["ymax"])) for d in detections]
        if len(self.objects) == 0:
            for c in centroids:
                oid = self.next_object_id
                self.objects[oid] = c
                self.lost[oid] = 0
                self.hist_positions[oid] = [(frame_idx, c)]
                self.next_object_id += 1
            return {i: idx for i, idx in enumerate(range(self.next_object_id - len(centroids), self.next_object_id))}

        # match centroids to existing objects by nearest neighbor
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids])
        new_centroids = np.array(centroids) if centroids else np.empty((0,2))
        assigned = {}
        if new_centroids.shape[0] > 0 and len(obj_centroids) > 0:
            dists = np.linalg.norm(obj_centroids[:,None,:] - new_centroids[None,:,:], axis=2)  # shape (n_obj, n_new)
            # greedy assignment
            while True:
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                minval = dists[idx]
                if np.isinf(minval):
                    break
                obj_idx, new_idx = idx
                oid = obj_ids[obj_idx]
                assigned[new_idx] = oid
                # mark row and col as assigned by setting to inf
                dists[obj_idx,:] = np.inf
                dists[:,new_idx] = np.inf
                if np.all(np.isinf(dists)):
                    break

        # update assigned
        used_oids = set()
        for new_idx, oid in assigned.items():
            c = tuple(new_centroids[new_idx].tolist())
            self.objects[oid] = c
            self.lost[oid] = 0
            self.hist_positions[oid].append((frame_idx, c))
            used_oids.add(oid)

        # unassigned new centroids -> create new objects
        for new_idx in range(new_centroids.shape[0]):
            if new_idx not in assigned:
                oid = self.next_object_id
                c = tuple(new_centroids[new_idx].tolist())
                self.objects[oid] = c
                self.lost[oid] = 0
                self.hist_positions[oid] = [(frame_idx, c)]
                self.next_object_id += 1
                assigned[new_idx] = oid

        # increment lost for not-updated objects
        for oid in list(self.objects.keys()):
            if oid not in used_oids and oid not in assigned.values():
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    # remove
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)
                    # keep hist_positions for reporting

        # return map: detection_index -> object_id
        return assigned

def compute_speeds(tracker: SimpleTracker, meters_per_pixel: float, fps: float) -> Dict[int, float]:
    """
    Based on hist_positions, compute instantaneous speed (km/h) using last two positions for each object.
    meters_per_pixel: scale provided by user (meters per pixel)
    fps: frames per second of the video processed
    """
    speeds_kmph = {}
    for oid, hist in tracker.hist_positions.items():
        if len(hist) >= 2:
            # use last two positions
            (f1, c1), (f2, c2) = hist[-2], hist[-1]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist_pixels = np.sqrt(dx*dx + dy*dy)
            dist_m = dist_pixels * meters_per_pixel
            time_s = (f2 - f1) / fps if fps > 0 else 1.0 / fps
            if time_s == 0:
                speed_m_s = 0.0
            else:
                speed_m_s = dist_m / time_s
            speed_kmph = speed_m_s * 3.6
            speeds_kmph[oid] = float(speed_kmph)
    return speeds_kmph

# ---------------------------
# UI: Sidebar
# ---------------------------
st.sidebar.title("Model & Input")
st.sidebar.write("Load model (local path relative to repo or upload below)")

model_path_input = st.sidebar.text_input("Model path (e.g., yolov8n.pt)", value="yolov8n.pt")
uploaded_model = st.sidebar.file_uploader("Or upload a .pt model file", type=["pt"], accept_multiple_files=False)

model_path = model_path_input
if uploaded_model is not None:
    # save to temp file and use
    tmp_model_file = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_model_file, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = tmp_model_file

load_model_btn = st.sidebar.button("Load model")
if load_model_btn:
    try:
        model = load_model(model_path)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")
        model = None
else:
    # try to load quietly, show message
    try:
        model = load_model(model_path)
    except Exception:
        model = None

confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IOU threshold (for NMS)", 0.0, 1.0, 0.45, 0.01)
max_det = st.sidebar.number_input("Max detections per frame", min_value=1, max_value=1000, value=300)

st.sidebar.write("---")
st.sidebar.write("Video speed estimation (optional)")
meters_per_pixel = st.sidebar.number_input("meters_per_pixel (meters per pixel scale)", min_value=0.0, value=0.0, format="%.6f")
fps_input = st.sidebar.number_input("FPS (frames per second of the video)", min_value=1.0, value=30.0)

# ---------------------------
# Main UI
# ---------------------------
st.title("Traffic Detection App (YOLO)")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Input")
    input_type = st.selectbox("Input type", ("Image", "Video"))
    uploaded_file = st.file_uploader("Upload file", type=["jpg","jpeg","png","mp4","mov","avi"])
    run_button = st.button("Run Detection")

with col2:
    st.header("Detections / Output")
    output_placeholder = st.empty()

# ---------------------------
# Run detection flows
# ---------------------------
if run_button:
    if model is None:
        st.error("Model not loaded. Either provide a valid model path in the sidebar or upload a .pt model.")
    elif uploaded_file is None:
        st.warning("Please upload an image or video file.")
    else:
        ext = uploaded_file.name.split(".")[-1].lower()
        if input_type == "Image" and ext in ("jpg","jpeg","png"):
            image_bytes = uploaded_file.read()
            img = read_image_from_bytes(image_bytes)
            # run prediction
            try:
                results = model.predict(source=img, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
            except TypeError:
                # older/newer ultralytics wrappers: try alternative call
                results = model(img, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
            detections = parse_yolo_results(results)
            annotated = draw_boxes_on_image(img, detections)
            st.image(cv2_to_pil(annotated), caption="Annotated image", use_column_width=True)
            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df)
                # prepare download of annotated image
                buf = cv2.imencode(".jpg", annotated)[1].tobytes()
                st.download_button("Download annotated image", data=buf, file_name="annotated.jpg", mime="image/jpeg")
            else:
                st.info("No detections above threshold.")
        elif input_type == "Video" and ext in ("mp4","mov","avi"):
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or float(fps_input)
            out_path = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
            tracker = SimpleTracker()
            frame_idx = 0
            results_list = []
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            processed = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # run detection on frame (BGR)
                try:
                    # ultralytics accepts numpy images (BGR)
                    results = model.predict(source=frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
                except TypeError:
                    results = model(frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
                detections = parse_yolo_results(results)
                # tracker update
                mapping = tracker.update(detections, frame_idx)
                # compute speeds if scale provided
                speeds = {}
                if meters_per_pixel > 0 and fps > 0:
                    speeds = compute_speeds(tracker, meters_per_pixel, fps)
                # annotate frame with boxes and speed text
                annotated = draw_boxes_on_image(frame, detections)
                # add speed labels near tracked objects
                for det_idx, oid in mapping.items():
                    if oid in speeds:
                        # place text in top-left corner of the bbox
                        d = detections[det_idx]
                        x1, y1 = int(d["xmin"]), int(d["ymin"])
                        speed_text = f"{speeds[oid]:.1f} km/h"
                        cv2.putText(annotated, speed_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                writer.write(annotated)
                frame_idx += 1
                processed += 1
                if total_frames > 0:
                    progress_bar.progress(min(processed/total_frames, 1.0))
            cap.release()
            writer.release()
            progress_bar.empty()
            st.success("Video processing finished.")
            # show video
            with open(out_path, "rb") as f:
                data = f.read()
                st.video(data)
                st.download_button("Download annotated video", data=data, file_name="annotated_video.mp4", mime="video/mp4")
        else:
            st.error("Uploaded file type does not match selected input type. Choose correct file or input type.")

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.markdown(
    """
    **Notes & tips**
    - Provide a model file (yolov8n.pt or your trained model) in the repo or upload it in the sidebar.
    - For speed estimation: you must provide a realistic `meters_per_pixel` calibration (how many meters correspond to one pixel in your frames) and correct FPS.
    - The tracker used here is a minimal centroid-matcher for demonstration only. For robust multi-object tracking consider SORT / DeepSORT / ByteTrack.
    - On Streamlit Cloud, include required packages in `requirements.txt`.
    """
)
