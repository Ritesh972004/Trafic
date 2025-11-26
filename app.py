# app.py
"""
Streamlit app for YOLO-based traffic detection + speed estimation (uses MoviePy).
Save this file to your repo and include requirements.txt (provided below).
Run locally: `streamlit run app.py`
"""

import os
import io
import tempfile
import time
from typing import List, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
import moviepy.editor as mpy  # ensure moviepy is installed via requirements.txt

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Traffic Detection (YOLO + Streamlit)", layout="wide")

@st.cache_resource
def load_model_cached(model_path: str):
    """Load and cache a YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def parse_yolo_results(results) -> List[Dict]:
    """
    Parse ultralytics YOLO results for a single frame/image.
    Returns list of dicts: xmin,ymin,xmax,ymax,conf,cls,name
    """
    detections = []
    if not results:
        return detections
    res = results[0]
    try:
        boxes = res.boxes
        # boxes.xyxy and others may be tensors
        xyxy = None
        confs = None
        clss = None
        if hasattr(boxes, "xyxy"):
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = np.array(boxes.xyxy)
        if hasattr(boxes, "conf"):
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = np.array(boxes.conf)
        if hasattr(boxes, "cls"):
            try:
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                clss = np.array(boxes.cls).astype(int)
        names = getattr(res, "names", {})

        if xyxy is not None:
            for i, b in enumerate(xyxy):
                xmin, ymin, xmax, ymax = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                conf = float(confs[i]) if confs is not None and len(confs) > i else (float(b[4]) if len(b) > 4 else 1.0)
                cls = int(clss[i]) if clss is not None and len(clss) > i else (int(b[5]) if len(b) > 5 else -1)
                name = names.get(cls, str(cls))
                detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
            return detections
    except Exception:
        # fallback path below
        pass

    # fallback: try res.boxes.data
    try:
        data = getattr(res.boxes, "data", None)
        if data is not None:
            arr = data.cpu().numpy()
            names = getattr(res, "names", {})
            for row in arr:
                xmin, ymin, xmax, ymax, conf, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5])
                name = names.get(cls, str(cls))
                detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
            return detections
    except Exception:
        pass

    # last resort: iterate boxes attribute
    try:
        for box in getattr(res, "boxes", []):
            coords = getattr(box, "xyxy", None)
            conf = float(getattr(box, "conf", 0.0))
            cls = int(getattr(box, "cls", -1))
            if coords is not None:
                if hasattr(coords, "__len__") and len(coords) >= 4:
                    xmin, ymin, xmax, ymax = map(float, coords[:4])
                else:
                    xmin, ymin, xmax, ymax = coords
                name = getattr(res, "names", {}).get(cls, str(cls))
                detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
    except Exception:
        pass

    return detections

def draw_boxes_on_image(img: np.ndarray, detections: List[Dict], show_conf=True) -> np.ndarray:
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        label = f"{det.get('name', det.get('cls',''))} {det.get('conf',0):.2f}" if show_conf else str(det.get('name', det.get('cls','')))
        color = tuple(int(c) for c in np.random.RandomState(det.get("cls",0)).randint(0,255,3))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

# ---------------------------
# Minimal tracker to annotate speeds
# (simple centroid tracker for demo — replace with SORT/ByteTrack for production)
# ---------------------------
class SimpleTracker:
    def __init__(self, max_lost=5):
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.lost = {}     # id -> lost count
        self.hist = {}     # id -> list of (frame_idx, centroid)
        self.max_lost = max_lost

    @staticmethod
    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2)/2.0, (y1 + y2)/2.0)

    def update(self, detections, frame_idx):
        centroids = [self.centroid((d["xmin"], d["ymin"], d["xmax"], d["ymax"])) for d in detections]
        if not self.objects:
            # create new objects
            mapping = {}
            for i, c in enumerate(centroids):
                oid = self.next_id
                self.objects[oid] = c
                self.lost[oid] = 0
                self.hist[oid] = [(frame_idx, c)]
                mapping[i] = oid
                self.next_id += 1
            return mapping

        # match by nearest neighbor (greedy)
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids])
        new_centroids = np.array(centroids) if centroids else np.empty((0,2))
        mapping = {}
        if new_centroids.shape[0] > 0 and obj_centroids.shape[0] > 0:
            dists = np.linalg.norm(obj_centroids[:,None,:] - new_centroids[None,:,:], axis=2)
            # greedy assignment
            while True:
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                minval = dists[idx]
                if np.isinf(minval):
                    break
                obj_idx, new_idx = idx
                oid = obj_ids[obj_idx]
                mapping[new_idx] = oid
                # mark row/col
                dists[obj_idx,:] = np.inf
                dists[:,new_idx] = np.inf
                if np.all(np.isinf(dists)):
                    break

        # update
        used_oids = set()
        for new_idx, oid in mapping.items():
            c = tuple(new_centroids[new_idx].tolist())
            self.objects[oid] = c
            self.lost[oid] = 0
            self.hist.setdefault(oid, []).append((frame_idx, c))
            used_oids.add(oid)

        # new ones
        for new_idx in range(new_centroids.shape[0]):
            if new_idx not in mapping:
                oid = self.next_id
                c = tuple(new_centroids[new_idx].tolist())
                self.objects[oid] = c
                self.lost[oid] = 0
                self.hist[oid] = [(frame_idx, c)]
                mapping[new_idx] = oid
                self.next_id += 1

        # increment lost for unmatched existing
        for oid in list(self.objects.keys()):
            if oid not in used_oids and oid not in mapping.values():
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)
        return mapping

def compute_speeds_kmph(tracker: SimpleTracker, meters_per_pixel: float, fps: float):
    speeds = {}
    for oid, hist in tracker.hist.items():
        if len(hist) >= 2:
            (_, c1), (_, c2) = hist[-2], hist[-1]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist_px = np.sqrt(dx*dx + dy*dy)
            dist_m = dist_px * meters_per_pixel
            delta_frames = hist[-1][0] - hist[-2][0]
            time_s = delta_frames / fps if fps > 0 else 1.0
            speed_m_s = dist_m / time_s if time_s > 0 else 0.0
            speeds[oid] = float(speed_m_s * 3.6)
    return speeds

# ---------------------------
# UI: Sidebar
# ---------------------------
st.sidebar.title("Model & Input")
st.sidebar.write("Load a YOLO model (.pt) or provide path to model in the repo.")

model_input = st.sidebar.text_input("Model path (e.g., yolov8n.pt)", value="yolov8n.pt")
uploaded_model = st.sidebar.file_uploader("Upload a YOLO .pt model (optional)", type=["pt"])

if uploaded_model:
    tmp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = tmp_model_path
else:
    model_path = model_input

load_btn = st.sidebar.button("Load model")
if load_btn:
    try:
        model = load_model_cached(model_path)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        model = None
else:
    # try to pre-load quietly (if file exists)
    try:
        model = load_model_cached(model_path)
    except Exception:
        model = None

confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IOU threshold (NMS)", 0.0, 1.0, 0.45, 0.01)
max_det = st.sidebar.number_input("Max detections per image", min_value=1, max_value=1000, value=300)

st.sidebar.write("---")
st.sidebar.write("Video speed estimation (optional)")
meters_per_pixel = st.sidebar.number_input("meters_per_pixel (m per pixel)", min_value=0.0, value=0.0, format="%.6f")
fps_input = st.sidebar.number_input("Video FPS (if unknown)", min_value=1.0, value=30.0)

# ---------------------------
# Main UI
# ---------------------------
st.title("Traffic Detection & Speed Estimation (YOLO)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    input_mode = st.selectbox("Input type", ("Image", "Video"))
    uploaded_file = st.file_uploader("Upload image/video", type=["jpg","jpeg","png","mp4","mov","avi"])
    run_button = st.button("Run Detection")

with col2:
    st.header("Output")
    output_area = st.empty()

# ---------------------------
# Run inference and present results
# ---------------------------
if run_button:
    if model is None:
        st.error("Model not loaded. Provide a valid model path or upload one in the sidebar.")
    elif uploaded_file is None:
        st.warning("Please upload a file.")
    else:
        ext = uploaded_file.name.split(".")[-1].lower()
        if input_mode == "Image" and ext in ("jpg", "jpeg", "png"):
            image_bytes = uploaded_file.read()
            img = read_image_from_bytes(image_bytes)
            try:
                results = model.predict(source=img, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
            except TypeError:
                results = model(img, imgsz=640, conf=confidence, iou=iou, max_det=max_det)

            detections = parse_yolo_results(results)
            annotated = draw_boxes_on_image(img, detections)
            st.image(cv2_to_pil(annotated), caption="Annotated image", use_column_width=True)
            if detections:
                st.dataframe(pd.DataFrame(detections))
                # download annotated image
                _, buf = cv2.imencode(".jpg", annotated)
                st.download_button("Download annotated image", data=buf.tobytes(), file_name="annotated.jpg", mime="image/jpeg")
            else:
                st.info("No detections above threshold.")

        elif input_mode == "Video" and ext in ("mp4", "mov", "avi"):
            # write uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()

            # use OpenCV to read frames, run detection, write annotated frames
            cap = cv2.VideoCapture(tfile.name)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or float(fps_input)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            out_temp = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_temp, fourcc, fps, (w, h))

            tracker = SimpleTracker()
            frame_idx = 0
            processed = 0
            progress = st.progress(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    results = model.predict(source=frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)
                except TypeError:
                    results = model(frame, imgsz=640, conf=confidence, iou=iou, max_det=max_det)

                detections = parse_yolo_results(results)
                mapping = tracker.update(detections, frame_idx)

                speeds = {}
                if meters_per_pixel > 0 and fps > 0:
                    speeds = compute_speeds_kmph(tracker, meters_per_pixel, fps)

                annotated = draw_boxes_on_image(frame, detections)
                for det_idx, oid in mapping.items():
                    if oid in speeds:
                        d = detections[det_idx]
                        x1, y1 = int(d["xmin"]), int(d["ymin"])
                        cv2.putText(annotated, f"{speeds[oid]:.1f} km/h", (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                writer.write(annotated)
                frame_idx += 1
                processed += 1
                if total_frames > 0:
                    progress.progress(min(processed / total_frames, 1.0))

            cap.release()
            writer.release()
            progress.empty()

            # use moviepy to re-encode or ensure proper metadata (optional)
            try:
                clip = mpy.VideoFileClip(out_temp)
                # optionally short-circuit heavy re-encoding; here we just ensure it's readable
                final_out = os.path.join(tempfile.gettempdir(), f"annotated_final_{int(time.time())}.mp4")
                clip.write_videofile(final_out, codec="libx264", audio=False, verbose=False, logger=None)
                clip.close()
            except Exception:
                # if moviepy fails for any reason, fallback to out_temp
                final_out = out_temp

            st.success("Video processing completed.")
            # show video in Streamlit
            with open(final_out, "rb") as f:
                vbytes = f.read()
                st.video(vbytes)
                st.download_button("Download annotated video", data=vbytes, file_name="annotated_video.mp4", mime="video/mp4")

        else:
            st.error("Uploaded file type does not match the selected input type. Make sure to upload a valid image or video file.")

st.markdown("---")
st.markdown(
    """
    **Notes**
    - Provide a YOLO model in repo or upload via the sidebar (e.g., yolov8n.pt).
    - For speed estimation you must specify `meters_per_pixel` and correct FPS; otherwise speed will be meaningless.
    - The tracker used here is a simple centroid tracker; for production use SORT/DeepSORT/ByteTrack.
    - On Streamlit Cloud ensure `requirements.txt` contains the packages (moviepy included).
    """
)
