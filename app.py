# app.py
"""
Streamlit app that loads a model from repo path (default: models/model.pkl)
and runs inference on uploaded images.
- Supports: YOLO .pt, pickled Python objects (.pkl), and torch saved modules (.pth/.pt).
- If you place your model in the repo (for example: models/model.pkl), set that path in the sidebar.
- Designed to run on Streamlit Cloud (no moviepy).
"""
import os
import io
import tempfile
import pickle
import streamlit as st
from typing import Any, List, Dict
import numpy as np
from PIL import Image
import cv2

# Try to import libraries (some might be optional)
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

st.set_page_config(page_title="Repo model → Streamlit", layout="wide")

# --------------------------
# Utilities
# --------------------------
def read_image_bytes(bytestr: bytes):
    arr = np.frombuffer(bytestr, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def bgr_to_rgb(img_bgr: np.ndarray):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb: np.ndarray):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def pil_from_bgr(img_bgr: np.ndarray):
    return Image.fromarray(bgr_to_rgb(img_bgr))

def preprocess_for_generic_model(img_bgr: np.ndarray, target_size=(224,224)):
    """Convert BGR image to normalized float32 RGB array shape (1,H,W,C) for generic models."""
    img_rgb = bgr_to_rgb(img_bgr)
    img_resized = cv2.resize(img_rgb, target_size)
    arr = img_resized.astype(np.float32) / 255.0
    # Return batch dimension first
    return np.expand_dims(arr, 0)  # shape (1,H,W,3)

# --------------------------
# Model loader (multi-format)
# --------------------------
@st.cache_resource
def load_model_from_path(path: str) -> Dict[str, Any]:
    """
    Load model based on file extension. Returns a dict:
    {
      "type": "yolo"|"pickle"|"torch"|"unknown",
      "model": object,
      "meta": optional
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt" or ext == ".pth":
        # First try Ultralytics YOLO if available and file looks like YOLO weight
        if _HAS_YOLO:
            try:
                m = YOLO(path)
                return {"type": "yolo", "model": m}
            except Exception:
                # not a YOLO model or ultralytics failed, try torch
                pass
        if _HAS_TORCH:
            try:
                m = torch.load(path, map_location="cpu")
                # If it's a Module, set eval
                if isinstance(m, torch.nn.Module):
                    m.eval()
                return {"type": "torch", "model": m}
            except Exception as e:
                # fallback: treat as generic file
                raise RuntimeError(f"Failed to load .pt/.pth as YOLO or Torch: {e}")
        raise RuntimeError("No loader available for .pt/.pth — install ultralytics or torch in requirements.")
    elif ext == ".pkl" or ext == ".pickle":
        # generic pickle load
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return {"type": "pickle", "model": obj}
    else:
        # try pickle as fallback
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
                return {"type": "pickle", "model": obj}
            except Exception:
                raise RuntimeError(f"Unknown/unsupported model extension: {ext}")

# --------------------------
# UI: Sidebar (model path)
# --------------------------
st.sidebar.title("Model path (in repo)")
st.sidebar.markdown("Place your model file in the repository (for example `models/model.pkl`) and enter that path here.")
model_path = st.sidebar.text_input("Model path (relative to repo)", value="models/model.pkl")

st.sidebar.write("---")
st.sidebar.write("Model type autodetects (.pt YOLO, .pkl pickle, .pth torch).")
st.sidebar.write("If your pickled model expects a different input format, see the app logs (printed below).")

load_btn = st.sidebar.button("Load model now")

# --------------------------
# Main area
# --------------------------
st.title("Load model from repo and run inference")
st.write("Upload an image and press **Run**. The app will try to apply your model automatically.")

# keep model state in session
if "model_info" not in st.session_state:
    st.session_state.model_info = None
    st.session_state.model_path = None
    st.session_state.msg = ""

if load_btn or (st.session_state.model_path != model_path and os.path.exists(model_path)):
    # Attempt to load
    try:
        st.session_state.model_info = load_model_from_path(model_path)
        st.session_state.model_path = model_path
        st.session_state.msg = f"Loaded model from {model_path} as type: {st.session_state.model_info['type']}"
        st.success(st.session_state.msg)
    except Exception as e:
        st.session_state.model_info = None
        st.session_state.msg = f"Failed to load model: {e}"
        st.error(st.session_state.msg)

if st.session_state.model_info:
    st.caption(st.session_state.msg)

# --------------------------
# Inference controls
# --------------------------
uploaded_file = st.file_uploader("Upload image for inference", type=["jpg","jpeg","png"])
run_btn2 = st.button("Run Inference")

if run_btn2:
    if st.session_state.model_info is None:
        st.error("No model loaded. Place model in repo and click 'Load model now'.")
        st.stop()
    if uploaded_file is None:
        st.error("Please upload an image.")
        st.stop()

    # read image
    img_bytes = uploaded_file.read()
    img_bgr = read_image_bytes(img_bytes)  # BGR numpy array
    h, w = img_bgr.shape[:2]

    model_info = st.session_state.model_info
    mtype = model_info["type"]
    model_obj = model_info["model"]

    st.write(f"Model type: **{mtype}** — attempting inference...")

    # Branch by type
    if mtype == "yolo":
        if not _HAS_YOLO:
            st.error("Ultralytics YOLO is not installed in environment.")
            st.stop()
        # run YOLO on numpy image (OpenCV BGR)
        try:
            results = model_obj.predict(source=img_bgr, imgsz=640)  # return list-like results
        except TypeError:
            # compatibility fallback
            results = model_obj(img_bgr, imgsz=640)
        # parse and show boxes (minimal)
        try:
            res0 = results[0]
            boxes = getattr(res0, "boxes", None)
            detections = []
            if boxes is not None:
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback if tensors not present
                    xyxy = np.array(boxes.xyxy)
                    confs = np.array(boxes.conf)
                    clss = np.array(boxes.cls).astype(int)
                names = getattr(res0, "names", {})
                for i, b in enumerate(xyxy):
                    xmin, ymin, xmax, ymax = map(float, b[:4])
                    conf = float(confs[i]) if len(confs) > i else 1.0
                    cls = int(clss[i]) if len(clss) > i else -1
                    name = names.get(cls, str(cls))
                    detections.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "conf": conf, "cls": cls, "name": name})
            # draw boxes simple
            out = img_bgr.copy()
            for d in detections:
                x1,y1,x2,y2 = int(d["xmin"]), int(d["ymin"]), int(d["xmax"]), int(d["ymax"])
                cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(out, f"{d['name']}:{d['conf']:.2f}", (x1, max(y1-6,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            st.image(pil_from_bgr(out), caption="YOLO annotated")
            if detections:
                st.dataframe(detections)
            else:
                st.info("No detections found by YOLO.")
        except Exception as e:
            st.error(f"YOLO inference failed: {e}")

    elif mtype == "torch":
        # try to use a PyTorch module (best-effort)
        if not _HAS_TORCH:
            st.error("PyTorch not installed in environment.")
            st.stop()
        try:
            model_obj.eval()
            # preprocess: convert to RGB float, resize to 224x224, transpose to (C,H,W) if model expects that
            inp = preprocess_for_generic_model(img_bgr, target_size=(224,224))  # (1,H,W,3)
            # try common shapes: (N,C,H,W)
            inp_t = np.transpose(inp, (0,3,1,2))
            import torch as _torch
            with _torch.no_grad():
                tensor = _torch.from_numpy(inp_t).float()
                out = model_obj(tensor)
                # attempt to get predictions
                try:
                    preds = out.cpu().numpy()
                except Exception:
                    preds = None
                st.write("PyTorch model output (raw):")
                st.write(preds if preds is not None else str(out))
        except Exception as e:
            st.error(f"PyTorch model inference failed: {e}")

    elif mtype == "pickle":
        # Generic pickled Python object. Try several predict APIs:
        obj = model_obj
        inp = preprocess_for_generic_model(img_bgr, target_size=(224,224))  # (1,H,W,3)
        # First try obj.predict(img_batch)
        tried = []
        success = False
        try:
            if hasattr(obj, "predict"):
                tried.append("predict(X)")
                # many scikit-learn or tf wrappers expect flat vectors; try both
                try:
                    res = obj.predict(inp)
                except Exception:
                    # try flatten per-sample
                    res = obj.predict(inp.reshape((inp.shape[0], -1)))
                st.write("Model.predict output:")
                st.write(res)
                success = True
            elif hasattr(obj, "forward") or hasattr(obj, "__call__"):
                tried.append("call()")
                # try calling directly
                try:
                    res = obj(inp)
                except Exception:
                    res = obj(inp.reshape((inp.shape[0], -1)))
                st.write("Callable object output:")
                st.write(res)
                success = True
        except Exception as e:
            st.write(f"Attempted calls {tried} but inference failed: {e}")
        if not success:
            st.error("Pickled model loaded but no supported inference method found (predict/call). Check model interface.")

    else:
        st.error(f"Unsupported model type: {mtype}")

# show helpful debug info
st.markdown("---")
st.subheader("Debug / Notes")
st.write("If you put your model file in the repo, use the exact relative path above (e.g., `models/model.pkl`).")
st.write("Pickled models: the app will attempt to call `.predict(X)` or call the object directly. If your model expects custom preprocessing or a different input shape, update the app accordingly or tell me the model input format and I will adapt the preprocessing.")
