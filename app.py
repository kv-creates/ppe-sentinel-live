"""PPE Sentinel Live — standalone Streamlit demo for Streamlit Community Cloud.

Zero heavy dependencies: ONNX Runtime inference (no torch), Pillow imaging
(no OpenCV), imageio-ffmpeg for video. Deploy: push to GitHub, then
https://share.streamlit.io -> New app.
"""
import glob
import io
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

CLASSES = ["person", "helmet", "vest", "no-helmet", "no-vest", "boots"]
VIOLATIONS = {"no-helmet", "no-vest"}
COMPLIANT = {"helmet", "vest", "boots"}
COLORS = {"person": (255, 200, 60), "helmet": (60, 200, 60), "vest": (60, 200, 60),
          "boots": (60, 200, 60), "no-helmet": (230, 60, 60), "no-vest": (230, 60, 60)}
WEIGHTS = os.path.join(os.path.dirname(__file__), "model", "best.onnx")
IMG_SIZE = 640

st.set_page_config(page_title="PPE Sentinel Live", layout="wide")
with open(os.path.join(os.path.dirname(__file__), "win7.css"), encoding="utf-8") as fh:
    st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLOv8 model (one-time)...")
def load_model():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(WEIGHTS, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def letterbox(img: Image.Image):
    w, h = img.size
    s = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(w * s), int(h * s)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
    canvas.paste(img.resize((nw, nh)), ((IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2))
    return canvas, s, (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def get_zone(cx, cy):
    if cy < 1 / 3:
        return "red"
    if cy < 2 / 3:
        return "yellow"
    return "green"


def detect(img: Image.Image, conf: float):
    """Full YOLOv8 post-processing in numpy. Returns [dets], PIL overlay."""
    sess = load_model()
    canvas, s, px, py = letterbox(img.convert("RGB"))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {"images": x})[0][0].T  # (8400, 10): xywh + 6 scores
    boxes, scores, cids = [], [], []
    W, H = img.size
    for row in out:
        cx, cy, bw, bh = row[:4]
        cid = int(row[4:].argmax())
        sc = float(row[4:].max())
        if sc < conf:
            continue
        x1 = ((cx - bw / 2) - px) / s
        y1 = ((cy - bh / 2) - py) / s
        x2 = ((cx + bw / 2) - px) / s
        y2 = ((cy + bh / 2) - py) / s
        boxes.append([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
        scores.append(sc)
        cids.append(cid)
    dets = []
    if boxes:
        boxes = np.array(boxes)
        for cid in set(cids):
            idx = [k for k, c in enumerate(cids) if c == cid]
            for k in nms(boxes[idx], np.array(scores)[idx]):
                j = idx[k]
                x1, y1, x2, y2 = boxes[j].tolist()
                dets.append({"class_name": CLASSES[cid],
                             "conf": round(scores[j], 3),
                             "bbox": [x1, y1, x2, y2],
                             "zone": get_zone((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)})
    ov = img.convert("RGB").copy()
    dr = ImageDraw.Draw(ov)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        col = COLORS.get(d["class_name"], (255, 255, 255))
        dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
        dr.text((x1 + 2, max(0, y1 - 14)), f"{d['class_name']} {d['conf']:.2f}", fill=col)
    return dets, ov


def notify_telegram(text: str) -> bool:
    try:
        sec = st.secrets.get("telegram", {})
        token, chat = sec.get("bot_token"), sec.get("chat_id")
        if not (token and chat):
            return False
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": str(chat), "text": text}, timeout=20)
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception:  # noqa: BLE001
        return False


st.markdown("<div class='win7-titlebar'>PPE Sentinel Live — PPE Non-Compliance Detector"
            "<span class='winbtns'><span>_</span><span>□</span>"
            "<span>×</span></span></div>", unsafe_allow_html=True)
st.markdown("<div class='win7-panel'>", unsafe_allow_html=True)
st.markdown("Real-time helmet / vest / boots compliance checking with YOLOv8. "
            "Pick a sample photo, upload one, or take a webcam snapshot, then "
            "press Run Detection.")

source = st.selectbox("Source", ["Sample photo", "Upload image",
                                 "Webcam snapshot", "Upload video (MP4)"])
conf = st.slider("Confidence threshold", 0.05, 0.80, 0.30, 0.05)

img, video_file = None, None
if source == "Sample photo":
    samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*")))
    samples = [s for s in samples if s.lower().endswith((".jpg", ".jpeg", ".png"))]
    choice = st.selectbox("Sample", samples, format_func=os.path.basename)
    if choice:
        img = Image.open(choice)
elif source == "Upload image":
    up = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(io.BytesIO(up.read()))
elif source == "Upload video (MP4)":
    import tempfile
    up = st.file_uploader("Choose an MP4 (short clips work best on free Cloud)",
                          type=["mp4", "mov"])
    if up:
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as fh:
            fh.write(up.read())
        video_file = tmp
else:
    shot = st.camera_input("Webcam snapshot")
    if shot:
        img = Image.open(io.BytesIO(shot.read()))

if img is not None and st.button("Run Detection"):
    with st.spinner("Detecting..."):
        dets, ov = detect(img, conf)
    st.image(ov, use_container_width=True)
    safe = sum(1 for d in dets if d["class_name"] in COMPLIANT)
    unsafe = sum(1 for d in dets if d["class_name"] in VIOLATIONS)
    pct = round(100.0 * safe / (safe + unsafe), 1) if (safe + unsafe) else 100.0
    c1, c2, c3 = st.columns(3)
    for col, name, val in ((c1, "Safe count", safe), (c2, "Unsafe count", unsafe),
                           (c3, "Compliance %", f"{pct}%")):
        with col:
            st.markdown(f"<div class='win7-kpi'><div class='k-name'>{name}</div>"
                        f"<div class='k-val'>{val}</div></div>", unsafe_allow_html=True)
    if unsafe:
        st.markdown("<div class='win7-alert red'>Warning: <b>PPE VIOLATION "
                    "DETECTED</b> — supervisor should intervene.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='win7-alert'>No violation — site compliant.</div>",
                    unsafe_allow_html=True)
    with st.expander("Raw detections"):
        st.json(dets)

if video_file and st.button("Process Video", key="vid"):
    import imageio.v2 as imageio
    reader = imageio.get_reader(video_file)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))
    out_tmp = video_file.replace(".mp4", "_annotated.mp4")
    writer = imageio.get_writer(out_tmp, fps=max(1.0, fps / 5), codec="libx264",
                                quality=8, macro_block_size=None)
    n_vio = n_red = vio_frames = n_scored = 0
    try:
        total_frames = int(meta.get("nframes", 0) or 0)
    except (OverflowError, ValueError):  # imageio may report inf
        total_frames = 0
    if not total_frames or total_frames != total_frames:  # 0/NaN guard
        total_frames = 0
    bar = st.progress(0, text="Scoring frames...")
    with st.spinner("Processing video..."):
        for n, frame in enumerate(reader):
            if n % 5:
                continue
            n_scored += 1
            dets, ov = detect(Image.fromarray(frame), conf)
            vio = [d for d in dets if d["class_name"] in VIOLATIONS]
            n_vio += len(vio)
            n_red += sum(1 for d in vio if d["zone"] == "red")
            vio_frames += bool(vio)
            writer.append_data(np.asarray(ov))
            if total_frames:
                bar.progress(min(0.99, (n + 1) / total_frames))
            else:
                bar.progress(min(0.99, (n_scored % 100) / 100))
    reader.close()
    writer.close()
    bar.progress(1.0, text="Done.")
    m1, m2 = st.columns(2)
    for col, name, val in ((m1, "Frames scored", n_scored), (m2, "Violations", n_vio)):
        with col:
            st.markdown(f"<div class='win7-kpi'><div class='k-name'>{name}</div>"
                        f"<div class='k-val'>{val}</div></div>", unsafe_allow_html=True)
    if vio_frames:
        st.markdown("<div class='win7-alert red'>Warning: violations found in "
                    f"{vio_frames} scored frame(s).</div>", unsafe_allow_html=True)
        if n_red:
            tg = notify_telegram(f"RED-ZONE PPE VIOLATION in uploaded video: "
                                 f"{n_red} red-zone box(es).")
            st.markdown("Telegram alert: **%s**" % ("sent" if tg else
                        "not sent (bot secrets missing)"))
        else:
            st.markdown("No red-zone violations — Telegram stays quiet "
                        "(red-zone-only policy).")
    st.video(out_tmp)
    os.remove(video_file)

st.markdown("---")
with st.expander("Model card"):
    st.markdown("YOLOv8m trained 60 epochs on a T4 GPU on 1,331 real "
                "construction-site photos (6 classes: person, helmet, vest, "
                "no-helmet, no-vest, boots). Test mAP@0.5 0.70, precision "
                "0.72, recall 0.70. Runs here as ONNX on CPU — no GPU needed. "
                "Full project, API backend, and training notebook: see the "
                "ppe-detector repository.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='win7-statusbar'><span>Ready</span>"
            "<span>Model: YOLOv8m-ONNX &nbsp;|&nbsp; Cloud demo build</span></div>",
            unsafe_allow_html=True)
