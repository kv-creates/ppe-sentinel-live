"""PPE Sentinel Live — standalone Streamlit demo for Streamlit Community Cloud.

Self-contained: YOLOv8 inference runs in-process (no FastAPI backend).
Deploy: push this folder to GitHub, then https://share.streamlit.io -> New app.
"""
import glob
import os

import cv2
import numpy as np
import streamlit as st

CLASSES = ["person", "helmet", "vest", "no-helmet", "no-vest", "boots"]
VIOLATIONS = {"no-helmet", "no-vest"}
COMPLIANT = {"helmet", "vest", "boots"}
COLORS = {"person": (255, 200, 60), "helmet": (60, 200, 60), "vest": (60, 200, 60),
          "boots": (60, 200, 60), "no-helmet": (60, 60, 230), "no-vest": (60, 60, 230)}
WEIGHTS = os.path.join(os.path.dirname(__file__), "model", "best.pt")

st.set_page_config(page_title="PPE Sentinel Live", layout="wide")
with open(os.path.join(os.path.dirname(__file__), "win7.css"), encoding="utf-8") as fh:
    st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLOv8 model (one-time, ~1 min)...")
def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)


def get_zone(cx, cy):
    if cy < 1 / 3:
        return "red"
    if cy < 2 / 3:
        return "yellow"
    return "green"


def detect(img_bgr, conf):
    model = load_model()
    res = model.predict(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    h, w = res.orig_shape
    dets = []
    for b in res.boxes:
        cid = int(b.cls.item())
        if not 0 <= cid < len(CLASSES):
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append({"class_name": CLASSES[cid], "conf": round(float(b.conf.item()), 3),
                     "bbox": [x1, y1, x2, y2],
                     "zone": get_zone((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)})
    return dets


def overlay(img_bgr, dets):
    return cv2.cvtColor(overlay_bgr(img_bgr, dets), cv2.COLOR_BGR2RGB)


def overlay_bgr(img_bgr, dets):
    img = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        col = COLORS.get(d["class_name"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{d['class_name']} {d['conf']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return img


def notify_telegram(text: str) -> bool:
    """Send via st.secrets telegram creds when present (Cloud-safe)."""
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

img, fname, video_file = None, "frame.jpg", None
if source == "Sample photo":
    samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*")))
    samples = [s for s in samples if s.lower().endswith((".jpg", ".jpeg", ".png"))]
    choice = st.selectbox("Sample", samples, format_func=os.path.basename)
    if choice:
        img = cv2.imread(choice)
elif source == "Upload image":
    up = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
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
        img = cv2.imdecode(np.frombuffer(shot.read(), np.uint8), cv2.IMREAD_COLOR)

if st.button("Run Detection", disabled=img is None):
    with st.spinner("Detecting..."):
        dets = detect(img, conf)
    st.image(overlay(img, dets), use_container_width=True)
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
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_vio, n_frames, vio_frames, n_red = 0, 0, 0, 0
    writer = None
    out_tmp = video_file.replace(".mp4", "_annotated.mp4")
    bar = st.progress(0, text="Scoring frames...")
    with st.spinner("Processing video (CPU ~1-2 s per scored frame)..."):
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            if (n_frames - 1) % 5:
                continue
            dets = detect(frame, conf)
            vio = [d for d in dets if d["class_name"] in VIOLATIONS]
            n_vio += len(vio)
            n_red += sum(1 for d in vio if d["zone"] == "red")
            vio_frames += bool(vio)
            if writer is None:
                h, w = frame.shape[:2]
                writer = imageio.get_writer(out_tmp, fps=max(1.0, fps / 5),
                                            codec="libx264", quality=8,
                                            macro_block_size=None)
            writer.append_data(cv2.cvtColor(
                overlay_bgr(frame, dets), cv2.COLOR_BGR2RGB))
            bar.progress(min(0.99, n_frames / max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)))
        cap.release()
        if writer:
            writer.close()
    bar.progress(1.0, text="Done.")
    m1, m2 = st.columns(2)
    for col, name, val in ((m1, "Frames scored", f"~{n_frames // 5}"),
                           (m2, "Violations", n_vio)):
        with col:
            st.markdown(f"<div class='win7-kpi'><div class='k-name'>{name}</div>"
                        f"<div class='k-val'>{val}</div></div>", unsafe_allow_html=True)
    if vio_frames:
        st.markdown("<div class='win7-alert red'>Warning: violations found in "
                    f"{vio_frames} scored frame(s).</div>", unsafe_allow_html=True)
        if n_red:
            tg = notify_telegram(f"RED-ZONE PPE VIOLATION in uploaded video: "
                                 f"{n_red} red-zone box(es).")
        else:
            tg = False
            st.markdown("No red-zone violations — Telegram stays quiet "
                        "(red-zone-only policy).")
        st.markdown("Telegram alert: **%s**" % ("sent" if tg else
                    "not sent (no red-zone violation, or bot secrets missing)"))
    if writer:
        st.video(out_tmp)
    os.remove(video_file)

st.markdown("---")
with st.expander("Model card"):
    st.markdown("YOLOv8m trained 60 epochs on a T4 GPU on 1,331 real "
                "construction-site photos (6 classes: person, helmet, vest, "
                "no-helmet, no-vest, boots). Test mAP@0.5 0.70, precision "
                "0.72, recall 0.70. Full project, API backend, and training "
                "notebook: see the ppe-detector repository.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='win7-statusbar'><span>Ready</span>"
            "<span>Model: YOLOv8m &nbsp;|&nbsp; Cloud demo build</span></div>",
            unsafe_allow_html=True)
