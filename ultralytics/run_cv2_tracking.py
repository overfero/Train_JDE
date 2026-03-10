from collections import defaultdict
from scipy.spatial.distance import cosine
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

# ─── Device Check ────────────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(DEVICE)} (CUDA {torch.version.cuda})")
    print(f"   Available GPUs: {torch.cuda.device_count()}x (using GPU {DEVICE} for tracking)")
else:
    print("⚠️  No GPU found, running on CPU — this will be very slow!")

# ─── Config ───────────────────────────────────────────────────────────────────
TRACKER_CFG = "botsort.yaml"
VIDEO_IN    = "video2.mp4"
VIDEO_OUT   = "output.mp4"
LOG_PATH    = "sim1.txt"
IMGSZ       = 1280

track_history = defaultdict(lambda: [])
model = YOLO("models/yolo26s.engine")
cap   = cv2.VideoCapture(VIDEO_IN)

w, h, fps    = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                            cv2.CAP_PROP_FRAME_HEIGHT,
                                            cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

frame_idx=0
for _ in tqdm(range(total_frames), desc="Processing"):
    ret, im0 = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model.track(im0, persist=True, verbose=True,
                            tracker=TRACKER_CFG, device=DEVICE, half=True, imgsz=IMGSZ)

    # ── Draw bboxes ───────────────────────────────────────────────────────
    active_ids, active_confs, active_cls = [], [], []
    if results[0].boxes.id is not None:
        annotator = Annotator(im0, line_width=2)
        boxes     = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs     = results[0].boxes.conf.cpu().tolist()
        clss      = results[0].boxes.cls.int().cpu().tolist()
        for box, tid, conf, cls in zip(boxes, track_ids, confs, clss):
            annotator.box_label(box,
                label=f"ID:{tid} {model.names[cls]} {conf:.2f}",
                color=colors(tid, True))
            active_ids.append(tid); active_confs.append(conf); active_cls.append(cls)
        im0 = annotator.result()

        

# ─── Cleanup ──────────────────────────────────────────────────────────────────
out.release()
cap.release()
print(f"✅ Done! Video → {VIDEO_OUT}  |  Log → {LOG_PATH}")