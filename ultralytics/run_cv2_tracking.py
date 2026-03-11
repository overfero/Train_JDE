from collections import defaultdict
from contextlib import contextmanager
from scipy.spatial.distance import cosine
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm
import time

@contextmanager
def cv2_resource(resource):
    """Context manager for cleanly releasing OpenCV resources."""
    try:
        yield resource
    finally:
        if resource is not None:
            resource.release()

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
IMGSZ       = 1280

track_history = defaultdict(lambda: [])
model = YOLO("models/yolo26s.engine")


t_start = time.perf_counter()
with cv2_resource(cv2.VideoCapture(VIDEO_IN)) as cap:
    w, h, fps    = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                cv2.CAP_PROP_FRAME_HEIGHT,
                                                cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, im0 = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(im0, persist=True, verbose=False,
                                tracker=TRACKER_CFG, device=DEVICE, half=True, imgsz=IMGSZ)

t_elapsed = time.perf_counter() - t_start
print(f"⏱  Total time: {t_elapsed:.2f}s  |  avg {t_elapsed / 3648 * 1000:.1f} ms/frame  |  {3648 / t_elapsed:.1f} FPS")
