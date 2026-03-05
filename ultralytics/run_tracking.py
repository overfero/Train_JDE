"""
Instance Segmentation + Tracking — YOLO11l JDE
Optimized: Threading Pipeline + Async Annotation + FP16
"""

from collections import defaultdict
import cv2
import torch
import threading
from queue import Queue
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
SOURCE      = "video2.mp4"
OUTPUT      = "output.mp4"
MODEL_PATH  = "runs/train4/train4/weights/best.pt"
DEVICE      = 0
FP16        = False
IMG_SIZE    = 1280
QUEUE_SIZE  = 64
SENTINEL    = None
TRACKER     = "botsort.yaml"

# ─────────────────────────────────────────────
# INISIALISASI
# ─────────────────────────────────────────────
print("[INFO] Loading YOLO11s JDE...")
model = YOLO(MODEL_PATH, task="jde")
track_history = defaultdict(list)

# ─────────────────────────────────────────────
# VIDEO I/O
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)
assert cap.isOpened(), f"Cannot open: {SOURCE}"

w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] {w}x{h} | {fps} FPS | {total_frames} frames")

writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# ─────────────────────────────────────────────
# QUEUES
# ─────────────────────────────────────────────
read_queue    = Queue(maxsize=QUEUE_SIZE)   # raw frames
results_queue = Queue(maxsize=QUEUE_SIZE)   # (frame, results) → annotator thread
write_queue   = Queue(maxsize=QUEUE_SIZE)   # annotated frames → writer thread

# ─────────────────────────────────────────────
# THREAD 1: VIDEO READER (prefetch ke RAM)
# ─────────────────────────────────────────────
def reader_thread():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        read_queue.put(frame)
    read_queue.put(SENTINEL)
    print("[Reader] Done.")

# ─────────────────────────────────────────────
# THREAD 2: ANNOTATION (CPU, paralel dengan inference)
# annotation dipisah dari inference agar GPU tidak nganggur
# ─────────────────────────────────────────────
def annotator_thread():
    while True:
        item = results_queue.get()
        if item is SENTINEL:
            break

        im0, results = item

        if results[0].boxes.id is not None:
            annotator = Annotator(im0, line_width=2)
            boxes     = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss      = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(
                    box,
                    label=f"ID:{track_id} {model.names[cls]}",
                    color=colors(track_id, True)
                )
                if results[0].masks is not None:
                    for mask in results[0].masks.xy:
                        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True))

            im0 = annotator.result()

        write_queue.put(im0)

    write_queue.put(SENTINEL)
    print("[Annotator] Done.")

# ─────────────────────────────────────────────
# THREAD 3: VIDEO WRITER
# ─────────────────────────────────────────────
def writer_thread():
    while True:
        frame = write_queue.get()
        if frame is SENTINEL:
            break
        writer.write(frame)
    print("[Writer] Done.")

# ─────────────────────────────────────────────
# MAIN: INFERENCE (harus sequential karena persist=True)
# ─────────────────────────────────────────────
def inference_loop():
    pbar = tqdm(total=total_frames, desc="Tracking", unit="frame", dynamic_ncols=True)

    while True:
        frame = read_queue.get()
        if frame is SENTINEL:
            break

        # Track — sequential wajib untuk JDE persist mode
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            device=DEVICE,
            half=FP16,
            tracker=TRACKER,
            imgsz=IMG_SIZE,
        )

        # Kirim ke annotator thread (non-blocking)
        results_queue.put((frame.copy(), results))
        pbar.update(1)

    results_queue.put(SENTINEL)
    pbar.close()
    print("[Inference] Done.")

# ─────────────────────────────────────────────
# JALANKAN SEMUA THREAD
# ─────────────────────────────────────────────
t_read   = threading.Thread(target=reader_thread,   daemon=True)
t_annot  = threading.Thread(target=annotator_thread, daemon=True)
t_write  = threading.Thread(target=writer_thread,   daemon=True)

t_read.start()
t_annot.start()
t_write.start()

inference_loop()   # main thread → GPU inference

t_read.join()
t_annot.join()
t_write.join()

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
cap.release()
writer.release()
# cv2.destroyAllWindows()
print(f"✅ Done! Output saved to: {OUTPUT}")