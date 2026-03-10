import cv2
import numpy as np
import torch
import time
import os
import threading
from queue import Queue
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort

# ─── Config ───────────────────────────────────────────────────────────────────
VIDEO_IN    = "video2.mp4"
YOLO_RUNS_DIR = "runs/detect/predict_thread"
TXT_DIR     = os.path.join(YOLO_RUNS_DIR, "labels")
IMGSZ       = 1280
CONF_THRESH = 0.1

TRACKOUT_TXT = "runs/track_boxmot_threaded.txt"

# ─── Device Check ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    raise RuntimeError("❌ BoxMOT requires a CUDA-capable GPU. No GPU found!")
    
if os.path.exists(TRACKOUT_TXT):
    os.remove(TRACKOUT_TXT)

# Shared Queue for inter-thread communication
# Queue items will be: (frame_id, orig_img, detections)
#   - detections is a numpy array of [x1, y1, x2, y2, conf, cls]
#     OR None if empty/end of stream
result_queue = Queue(maxsize=100) # Prevents memory overflow if YOLO is faster

# ─── THREAD 1: YOLO Prediction ───────────────────────────────────────────────
def run_yolo_predict():
    print("🚀 [YOLO Thread] Starting prediction...")
    model = YOLO("models/yolo26s_gpu_fp16.onnx", task="detect")
    
    # Run predict
    # Using stream=True returns a generator of `Results` objects
    results = model.predict(source=VIDEO_IN, device=DEVICE, half=True, imgsz=IMGSZ, 
                            conf=CONF_THRESH, verbose=True, stream=True,
                            project="runs/detect", name="predict_thread", exist_ok=True)
    
    frame_id = 0
    for result in results:
        frame_id += 1
        
        # orig_img is the BGR image directly from the video frame (numpy array)
        orig_img = result.orig_img
        
        # Parse detections to [x1, y1, x2, y2, conf, cls]
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            dets = np.column_stack((boxes, confs, clss))
        else:
            dets = np.empty((0, 6)) # empty array if no detection
            
        # Put into queue, this will block if Queue is full (waiting for tracker)
        result_queue.put((frame_id, orig_img, dets))
        
        if frame_id % 100 == 0:
            print(f"👁️ [YOLO Thread] Processed {frame_id} frames")

    # Signal the end of the video
    result_queue.put((-1, None, None))
    print("✅ [YOLO Thread] Prediction finished!")

# ─── THREAD 2: BoxMOT Tracking ───────────────────────────────────────────────
def run_boxmot_tracking():
    print("🚀 [Tracker Thread] Starting BoxMOT tracking...")
    
    tracker = BotSort(
        reid_weights=Path(""), 
        device=torch.device(DEVICE), 
        half=True,
        with_reid=False
    )
    
    # We will write the final tracking results here
    with open(TRACKOUT_TXT, 'w') as f:
        pass
        
    t_start = time.perf_counter()
    processed_frames = 0
    
    while True:
        # Get item from queue (this blocks/idles until YOLO puts a new frame)
        frame_id, frame, dets = result_queue.get()
        
        # End of video signal
        if frame_id == -1:
            break
            
        processed_frames += 1
        w, h = frame.shape[1], frame.shape[0]

        # Update the tracker
        # Even if dets is empty, some trackers need an update to age out lost tracks.
        # BoxMOT handles empty updates if we give it shape (0, 6)
        track_dets = tracker.update(dets, frame) 
        
        # Save to txt file
        if track_dets is not None and len(track_dets) > 0:
            with open(TRACKOUT_TXT, "a") as f:
                for t in track_dets:
                    x1, y1, x2, y2, track_id, conf, cls, ind = t
                    
                    # YOLO normalized format (cls, x_c, y_c, w, h, track_id)
                    w_box = x2 - x1
                    h_box = y2 - y1
                    x_c = (x1 + w_box / 2) / w
                    y_c = (y1 + h_box / 2) / h
                    w_norm = w_box / w
                    h_norm = h_box / h
                    
                    f.write(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f} {int(track_id)}\n")

        # Mark task as done in the queue (optional, good for join())
        result_queue.task_done()
        
        if processed_frames % 100 == 0:
            print(f"🎯 [Tracker Thread] Tracked {processed_frames} frames")

    t_elapsed = time.perf_counter() - t_start
    fps = processed_frames / t_elapsed if t_elapsed > 0 else 0
    print(f"⏱  Tracking finished: {t_elapsed:.2f}s  |  avg {t_elapsed / processed_frames * 1000:.1f} ms/frame  |  {fps:.1f} FPS")
    print(f"✅ Tracking boxes saved to {TRACKOUT_TXT}")


if __name__ == "__main__":
    t_main_start = time.perf_counter()
    
    # Creating threads
    yolo_thread = threading.Thread(target=run_yolo_predict)
    tracker_thread = threading.Thread(target=run_boxmot_tracking)
    
    # Start threads
    yolo_thread.start()
    tracker_thread.start()
    
    # Wait for both to finish
    yolo_thread.join()
    tracker_thread.join()
    
    t_main_elapsed = time.perf_counter() - t_main_start
    print(f"🎉 All processes finished in {t_main_elapsed:.2f}s!")
