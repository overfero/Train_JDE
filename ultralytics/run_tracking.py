from ultralytics import YOLO
import time

# model_name = "models/yolo26s_cpu.onnx"
# model_name = "models/yolo26s_openvino_model"
# model_name = "models/yolo26s_gpu_fp16.onnx"
model_name = "models/yolo26s.engine"
# model_name = "models/yolo26s_saved_model/yolo26s_float16.tflite"

model = YOLO(model_name)

N_FRAMES = 3648

# PENTING: stream=True agar hasil tidak akumulasi di RAM
# Tanpa stream=True → 3648 * (1920x1080x3 uint8) ≈ 22GB RAM → GC pressure → +33ms/frame
t_start = time.perf_counter()
frame_count = 0
for r in model.track(
    source="video2.mp4",
    half=True,
    device="cuda:0",
    persist=True,
    tracker="botsort.yaml",
    stream=True,   # ← WAJIB untuk pipeline yang efisien
    verbose=False,
):
    frame_count += 1
    # Akses hasil di sini kalau perlu, misal:
    # boxes = r.boxes.xyxy  # bounding boxes
    # track_ids = r.boxes.id  # track IDs

t_elapsed = time.perf_counter() - t_start
print(f"⏱  Total time: {t_elapsed:.2f}s  |  avg {t_elapsed / frame_count * 1000:.1f} ms/frame  |  {frame_count / t_elapsed:.1f} FPS  ({frame_count} frames)")