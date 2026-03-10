from ultralytics import YOLO
from pathlib import Path

# model_name = "models/yolo26s_cpu.onnx"
# model_name = "models/yolo26s_openvino_model"
# model_name = "models/yolo26s_gpu_fp16.onnx"
model_name = "models/yolo26s.engine"
# model_name = "models/yolo26s_saved_model/yolo26s_float16.tflite"

print(f"Loading model: {model_name}")
if not Path(model_name).exists():
    print(f"Model not found locally, will download from Ultralytics...")

model = YOLO(model_name)

results = model.track(
    source="video2.mp4",
    half=True,   
    device="cuda:0",     
    persist=True,  
    tracker="botsort.yaml",
)