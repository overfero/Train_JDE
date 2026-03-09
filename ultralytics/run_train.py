from ultralytics import YOLO

model = YOLO("yolo26s-jde.yaml", task="jde").load("best.pt")

model.train(
    # Dataset
    data="data.yaml",

    # Full training setup (from paper)
    epochs=100,
    batch=16,
    imgsz=1280,

    # JDE critical — mosaic never turned off
    close_mosaic=0,

    # Optimizer
    patience=25,

    # Device — sesuaikan dengan GPU yang tersedia
    device=[0,1],  # single GPU, ganti ke [0,1,...] kalau multi-GPU

    # Pretrained weights
    # sudah di-load via .load() di atas

    # Logging
    project="runs/jde",
    name="crowdmot",
)