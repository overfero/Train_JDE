from ultralytics import YOLO

model = YOLO("yolo11s-jde.yaml", task="jde").load("yolo11s.pt")

model.train(
    # Dataset
    data="crowdhuman_mot20_custom.yaml",

    # Full training setup (from paper)
    epochs=100,
    batch=64,
    imgsz=1280,

    # JDE critical — mosaic never turned off
    close_mosaic=0,

    # Optimizer
    patience=25,

    # Device — sesuaikan dengan GPU yang tersedia
    device=0,  # single GPU, ganti ke [0,1,...] kalau multi-GPU

    # Pretrained weights
    # sudah di-load via .load() di atas

    # Logging
    project="runs/jde",
    name="crowdhuman_mot20_custom_100e_64b",

    # Recommended untuk JDE
    workers=2,
    cache=True,
)