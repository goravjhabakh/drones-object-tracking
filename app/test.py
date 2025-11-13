from ultralytics import YOLO

model = YOLO("app/model/best.pt")

results = model("app/sample.jpeg")
results[0].show()