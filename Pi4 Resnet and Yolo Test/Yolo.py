from ultralytics import YOLO
import sys

model = YOLO("yolo26cls_best_model.pt")

# image_path = sys.argv[1]
image_path = "A02_17.png"
results = model(image_path)

for result in results:
    probs = result.probs
    top1_class = result.names[probs.top1]
    top1_conf  = probs.top1conf.item()
    print(f"Class: {top1_class} | Confidence: {top1_conf:.4f}")