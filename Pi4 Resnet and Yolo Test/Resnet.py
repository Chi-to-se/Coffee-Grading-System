import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import sys

# ── Config ──────────────────────────────────────────────
MODEL_PATH  = "best_model_resnet18.pth"
NUM_CLASSES = 5
CLASS_NAMES = ["class0", "class1", "class2", "class3", "class4"]  # adjust to actual class names
# ────────────────────────────────────────────────────────

device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

image_path = "A02_17.png"
img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# warmup once before timing
with torch.no_grad():
    _ = model(img)

start = time.perf_counter()
with torch.no_grad():
    output = model(img)
elapsed = (time.perf_counter() - start) * 1000  # ms

prob       = torch.softmax(output, dim=1)
top1_idx   = prob.argmax().item()
top1_conf  = prob[0, top1_idx].item()
top1_class = CLASS_NAMES[top1_idx]

print(f"Class: {top1_class} | Confidence: {top1_conf:.4f} | Time: {elapsed:.2f} ms")