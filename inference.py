import os, json, torch
from PIL import Image
import torch.nn as nn   
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Config
PROJECT_ROOT = "C:\\Users\\jambu\\13th_June_Bat"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_bat.pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes.json")

# Device & transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Load classes
with open(CLASSES_PATH) as f:
    classes = json.load(f)

# Load model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, len(classes))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
except:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval().to(device)

# … (imports and model‐loading as before)

def classify_image(img_path, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
    if conf < threshold:
        return "Unknown", conf.item()
    else:
        return classes[idx], conf.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_png>")
        sys.exit(1)
    img_path = sys.argv[1]
    species, confidence = classify_image(img_path, threshold=0.5)  # tweak threshold if needed
    print(f"Predicted: {species} ({confidence*100:.1f}%)")