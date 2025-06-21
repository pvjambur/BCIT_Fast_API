import os, json, torch
from PIL import Image
import torch.nn as nn   
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Config
PROJECT_ROOT = "C:\\Users\\jambu\\13th_June_Bat"
MODEL_PATH_10 = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_bat.pth")
MODEL_PATH_31 = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_bat_31_modifiedspecies.pth")
CLASSES_PATH_10 = os.path.join(PROJECT_ROOT, "models", "classes.json")
CLASSES_PATH_31 = os.path.join(PROJECT_ROOT, "models", "new_classes_31.json")

# Device & transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Load classes
with open(CLASSES_PATH_10, 'r', encoding='utf-8') as f:
    classes_10 = json.load(f)
with open(CLASSES_PATH_31, 'r', encoding='utf-8') as f:
    classes_31 = json.load(f)

# Load models
def load_model(model_path, num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval().to(device)
    return model

model_10 = load_model(MODEL_PATH_10, len(classes_10))
model_31 = load_model(MODEL_PATH_31, len(classes_31))

def classify_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    # First try with 10-species model
    with torch.no_grad():
        out = model_10(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
        confidence_percent = round(conf.item() * 100, 2)
    
    # If confidence >= 99%, return from 10-species model
    if confidence_percent >= 99:
        predicted_class = classes_10[idx]
        print(f"Predicted (10-species model): {predicted_class} ({confidence_percent}%)")
        return predicted_class, confidence_percent
    
    # Otherwise try with 31-species model
    with torch.no_grad():
        out = model_31(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
        confidence_percent = round(conf.item() * 100, 2)
    
    if confidence_percent < 50:  # You can adjust this threshold
        print("Low confidence, returning fallback class")
        return "Unknown", confidence_percent
    else:
        predicted_class = classes_31[idx]
        print(f"Predicted (31-species model): {predicted_class} ({confidence_percent}%)")
        return predicted_class, confidence_percent

# Example usage
result = classify_image("C:\\Users\\jambu\\Downloads\\spectoo11.jpg")
print(result)