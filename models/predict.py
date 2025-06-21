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
    
    # Get predictions from both models
    with torch.no_grad():
        # 10-species model
        out_10 = model_10(x)
        probs_10 = torch.softmax(out_10, dim=1)[0]
        conf_10, idx_10 = probs_10.max(0)
        confidence_10 = round(conf_10.item() * 100, 2)
        predicted_class_10 = classes_10[idx_10]
        
        # 31-species model
        out_31 = model_31(x)
        probs_31 = torch.softmax(out_31, dim=1)[0]
        conf_31, idx_31 = probs_31.max(0)
        confidence_31 = round(conf_31.item() * 100, 2)
        predicted_class_31 = classes_31[idx_31]
    
    # Compare confidences and return the best prediction
    if confidence_10 >= confidence_31:
        print(f"Predicted (10-species model): {predicted_class_10} ({confidence_10}%)")
        return predicted_class_10, confidence_10
    else:
        print(f"Predicted (31-species model): {predicted_class_31} ({confidence_31}%)")
        return predicted_class_31, confidence_31

# Example usage
# result = classify_image("path_to_your_spectrogram.jpg")
# print(result)