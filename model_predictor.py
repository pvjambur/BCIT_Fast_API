import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from efficientnet_pytorch import EfficientNet

class BatSpeciesClassifier:
    def __init__(self, model_path="models/efficientnet_b0_bat.pth", 
                 class_map_path="models/classes.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 224
        
        # Load class labels
        with open(class_map_path, "r") as f:
            self.class_names = json.load(f)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = EfficientNet.from_name("efficientnet-b0")
        self.model._fc = nn.Linear(self.model._fc.in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
    
    def predict(self, img_path):
        if not os.path.exists(img_path):
            return None
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, pred_idx = torch.max(outputs, 1)
            predicted_class = self.class_names[pred_idx.item()]
        
        return predicted_class