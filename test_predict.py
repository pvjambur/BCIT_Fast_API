import os
import json
import torch
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import gc
import cv2
import numpy as np

def clear_cache():
    """Clear all relevant caches to ensure fresh predictions"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def preprocess_spectrogram(image_path):
    """Enhanced spectrogram preprocessing"""
    try:
        # Read image with OpenCV for better control
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Convert to PIL Image for compatibility with torchvision transforms
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        # Apply histogram equalization for better contrast
        img_array = np.array(pil_img)
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        return Image.fromarray(img_eq)
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return Image.open(image_path).convert('RGB')  # Fallback

def main():
    # Clear cache before starting
    clear_cache()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bat Species Classifier')
    parser.add_argument('spectrogram_path', type=str, help='Path to spectrogram image')
    args = parser.parse_args()

    # Configuration
    PROJECT_ROOT = "C:\\Users\\jambu\\13th_June_Bat"
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_bat.pth")
    CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes.json")

    # Device & transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load class names
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    # Load model with proper handling of size mismatch
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Get the number of classes from the class names file
    num_classes = len(class_names)
    
    # Load the saved state dict
    saved_state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Check for size mismatch in the final layer
    if saved_state_dict['_fc.weight'].shape[0] != num_classes:
        print(f"Warning: Number of classes in model ({saved_state_dict['_fc.weight'].shape[0]}) "
              f"doesn't match current class count ({num_classes}). Adjusting model...")
        
        # Remove the final layer weights from the saved state dict
        saved_state_dict.pop('_fc.weight')
        saved_state_dict.pop('_fc.bias')
        
        # Initialize model with correct number of classes
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        
        # Load the modified state dict (strict=False allows missing keys)
        model.load_state_dict(saved_state_dict, strict=False)
    else:
        # No mismatch, load normally
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        model.load_state_dict(saved_state_dict)
    
    model.eval().to(device)

    # Process single image with enhanced preprocessing
    try:
        clear_cache()  # Clear cache before processing
        
        # Enhanced preprocessing
        preprocessed_img = preprocess_spectrogram(args.spectrogram_path)
        
        # Transform and add batch dimension
        img_tensor = transform(preprocessed_img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[pred_idx.item()]
        confidence_score = round(confidence.item() * 100, 2)
        
        print(f"\nPrediction Result:")
        print(f"Species: {predicted_class}")
        print(f"Confidence: {confidence_score}%")
        
    except Exception as e:
        print(f"\nError processing image: {str(e)}")
        print("Prediction failed")
    finally:
        clear_cache()  # Clear cache after processing

if __name__ == "__main__":
    main()