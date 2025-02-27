import torch
from torchvision import models, transforms
from PIL import Image
import os
from ner.inference_ner import extract_animal 
import warnings

# Check if the model file exists
model_path = "classification_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found! Train the model first.")

# Load trained MobileNetV2 model
model = models.mobilenet_v2()

# Load the trained weights and determine class count dynamically
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
num_classes = checkpoint["classifier.1.weight"].shape[0]  # Extract number of classes
num_features = model.classifier[1].in_features

# Adjust classifier layer to match the trained model
model.classifier[1] = torch.nn.Linear(num_features, num_classes)
model.load_state_dict(checkpoint)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Match training preprocessing
    transforms.ToTensor(),
])

def classify_image(image_path):
    """Classifies an image and returns the predicted class index."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file '{image_path}' not found.")

    image = Image.open(image_path).convert("RGB")  
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class

# Map class indexes to names (auto-detect from training)
class_mapping = {i: f"class_{i}" for i in range(num_classes)}  
custom_mapping = {0: "butterfly", 1: "cat", 2: "cow", 3: "dog", 4: "elephant",
                  5: "hen", 6: "horse", 7: "squirrel", 8: "spider", 9: "sheep"}

# Update mapping if classes match
if len(custom_mapping) == num_classes:
    class_mapping = custom_mapping

def check_text_image_match(text, image_path):
    """Checks if the text correctly describes the image."""
    if not text:
        raise ValueError("❌ Provided text is empty.")

    extracted_animals = extract_animal(text)  # Get animal names from text
    predicted_class = classify_image(image_path)  # Predict image class

    if extracted_animals:
        return class_mapping[predicted_class] in extracted_animals

    return False
