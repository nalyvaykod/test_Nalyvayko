import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load the trained model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 класів
model.load_state_dict(torch.load("classification_model.pth"))
model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output).item()
    return predicted_class

# Example usage
print(classify_image("../data/Animals-10/butterfly/1.jpg"))
