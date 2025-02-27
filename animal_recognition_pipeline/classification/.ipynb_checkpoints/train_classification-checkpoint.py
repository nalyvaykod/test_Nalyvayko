import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
from torchvision.models import MobileNet_V2_Weights

# Set CPU optimizations
torch.set_num_threads(8)  # Use 8 CPU threads for better performance
torch.backends.mkldnn.enabled = True  

# Path to the dataset
dataset_path = r"C:\Users\rainb\OneDrive\Desktop\test_tasks\animal_recognition_pipeline\data\Animals-10"

# Remove .ipynb_checkpoints folder if it exists
checkpoints_path = os.path.join(dataset_path, ".ipynb_checkpoints")
if os.path.exists(checkpoints_path):
    shutil.rmtree(checkpoints_path)
    print("Removed .ipynb_checkpoints")

# Ensure the dataset directory exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check your dataset location.")

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
])

# Load dataset and prepare DataLoader
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0) 

# Load a pre-trained MobileNetV2 model and modify the classifier
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1) 
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))

# Optimize memory
model = model.to(torch.float32)
model = model.to(memory_format=torch.channels_last) 

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1) 

# Enable gradient accumulation
accumulation_steps = 4  

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=1): 
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()  # Start accumulating gradients
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(torch.float32), labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights every 4 batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "classification_model.pth")

print("Model training completed and saved successfully!")
