import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier_interface import MnistClassifierInterface

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self, lr=0.001):
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, data, labels):
        self.model.train()
        data = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28)  # Оновлено
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28)  # Оновлено
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
