import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self, lr=0.001):
        self.model = FeedForwardNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, data, labels):
        self.model.train()
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
