import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class MnistClassifier:
    def __init__(self, algorithm):
        match algorithm:
            case "rf":
                from rf_classifier.random_forest_classifier import RandomForestMnistClassifier
                self.classifier = RandomForestMnistClassifier()
            case "nn":
                from fnn_classifier.feed_forward_classifier import FeedForwardMnistClassifier
                self.classifier = FeedForwardMnistClassifier()
            case "cnn":
                from cnn_classifier.convolutional_classifier import CNNMnistClassifier
                self.classifier = CNNMnistClassifier()
            case _:
                raise ValueError("Unsupported algorithm. Choose from: 'rf', 'nn', 'cnn'.")

    @staticmethod
    def load_data():
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        
       
        X = dataset.data.numpy().reshape(len(dataset), -1) / 255.0
        y = dataset.targets.numpy()

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test