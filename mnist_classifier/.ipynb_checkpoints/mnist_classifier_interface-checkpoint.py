from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, data, labels):
        """Train Model"""
        pass

    @abstractmethod
    def predict(self, data):
        """Prediction"""
        pass