from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mnist_classifier_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data, labels):
        data = self.scaler.fit_transform(data)
        self.model.fit(data, labels)

    def predict(self, data):
        data = self.scaler.transform(data)
        return self.model.predict(data)
