MNIST Classifier

Overview

This project implements three different models for MNIST digit classification using Object-Oriented Programming (OOP):

Random Forest (RF) - Traditional machine learning approach

Feed-Forward Neural Network (FNN) - Basic neural network

Convolutional Neural Network (CNN) - Deep learning approach

Each model is encapsulated in a class that implements the MnistClassifierInterface, ensuring a uniform API.

Installation

Clone the repository:

git clone <repository_url>
cd mnist_classifier

Install dependencies:

pip install -r requirements.txt

Usage

Running the Classifier

You can run the classifier by executing the following script in a Jupyter Notebook.

Explanation of the Approach

All models implement the MnistClassifierInterface, ensuring a unified API for training and prediction. MnistClassifier abstracts the model selection logic, allowing easy switching between different algorithms. load_data() ensures the dataset is preprocessed correctly for all models.Each model has its own directory, making it easy to modify or extend.

Requirements

Ensure you have the following dependencies installed:

  pip install torch torchvision numpy scikit-learn

Notes

The dataset is automatically downloaded if not found.

The models can be run independently using their respective scripts inside cnn_classifier, fnn_classifier, and rf_classifier.
