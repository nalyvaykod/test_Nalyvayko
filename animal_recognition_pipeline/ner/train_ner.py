import spacy
import os
import random
import json
from spacy.training import Example

# Fix OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path to dataset
dataset_path = r"C:\Users\rainb\OneDrive\Desktop\test_tasks\animal_recognition_pipeline\data\Animals-10"

# Ensure dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Extract class names (animal names) from folder names
animal_classes = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

# Generate training data with correct offsets
train_data = []
for animal in animal_classes:
    sentences = [
        f"There is a {animal} in the picture.",
        f"I see a {animal}.",
        f"A {animal} is playing in the park.",
        f"The {animal} is running fast.",
        f"I have a pet {animal}."
    ]
    
    for text in sentences:
        start = text.find(animal)
        if start != -1:
            end = start + len(animal)
            train_data.append({"text": text, "entities": [[start, end, "ANIMAL"]]})

# Save the generated dataset to a JSON file
json_path = os.path.join(dataset_path, "ner_train.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4)

print(f"NER training data saved to {json_path}")

# Load spaCy model
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels from the dataset
for item in train_data:
    for entity in item["entities"]:
        ner.add_label("ANIMAL")

# Initialize training
nlp.initialize()

# Convert training data into spaCy's `Example` format
examples = []
for item in train_data:
    doc = nlp.make_doc(item["text"])
    example = Example.from_dict(doc, {"entities": item["entities"]})
    examples.append(example)

# Train the model
for i in range(10):  # Train for 10 epochs
    random.shuffle(examples)
    losses = {}
    nlp.update(examples, drop=0.5, losses=losses)
    print(f"Epoch {i+1}, Loss: {losses}")

# Save the trained model
model_path = r"C:\Users\rainb\OneDrive\Desktop\test_tasks\animal_recognition_pipeline\ner_model"
nlp.to_disk(model_path)

print(f"NER model training completed and saved to '{model_path}'")
