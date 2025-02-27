import spacy

# Load trained model
nlp = spacy.load("ner_model")

def extract_animal(text):
    doc = nlp(text)
    animals = [ent.text for ent in doc.ents if ent.label_ == "ANIMAL"]
    return animals

# Example usage
print(extract_animal("There is a dog in the picture."))
