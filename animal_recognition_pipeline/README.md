Installation:

1.pip install -r requirements.txt

Training:
1.Train Image Classification Model

python classification/train_classification.py


2.Train NER Model

python ner/train_ner.py

Inference:


1.Classify Image

python classification/inference_classification.py --image_path "path_to_image"

2.Extract Animal Name

python ner/inference_ner.py --text "There is a cat in the picture."


Run the Full Pipeline:

python pipeline/pipeline.py --text "There is a dog in the picture." --image "path_to_image"
