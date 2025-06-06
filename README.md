# Arabic Propaganda Detection

This repository contains a machine learning model for detecting propaganda in Arabic text. The model uses AraBERT, a pre-trained Arabic language model, fine-tuned on a propaganda detection task.

## Overview

Propaganda detection is a challenging NLP task that aims to identify manipulative text designed to influence opinions, attitudes, or behaviors. This project implements a binary classification approach to determine whether a given Arabic text paragraph contains propaganda (true) or not (false).

## Dataset

The project uses a combined dataset of Arabic text paragraphs with binary labels:
- **true**: The paragraph contains propaganda
- **false**: The paragraph does not contain propaganda

Dataset statistics:
- Total samples: 8,000
- Propaganda samples (true): 5,034 (62.9%)
- Non-propaganda samples (false): 2,966 (37.1%)
- Train/Validation/Test split: 6002/672/1326

## Preprocessing

The text preprocessing pipeline includes:
- Normalization of Arabic characters
- Removal of diacritics
- Removal of punctuation
- Removal of non-Arabic letters
- Removal of stopwords
- Handling of repeating characters

## Model

The model uses a pre-trained AraBERT model ("Bmalmotairy/arabert-fully-supervised-arabic-propaganda") and fine-tunes it for the propaganda detection task.

### Training

- Batch size: 16
- Epochs: 3
- Optimizer: AdamW (default in Hugging Face Trainer)

## Results

The model achieves the following performance metrics on the test set:

- Accuracy: 77.60%
- Precision (Binary): 81.89%
- Recall (Binary): 83.16%
- F1-score (Binary): 82.52%
- F1-score (Micro): 77.60%
- F1-score (Macro): 75.68%

## Requirements

- Python 3.x
- Torch
- Transformers
- NLTK
- pandas
- scikit-learn
- BeautifulSoup
- emoji

## Usage

The model can be used to detect propaganda in Arabic text by loading the fine-tuned model and tokenizer:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("path_to_fine_tuned_model")
model = AutoModelForSequenceClassification.from_pretrained("path_to_fine_tuned_model")

# Preprocess and tokenize input text
text = "Your Arabic text here"
# Apply preprocessing if needed
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

print("Propaganda" if prediction == 1 else "Not propaganda")
```

## Future Work

- Expand the dataset with more diverse examples
- Experiment with different Arabic language models
- Implement multi-class propaganda technique classification
- Deploy the model as a web service

## Citation

If you use this model in your research, please cite:

```
@misc{arabic_propaganda_detection,
  author = {Begolosamazaref},
  title = {Arabic Propaganda Detection},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Begolosamazaref/Propaganda-Detection}}
}
``` 