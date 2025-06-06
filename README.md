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
