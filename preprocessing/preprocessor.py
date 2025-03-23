import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.stats = {
            'original_lengths': [],
            'cleaned_lengths': [],
            'dropped_samples': 0,
            'label_distribution': {},
            'empty_texts': 0
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            self.stats['dropped_samples'] += 1
            return ""

        original_length = len(text)
        self.stats['original_lengths'].append(original_length)

        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())

        cleaned_length = len(text)
        self.stats['cleaned_lengths'].append(cleaned_length)

        if cleaned_length == 0:
            self.stats['empty_texts'] += 1

        return text

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return tokenized

    def encode_labels(self, labels):
        encoded = []
        for label in labels:
            if isinstance(label, (int, float, np.integer)):
                encoded.append(int(label))
            elif isinstance(label, str):
                if label.lower() in ['fake', 'false', 'pants-fire', 'barely-true']:
                    encoded.append(0)
                elif label.lower() in ['real', 'true', 'mostly-true']:
                    encoded.append(1)
                else:
                    self.stats['dropped_samples'] += 1
                    encoded.append(0)
            
            self.stats['label_distribution'][encoded[-1]] = \
                self.stats['label_distribution'].get(encoded[-1], 0) + 1

        return encoded

    def preprocess_dataset(self, texts, labels, split=True):
        print("Starting preprocessing...")
        
        cleaned_texts = [self.clean_text(text) for text in tqdm(texts, desc="Cleaning texts")]
        
        valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 0]
        cleaned_texts = [cleaned_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        print("Tokenizing texts...")
        tokenized = self.tokenize(cleaned_texts)
        
        print("Encoding labels...")
        encoded_labels = self.encode_labels(labels)
        
        if split:
            return self.split_data(tokenized, encoded_labels)
        return tokenized, encoded_labels

    def split_data(self, features, labels, test_size=0.2, val_size=0.25):
        if len(features['input_ids']) != len(labels):
            raise ValueError("Features and labels must have the same length")

        train_val_f, test_f, train_val_l, test_l = train_test_split(
            features['input_ids'], labels, test_size=test_size, random_state=42
        )

        train_f, val_f, train_l, val_l = train_test_split(
            train_val_f, train_val_l, test_size=val_size, random_state=42
        )

        return {
            'train': (train_f, train_l),
            'validation': (val_f, val_l),
            'test': (test_f, test_l)
        }

    def print_stats(self):
        print("\nPreprocessing Statistics:")
        print(f"Total samples processed: {len(self.stats['original_lengths'])}")
        print(f"Samples dropped: {self.stats['dropped_samples']}")
        print(f"Empty texts after cleaning: {self.stats['empty_texts']}")
        print("\nText length statistics:")
        print(f"Average original length: {np.mean(self.stats['original_lengths']):.2f}")
        print(f"Average cleaned length: {np.mean(self.stats['cleaned_lengths']):.2f}")
        print("\nLabel distribution:")
        for label, count in self.stats['label_distribution'].items():
            print(f"Label {label}: {count} samples") 