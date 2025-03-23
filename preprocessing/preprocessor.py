import re
import nltk
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Try to download NLTK data, but continue even if it fails
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except:
    print("Warning: Could not download NLTK data. Continuing without stopword removal.")
    STOP_WORDS = set()

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.stop_words = STOP_WORDS
        
        # Statistics for validation
        self.stats = {
            'original_lengths': [],
            'cleaned_lengths': [],
            'dropped_samples': 0,
            'label_distribution': {},
            'empty_texts': 0
        }

    def clean_text(self, text):
        """Clean text with validation checks."""
        if not isinstance(text, str):
            self.stats['dropped_samples'] += 1
            return ""

        original_length = len(text)
        self.stats['original_lengths'].append(original_length)

        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        text = ' '.join([word for word in words if word not in self.stop_words])

        cleaned_length = len(text)
        self.stats['cleaned_lengths'].append(cleaned_length)

        if cleaned_length == 0:
            self.stats['empty_texts'] += 1

        return text

    def tokenize(self, texts):
        """Tokenize texts with validation."""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Validate tokenization
        if tokenized['input_ids'].shape[1] > self.max_length:
            raise ValueError(f"Tokenized sequence length exceeds max_length {self.max_length}")

        return tokenized

    def encode_labels(self, labels):
        """Encode labels with validation."""
        encoded = []
        for label in labels:
            if isinstance(label, (int, float, np.integer)):
                encoded.append(int(label))
            elif isinstance(label, str):
                # Binary encoding: fake=0, real=1
                if label.lower() in ['fake', 'false', 'pants-fire', 'barely-true']:
                    encoded.append(0)
                elif label.lower() in ['real', 'true', 'mostly-true']:
                    encoded.append(1)
                else:
                    self.stats['dropped_samples'] += 1
                    encoded.append(0)  # Default to fake for unknown labels
            
            # Update label distribution statistics
            self.stats['label_distribution'][encoded[-1]] = \
                self.stats['label_distribution'].get(encoded[-1], 0) + 1

        return encoded

    def preprocess_dataset(self, texts, labels, split=True):
        """Preprocess dataset with validation checks."""
        print("Starting preprocessing...")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in tqdm(texts, desc="Cleaning texts")]
        
        # Remove empty samples
        valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 0]
        cleaned_texts = [cleaned_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        # Tokenize
        print("Tokenizing texts...")
        tokenized = self.tokenize(cleaned_texts)
        
        # Encode labels
        print("Encoding labels...")
        encoded_labels = self.encode_labels(labels)
        
        if split:
            return self.split_data(tokenized, encoded_labels)
        return tokenized, encoded_labels

    def split_data(self, features, labels, test_size=0.2, val_size=0.25):
        """Split data with validation checks."""
        if len(features['input_ids']) != len(labels):
            raise ValueError("Features and labels must have the same length")

        # First split: train + validation, test
        train_val_f, test_f, train_val_l, test_l = train_test_split(
            features['input_ids'], labels, test_size=test_size, random_state=42
        )

        # Second split: train, validation
        train_f, val_f, train_l, val_l = train_test_split(
            train_val_f, train_val_l, test_size=val_size, random_state=42
        )

        return {
            'train': (train_f, train_l),
            'validation': (val_f, val_l),
            'test': (test_f, test_l)
        }

    def print_stats(self):
        """Print preprocessing statistics."""
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