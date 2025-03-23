import sys
import os
import pandas as pd
from tqdm import tqdm

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessor and dataset loader
from preprocessing.preprocessor import TextPreprocessor
from data_loaders.fakenewsnet_load import load_fakenewsnet_dataset

def test_preprocessor():
    print("Loading FakeNewsNet dataset...")
    dataset = load_fakenewsnet_dataset()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(model_name='bert-base-uncased', max_length=128)
    
    # Process each split
    for split_name in dataset.keys():
        print(f"\nTesting preprocessing on {split_name} split...")
        split_data = dataset[split_name]
        
        # Get text and labels (using 'title' instead of 'text')
        texts = split_data['title']
        # For FakeNewsNet, label is determined by the split name
        labels = ['fake' if 'fake' in split_name.lower() else 'real'] * len(texts)
        
        # Process a small subset first for testing
        test_size = min(100, len(texts))
        processed_data = preprocessor.preprocess_dataset(
            texts[:test_size], 
            labels[:test_size],
            split=False
        )
        
        # Print statistics
        print(f"\nStatistics for {split_name}:")
        preprocessor.print_stats()
        
        # Print sample outputs
        print(f"\nSample preprocessing results for {split_name}:")
        for i in range(min(3, test_size)):
            print(f"\nOriginal text: {texts[i]}")  # Titles are usually short, so no need to truncate
            print(f"Original label: {labels[i]}")
            print(f"Cleaned text length: {len(preprocessor.clean_text(texts[i]))}")
            print(f"Tokenized length: {processed_data[0]['input_ids'][i].shape}")
        
        # Reset statistics for next split
        preprocessor.stats = {
            'original_lengths': [],
            'cleaned_lengths': [],
            'dropped_samples': 0,
            'label_distribution': {},
            'empty_texts': 0
        }

if __name__ == "__main__":
    test_preprocessor() 