import sys
import os
import pandas as pd
from tqdm import tqdm

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessor and dataset loader
from preprocessing.preprocessor import TextPreprocessor
from data_loaders.fakeandreal_load import load_fakeandreal_dataset

def test_preprocessor():
    print("Loading Fake and Real News dataset...")
    dataset = load_fakeandreal_dataset()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(model_name='bert-base-uncased', max_length=512)  # Increased max_length for full articles
    
    print("\nTesting preprocessing...")
    # Get text, titles, and labels
    titles = dataset['title']
    texts = dataset['text']
    labels = dataset['label']
    
    # Combine title and text for preprocessing
    combined_texts = [f"{title} | {text}" for title, text in zip(titles, texts)]
    
    # Process a small subset first for testing
    test_size = min(100, len(texts))
    processed_data = preprocessor.preprocess_dataset(
        combined_texts[:test_size], 
        labels[:test_size],
        split=True  # This time we'll test the splitting functionality
    )
    
    # Print statistics
    preprocessor.print_stats()
    
    # Print sample outputs for each split
    for split_name, (features, split_labels) in processed_data.items():
        print(f"\nSample preprocessing results for {split_name} split:")
        for i in range(min(3, len(features))):
            original_idx = i  # This is simplified; in reality, we'd need to track the mapping
            print(f"\nOriginal title: {titles[original_idx]}")
            print(f"Original text: {texts[original_idx][:200]}...")  # Show first 200 chars
            print(f"Original label: {labels[original_idx]}")
            print(f"Encoded label: {split_labels[i]}")
            print(f"Cleaned text length: {len(preprocessor.clean_text(combined_texts[original_idx]))}")
            print(f"Tokenized length: {features[i].shape}")

if __name__ == "__main__":
    test_preprocessor() 