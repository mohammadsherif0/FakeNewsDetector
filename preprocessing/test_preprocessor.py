import sys
import os
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessor and dataset loader
from preprocessing.preprocessor import TextPreprocessor
from data_loaders.liar_load import load_liar_dataset

def test_preprocessor():
    print("Loading LIAR dataset...")
    dataset = load_liar_dataset()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(model_name='bert-base-uncased', max_length=128)
    
    print("\nTesting preprocessing on training set...")
    # Get text and labels from training set
    train_data = dataset['train']
    texts = train_data['statement']
    labels = train_data['label']
    
    # Process a small subset first for testing
    test_size = min(100, len(texts))
    processed_data = preprocessor.preprocess_dataset(
        texts[:test_size], 
        labels[:test_size],
        split=False
    )
    
    # Print statistics
    preprocessor.print_stats()
    
    # Print sample outputs
    print("\nSample preprocessing results:")
    for i in range(min(3, test_size)):
        print(f"\nOriginal text: {texts[i]}")
        print(f"Original label: {labels[i]}")
        print(f"Cleaned text length: {len(preprocessor.clean_text(texts[i]))}")
        print(f"Tokenized length: {processed_data[0]['input_ids'][i].shape}")

if __name__ == "__main__":
    test_preprocessor() 