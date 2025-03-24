import torch
from model.dataset import FakeNewsDataset

def test_dataset():
    """Test the FakeNewsDataset class with improved functionality"""
    print("Initializing dataset in test mode...")
    dataset = FakeNewsDataset(test_mode=True)
    
    print("\nTesting basic functionality:")
    print(f"Total samples in dataset: {len(dataset)}")
    
    print("\nTesting sample retrieval:")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels']}")
    
    print("\nTesting split functionality:")
    splits = dataset.get_splits()
    print(f"Split sizes: {splits}")
    
    for split in ['train', 'val', 'test']:
        print(f"\nTesting {split} split:")
        dataset.set_split(split)
        print(f"Current split: {dataset.current_split}")
        print(f"Split size: {len(dataset)}")
        
        # Test sample retrieval from this split
        sample = dataset[0]
        print(f"Sample from {split} split:")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")
        print(f"Label: {sample['labels']}")
    
    print("\nTesting data validation:")
    try:
        dataset.validate_data()
        print("Data validation passed successfully")
    except ValueError as e:
        print(f"Data validation failed: {e}")
    
    print("\nTesting error handling:")
    try:
        dataset.set_split('invalid_split')
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    try:
        dataset[999999]  # Try to access an invalid index
    except IndexError as e:
        print(f"Expected error caught: {e}")

if __name__ == "__main__":
    test_dataset()
