from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

from preprocessing.preprocessor import TextPreprocessor
from data_loaders.fakenewspred_load import load_fakenewspred_dataset
from data_loaders.fakeandreal_load import load_fakeandreal_dataset
from data_loaders.fakenewsnet_load import load_fakenewsnet_dataset
from data_loaders.liar_load import load_liar_dataset


class FakeNewsDataset(Dataset):
    """
    A PyTorch Dataset class for fake news detection that combines multiple datasets.
    
    This class loads and preprocesses data from multiple sources (LIAR, Fake and Real News,
    FakeNewsNet, and Fake News Prediction) and provides a unified interface for training.
    
    Attributes:
        preprocessor (TextPreprocessor): Text preprocessing utility
        tokenizer (AutoTokenizer): Hugging Face tokenizer for text tokenization
        splits (dict): Dictionary containing train, validation, and test splits
        current_split (str): Currently active split ('train', 'val', or 'test')
    """
    
    def __init__(self, test_mode=False):
        """
        Initialize the dataset.
        
        Args:
            test_mode (bool): If True, only load 100 samples from each dataset for testing
        """
        print("Initializing dataset in test mode..." if test_mode else "Initializing dataset...")
        self.test_mode = test_mode
        test_size = 100 if test_mode else None
        
        # Initialize preprocessor and tokenizer
        print("Initializing preprocessor...")
        self.preprocessor = TextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize data storage
        self.texts = []
        self.labels = []
        
        # Load and combine datasets
        print("Loading LIAR dataset...")
        liar_dataset = load_liar_dataset()
        # LIAR dataset is already split, we'll use the train split for now
        liar_texts = liar_dataset['train']['statement'][:test_size]
        liar_labels = liar_dataset['train']['label'][:test_size]
        self.texts.extend(liar_texts)
        self.labels.extend(liar_labels)
        
        print("Loading Fake and Real News dataset...")
        fakeandreal_dataset = load_fakeandreal_dataset()
        for i in tqdm(range(len(fakeandreal_dataset['title'][:test_size])), desc="Processing Fake and Real News"):
            combined_text = f"{fakeandreal_dataset['title'][i]} | {fakeandreal_dataset['text'][i]}"
            self.texts.append(combined_text)
            self.labels.append(fakeandreal_dataset['label'][i])
        
        print("Loading FakeNewsNet dataset...")
        fakenewsnet_dataset = load_fakenewsnet_dataset()
        for split_name in ['politifact_fake', 'politifact_real', 'gossipcop_fake', 'gossipcop_real']:
            split_data = fakenewsnet_dataset[split_name]
            texts = split_data['title'][:test_size]
            self.texts.extend(texts)
            label = 1 if 'real' in split_name else 0
            self.labels.extend([label] * len(texts))
        
        print("Loading Fake News Prediction dataset...")
        fakenewspred_dataset = load_fakenewspred_dataset()
        data = fakenewspred_dataset['data']
        for i in tqdm(range(len(data['title'][:test_size])), desc="Processing Fake News Prediction"):
            combined_text = f"{data['title'][i]} | {data['text'][i]}"
            self.texts.append(combined_text)
            self.labels.append(data['label'][i])
        
        # Preprocess all texts
        print("Preprocessing all texts...")
        processed_data = self.preprocessor.preprocess_dataset(self.texts, self.labels, split=True)
        
        # Initialize splits dictionary
        self.splits = {
            'train': {
                'features': processed_data['train'][0],
                'attention_mask': torch.ones_like(processed_data['train'][0]),
                'labels': processed_data['train'][1]
            },
            'val': {
                'features': processed_data['validation'][0],
                'attention_mask': torch.ones_like(processed_data['validation'][0]),
                'labels': processed_data['validation'][1]
            },
            'test': {
                'features': processed_data['test'][0],
                'attention_mask': torch.ones_like(processed_data['test'][0]),
                'labels': processed_data['test'][1]
            }
        }
        
        # Set default split to train
        self.current_split = 'train'
        
        # Validate data consistency
        self.validate_data()
        
        print(f"Dataset initialized with {len(self.texts)} samples")
    
    def validate_data(self):
        """
        Ensures data consistency across all splits.
        
        Raises:
            ValueError: If there are inconsistencies in data lengths across splits
        """
        for split_name, split_data in self.splits.items():
            features_len = len(split_data['features'])
            labels_len = len(split_data['labels'])
            mask_len = len(split_data['attention_mask'])
            
            if not (features_len == labels_len == mask_len):
                raise ValueError(
                    f"Inconsistent lengths in {split_name} split: "
                    f"features={features_len}, labels={labels_len}, mask={mask_len}"
                )
    
    def set_split(self, split):
        """
        Set the current split for data access.
        
        Args:
            split (str): Split name ('train', 'val', or 'test')
            
        Raises:
            ValueError: If split name is invalid
        """
        if split not in self.splits:
            raise ValueError(f"Split must be one of {list(self.splits.keys())}")
        self.current_split = split
    
    def __len__(self):
        """
        Get the length of the current split.
        
        Returns:
            int: Number of samples in the current split
        """
        return len(self.splits[self.current_split]['features'])
    
    def __getitem__(self, idx):
        """
        Get a sample from the current split.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
            
        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If there's an error accessing the data
        """
        try:
            split_data = self.splits[self.current_split]
            if idx >= len(split_data['features']):
                raise IndexError(f"Index {idx} out of bounds for split {self.current_split}")
            
            return {
                'input_ids': split_data['features'][idx],
                'attention_mask': split_data['attention_mask'][idx],
                'labels': split_data['labels'][idx]
            }
        except Exception as e:
            raise RuntimeError(f"Error accessing item {idx} in split {self.current_split}: {str(e)}")
    
    def get_splits(self):
        """
        Get the sizes of all splits.
        
        Returns:
            dict: Dictionary containing the size of each split
        """
        return {
            split: len(split_data['features'])
            for split, split_data in self.splits.items()
        }
        