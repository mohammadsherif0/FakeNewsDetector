from datasets import load_dataset
import pandas as pd

def load_fakenewspred_dataset():
    # Load the CSV file using pandas with proper quoting
    df = pd.read_csv(
        "FakeNewsPrediction_dataset/news.csv",
        quoting=1,  # QUOTE_ALL
        escapechar='\\'
    )
    
    # Normalize labels to numeric values: 0 for fake, 1 for real
    df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'fake' else 1)
    
    # Print unique labels to verify
    print("Unique labels in dataset:", df['label'].unique())
    
    # Save the properly formatted CSV
    df.to_csv("FakeNewsPrediction_dataset/news_fixed.csv", index=False)
    
    # Convert to HuggingFace dataset
    dataset = load_dataset(
        "csv",
        data_files={"data": "FakeNewsPrediction_dataset/news_fixed.csv"},
        delimiter=","
    )
    
    return dataset

if __name__ == "__main__":
    dataset = load_fakenewspred_dataset()
    print("\nDataset structure:")
    print(dataset)
    print("\nFirst sample:")
    print(dataset["data"][0])
