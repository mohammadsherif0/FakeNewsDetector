from datasets import load_dataset, concatenate_datasets

def load_fakeandreal_dataset():
    # Load the Fake.csv file
    fake_dataset = load_dataset("csv", data_files={"train": "FakeandReal_dataset/Fake.csv"}, delimiter=",")

    # Load the True.csv file
    true_dataset = load_dataset("csv", data_files={"train": "FakeandReal_dataset/True.csv"}, delimiter=",")

    # Add label column: 0 for fake news, 1 for true news
    fake_dataset = fake_dataset["train"].map(lambda example: {"label": 0})
    true_dataset = true_dataset["train"].map(lambda example: {"label": 1})

    # Combine datasets
    combined_dataset = concatenate_datasets([fake_dataset, true_dataset])
    
    return combined_dataset

if __name__ == "__main__":
    dataset = load_fakeandreal_dataset()
    print(dataset)
    print("First sample from combined dataset:", dataset[0])
