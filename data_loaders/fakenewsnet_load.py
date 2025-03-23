from datasets import load_dataset

def load_fakenewsnet_dataset():
    dataset = load_dataset("csv",
                        data_files={
                            "politifact_fake": "FakeNewsNet_dataset/politifact_fake.csv",
                            "politifact_real": "FakeNewsNet_dataset/politifact_real.csv",
                            "gossipcop_fake": "FakeNewsNet_dataset/gossipcop_fake.csv",
                            "gossipcop_real": "FakeNewsNet_dataset/gossipcop_real.csv"
                        },
                        delimiter=",")
    return dataset

if __name__ == "__main__":
    dataset = load_fakenewsnet_dataset()
    for key in dataset.keys():
        print(f"\nDataset split: {key}")
        print("Number of samples:", len(dataset[key]))
        print("First sample:")
        print(dataset[key][0])