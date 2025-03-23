from datasets import load_dataset

def load_fakenewspred_dataset():
    dataset = load_dataset(
        "csv",
        data_files={"data": "FakeNewsPrediction_dataset/news.csv"},
        delimiter=","
    )
    return dataset

if __name__ == "__main__":
    dataset = load_fakenewspred_dataset()
    print(dataset)
    print("First sample:")
    print(dataset["data"][0])
