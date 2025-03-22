from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files={"data": "FakeNewsPrediction_dataset/news.csv"},
    delimiter=","
)

print(dataset)
print("First sample:")
print(dataset["data"][0])
