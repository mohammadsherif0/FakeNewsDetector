from datasets import load_dataset

dataset = load_dataset("csv", 
                       data_files={
                           "train": "liar_dataset/train.tsv",
                           "test": "liar_dataset/test.tsv",
                           "validation": "liar_dataset/valid.tsv"
                       },
                       delimiter="\t",
                       column_names=[
                           "ID", "label", "statement", "subject", "speaker",
                           "speaker_job_title", "state_info", "party_affiliation",
                           "barely_true_count", "false_count", "half_true_count",
                           "mostly_true_count", "pants_on_fire_count", "context"
                       ]
                      )

print(dataset)
print("First training sample:")
print(dataset["train"][0])