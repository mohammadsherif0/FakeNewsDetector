from transformers import Trainer
from .metrics import compute_metrics
from .config import get_model_config, get_training_args
from .model import FakeNewsClassifier

class FakeNewsTrainer:

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model_config = get_model_config()
        self.training_args = get_training_args()

        self.model = FakeNewsClassifier(self.model_config)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

    def train(self):
        return self.trainer.train()
    
    def evaluate(self, test_dataset=None):
        return self.trainer.evaluate(test_dataset or self.val_dataset)
    
    def save_model(self, path):
        self.model.save_model(path)

    def predict(self, texts):
        return self.trainer.predict(texts)