from model.trainer import FakeNewsTrainer
from model.dataset import FakeNewsDataset

def main():
    dataset = FakeNewsDataset(test_mode=False)  # Use full dataset, not test mode
    
    # Set splits
    dataset.set_split('train')
    train_dataset = dataset
    
    dataset.set_split('val')
    val_dataset = dataset
    
    dataset.set_split('test')
    test_dataset = dataset
    
    trainer = FakeNewsTrainer(train_dataset, val_dataset)
    
    print("Starting training...")
    training_results = trainer.train()
    print(f"Training completed: {training_results}")
    
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    trainer.save_model("./saved_models/fake_news_detector")

if __name__ == "__main__":
    main()