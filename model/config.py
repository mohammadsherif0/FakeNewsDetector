from transformers import BertConfig, TrainingArguments

def get_model_config():

    return BertConfig(
        hidden_size=768,
        num_hidden_layers=12,         
        num_attention_heads=12,       
        intermediate_size=3072,       
        hidden_dropout_prob=0.1,      
        attention_probs_dropout_prob=0.1,  
        max_position_embeddings=512,  
        type_vocab_size=2,           
        initializer_range=0.02,
    )

# def get_training_args():
#     return TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=64,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         greater_is_better=True
#     )

def get_training_args():
    return TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  # Reduced from 3
        per_device_train_batch_size=4,  # Reduced from 16
        per_device_eval_batch_size=8,  # Reduced from 64
        warmup_steps=100,  # Reduced from 500
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"  # Explicitly disable wandb
    )