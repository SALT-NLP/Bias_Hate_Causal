# Importing necessary libraries
import os
import random
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import wandb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)

# Setting random seed for reproducibility
def set_random_seed(seed):
    """
    Sets the seed for random number generation to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom dataset class for handling the input data
class CustomDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for preparing data for the model.
    """
    def __init__(self, encodings, labels):
        self.encodings = {
            'input_ids': encodings['input_ids'],
            'token_type_ids': encodings['token_type_ids'],
            'attention_mask': encodings['attention_mask']
        }
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to split dataset into training, validation, and testing sets
def split_dataset(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, if_split, tokenizer):
    """
    Splits the dataset into training, validation, and testing sets.
    """
    if if_split:
        train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.2)
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.25)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset

# Function to load and process text data
def txt_data_process(name):
    """
    Processes text data from files for training, validation, and testing.
    """
    # Function for reading data from a file
    def read_data(file_path):
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                label, sentence = line.split('\t')
                texts.append(sentence.strip())
                labels.append(int(label))
        return texts, labels
    
    text_train, label_train = read_data(f'Data_txt/{name}/{name}_train_change.txt')
    text_val, label_val = read_data(f'Data_txt/{name}/{name}_val.txt')
    text_test, label_test = read_data(f'Data_txt/{name}/{name}_test.txt')

    return text_train, label_train, text_val, label_val, text_test, label_test

# Main function
def main():
    # Argument parsing for command line inputs
    import argparse
    parser = argparse.ArgumentParser(description="Hate Speech Classification")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--bz", type=int, default=10)
    parser.add_argument("--data", type=str, default="AHS")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--norm", type=float, default=0.8)
    parser.add_argument('--gpu', default='6', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--model_checkpoint", type=str, default="bert-base-cased", help="Model checkpoint for the tokenizer and model. You can change it to the model after MTI to get the best performance.")
    args = parser.parse_args()

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="hate_speech_detection", config=args)

    # Setting the GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setting the random seed
    set_random_seed(args.seed)

    # Loading the tokenizer and the model using the model checkpoint from args
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    # Processing text data
    text_train, label_train, text_val, label_val, text_test, label_test = txt_data_process(name=args.data)

    # Splitting the dataset
    train_data, val_data, test_data = split_dataset(
        text_train, label_train, text_val, label_val, text_test, label_test, if_split=True, tokenizer=tokenizer
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="multi_output",
        evaluation_strategy="epoch",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.bz,
        per_device_eval_batch_size=args.bz,
        save_strategy='no',
        learning_rate=args.lr,
        weight_decay=args.wd,
        max_grad_norm=args.norm
    )

    # Initializing the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda eval_pred: {
            'accuracy': accuracy_score(eval_pred[1], np.argmax(eval_pred[0], axis=-1)),
            'micro-F1': f1_score(eval_pred[1], np.argmax(eval_pred[0], axis=-1), average='micro'),
            'macro-F1': f1_score(eval_pred[1], np.argmax(eval_pred[0], axis=-1), average='macro')
        }
    )

    # Training the model
    trainer.train()

    # Evaluating the model on the test dataset
    test_output = trainer.evaluate(test_data, metric_key_prefix='test')
    print('Results on test set:', test_output)

if __name__ == "__main__":
    main()
