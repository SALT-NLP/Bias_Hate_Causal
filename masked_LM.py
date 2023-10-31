import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import numpy as np
import preprocess_datasets
from datasets import Dataset,concatenate_datasets
from transformers import default_data_collator
import random
from transformers import DataCollatorForLanguageModeling
from datasets import load_metric
import json
import collections
import datasets
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
import math
from sklearn.metrics import accuracy_score,f1_score


def main(args):
    all_train_data, all_val_data = preprocess_datasets(args.datasets)
    device = args.device
    model_checkpoint = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to(device)

    all_train_data = concatenate_datasets(all_train_data)
    all_train_data = all_train_data.shuffle(seed = 42)
    all_val_data =  concatenate_datasets(all_val_data)
    all_val_data = all_val_data.shuffle(seed = 42)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        push_to_hub=False,
        fp16=args.fp16,
        logging_steps=len(all_train_data) // args.logging_divisor,
        num_train_epochs=args.num_epochs,
        save_strategy=args.save_strategy)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=all_train_data,
    eval_dataset=all_val_data,
    data_collator=data_collator)
    trainer.train()
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        type=str,
                        help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--model_checkpoint",
                        default="bert-base-cased",
                        type=str,
                        help="Model checkpoint to use.")
    parser.add_argument("--output_dir",
                        default="MLM_large_corpus",
                        type=str,
                        help="Output directory for trained model.")
    parser.add_argument("--evaluation_strategy",
                        default="epoch",
                        type=str,
                        help="Evaluation strategy.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Training batch size.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Evaluation batch size.")
    parser.add_argument("--fp16",
                        default=False,
                        action="store_true",
                        help="Use half precision training.")
    parser.add_argument("--datasets",
                        default="",
                        type=str,
                        help="Comma-separated list of dataset names to use.")
    parser.add_argument(
        "--logging_divisor",
        default=4,
        type=int,
        help=
        "How often to log. Will log every len(train_data) // logging_divisor steps."
    )
    parser.add_argument("--num_epochs",
                        default=5,
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument("--save_strategy",
                        default="epoch",
                        type=str,
                        help="Strategy to save the model.")

    args = parser.parse_args()
    main(args)
