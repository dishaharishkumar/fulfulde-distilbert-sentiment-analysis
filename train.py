# newtrain.py
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from data import load_data, tokenize_data
import argparse
import os

def main(args):
    # Load data
    train_texts, val_texts, train_labels, val_labels = load_data(args.file_path)

    # Check if data is empty
    if not train_texts:
        print("Error: No valid training data found. Please check the input file.")
        return

    # Tokenize data
    train_dataset, val_dataset, tokenizer = tokenize_data(train_texts, val_texts, train_labels, val_labels)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training completed and model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DistilBERT model for sentiment analysis.")

    # File path argument
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input file")

    # Training arguments as command-line options
    parser.add_argument("--output_dir", type=str, default="./distilbert_fulfulde_sentiment", help="Output directory for model checkpoints")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps interval")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: The file '{args.file_path}' does not exist.")
    else:
        print(f"Processing file: {args.file_path}")
        main(args)
