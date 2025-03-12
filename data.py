# data.py
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import argparse


def load_data(file_path):
    try:
        # Read the dataset
        df = pd.read_csv(file_path, sep="\t", encoding='latin-1', on_bad_lines='skip')

        if df is None or df.empty:
            print("Error: Loaded data is empty or None.")
            return [], [], [], []  # Return empty lists to prevent unpacking errors

        # Ensure required columns exist
        if "Sentence" not in df.columns or "Sentiment" not in df.columns:
            print("Error: Missing required columns 'Sentence' and 'Sentiment'.")
            return [], [], [], []

        # Map sentiment labels
        label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
        df["Sentiment"] = df["Sentiment"].map(label_map)

        # Drop rows with missing values
        df.dropna(subset=["Sentence", "Sentiment"], inplace=True)

        # Split dataset into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["Sentence"].tolist(), df["Sentiment"].tolist(), test_size=0.2, random_state=42
        )

        return train_texts, val_texts, train_labels, val_labels  # Ensure function returns a tuple

    except Exception as e:
        print(f"Error loading data: {e}")
        return [], [], [], []  # Prevent NoneType error


def tokenize_data(train_texts, val_texts, train_labels, val_labels):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess data from a TSV file.")

    # Add file path argument
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input TSV file")

    args = parser.parse_args()

    # Run the load_data function
    train_texts, val_texts, train_labels, val_labels = load_data(args.file_path)

    # Check if data loaded successfully
    if train_texts:
        print(f"✅ Data loaded successfully! \nTrain samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    else:
        print("❌ Error: No valid data found. Please check your input file.")


#python data.py --file_path "/path/to/fulfulde_sentiment_tsv.tsv"

