import argparse
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load model and tokenizer
MODEL_PATH = "./distilbert_fulfulde_sentiment"

try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def predict_sentiment(sentence):
    """Predict sentiment for a single sentence."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
    return sentiment_map.get(prediction, "Unknown")

def batch_predict(sentences):
    """Predict sentiment for a list of sentences."""
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    sentiment_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
    return [sentiment_map.get(pred, "Unknown") for pred in predictions]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment using a trained model.")

    # Add options to provide either a single sentence or a file of sentences
    parser.add_argument("--sentence", type=str, help="Single sentence to predict sentiment")
    parser.add_argument("--file", type=str, help="Path to a text file containing sentences (one per line)")

    args = parser.parse_args()

    if args.sentence:
        prediction = predict_sentiment(args.sentence)
        print(f"Sentence: {args.sentence} --> Sentiment: {prediction}")

    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f.readlines() if line.strip()]
            predictions = batch_predict(sentences)
            for sent, pred in zip(sentences, predictions):
                print(f"Sentence: {sent} --> Sentiment: {pred}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print("❌ Please provide either --sentence or --file as input.")
