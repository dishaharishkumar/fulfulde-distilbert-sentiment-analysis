# Fulfulde Sentiment Analysis

# Sentiment Analysis using DistilBERT

This project is a sentiment analysis system using DistilBERT. It includes training, testing, and data handling scripts to process sentiment classification for given text inputs.

## Installation

Before running the scripts, install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas
```

## File Structure

```
|-- data.py          # Loads and processes the dataset
|-- train.py         # Trains the model
|-- test.py          # Tests the model
|-- sentences.txt    # Sample sentences for testing
|-- README.md        # Documentation
```

## Data Preparation

Make sure you have a tab-separated dataset (`.tsv`) containing sentences and their corresponding sentiment labels.

## Running the Scripts

### 1. Running `data.py`

To check if the dataset loads properly, run:

```bash
python data.py --file_path /path/to/fulfulde_sentiment_tsv.tsv
```

This will load and split the data into training and validation sets.

### 2. Training the Model

Run the following command to train the model:

```bash
python train.py --file_path /path/to/fulfulde_sentiment_tsv.tsv --output_dir ./distilbert_fulfulde_sentiment --num_train_epochs 3 --batch_size 16 --logging_dir ./logs --logging_steps 10
```

### 3. Testing the Model

To test sentiment predictions on predefined sentences, run:

```bash
python test.py
```

To test on a custom file with sentences, create a `sentences.txt` file and run:

```bash
python test.py --file sentences.txt
```

## Example Sentences for `sentences.txt`

```
Heddugol e jam.
Mi seedi e hoore.
Dewgal ndaa mi.
Aaduna woni dow nyawdi.
Ko fayde makko.
Mi yidi jam.
Kila mi andi ko min mbaawi?
Ko min mbaawi hulataa.
Ndee jamaa woni dow mari.
Mi sali Allah.
```

## Notes

- Ensure the dataset file path is correct when passing as an argument.
- If you encounter errors related to missing dependencies, install them using `pip install`.
- Modify training parameters in `train.py` as needed.

## Troubleshooting

If you face errors like `TypeError: cannot unpack non-iterable NoneType object`, verify that:

- The dataset file exists and contains the required columns (`Sentence`, `Sentiment`).
- `data.py` correctly loads and splits the dataset.
- The training script receives valid arguments.

For further questions, feel free to reach out!



This project performs sentiment analysis on Fulfulde text using DistilBERT.

## Installation
Before running the scripts, install dependencies:


pip install torch transformers datasets scikit-learn pandas
from sklearn.model_selection import train_test_split # Import train_test_split
from datasets import Dataset
