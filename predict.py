import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import argparse
import torch
from sklearn.model_selection import train_test_split
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df
df=load_data('train.json')
string_to_id = {}
for i, string in enumerate(set(df['subject'])):
    string_to_id[string] = i
class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "label": torch.tensor(string_to_id[self.labels[idx]], dtype=torch.long)
        }

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}
def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)
def main(train_file, valid_file, output_file):
    # Load and preprocess the data
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)

    X_train, X_val,y_train,y_val= train_test_split(train_data['abstract'], train_data['subject'])

    # Initialize the tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(y_train.unique()))

    # Create dataset and dataloader
    train_dataset = ArxivDataset(X_train, y_train, tokenizer, max_length=256)
    valid_dataset=ArxivDataset(X_val,y_val,tokenizer,max_length=256)
    

    # Create the trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Make predictions
    valid_texts = valid_data['abstract']
    valid_dataset = ArxivDataset(valid_texts, [0]*len(valid_texts), tokenizer, max_length=256)
    preds = trainer.predict(valid_dataset)
    predicted_labels = preds.predictions.argmax(axis=-1)

    # Save the predictions
    valid_data['predicted_subject'] = predicted_labels
    save_data(output_file, valid_data.to_dict(orient='records'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.train, args.valid, args.output)
