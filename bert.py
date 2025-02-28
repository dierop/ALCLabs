from dataloader import load_data
from preprocess import preprocess_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # BERT en español
MAX_LENGTH = 128  
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class TweetDataset(Dataset):
    """Dataset para tweets con BERT."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_bert(train_data_path):
    """Carga los datos y entrena BERT para clasificación de tweets."""
    train_data = load_data(train_data_path)
    train_data['tweet'] = train_data['tweet'].apply(preprocess_data)

    X = train_data["tweet"].tolist()
    y = train_data["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TweetDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = TweetDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        print(f"\n🔹 Epoch {epoch + 1}/{EPOCHS}")
        model.train()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n✅ Accuracy de BERT: {accuracy:.4f}")

    return model, accuracy