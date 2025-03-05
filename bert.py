from dataloader import load_data
from preprocess import preprocess_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Configuración
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # BERT en español
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
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


def train_bert_kfold(train_data, K=5):
    """Entrena un modelo BERT usando validación cruzada K-Fold y devuelve predicciones."""

    X = train_data["tweet"].tolist()
    y = train_data["label"].tolist()

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    all_predictions = np.zeros(len(y))  # Para almacenar todas las predicciones
    fold_accuracies = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔹 Fold {fold + 1}/{K}")

        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        train_dataset = TweetDataset(X_train, y_train, tokenizer, MAX_LENGTH)
        test_dataset = TweetDataset(X_test, y_test, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Inicializar modelo
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        num_training_steps = len(train_loader) * EPOCHS
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Entrenamiento
        for epoch in range(EPOCHS):
            print(f"\n🟢 Epoch {epoch + 1}/{EPOCHS} - Fold {fold + 1}")
            model.train()

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Evaluación en test set del fold
        model.eval()
        fold_predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                fold_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(batch["labels"].cpu().numpy())

        # Calcular precisión para este fold
        accuracy = accuracy_score(true_labels, fold_predictions)
        fold_accuracies.append(accuracy)
        print(f"✅ Accuracy del Fold {fold + 1}: {accuracy:.4f}")

        # Guardar predicciones en la posición correcta
        for i, idx in enumerate(test_idx):
            all_predictions[idx] = fold_predictions[i]

    # Promedio de accuracy en todos los folds
    mean_accuracy = np.mean(fold_accuracies)
    print(f"\n🔹 Accuracy promedio en {K}-Fold: {mean_accuracy:.4f}")

    return all_predictions, mean_accuracy
