import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 1️⃣ Dataset personalizado para PyTorch
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 2️⃣ Modelo basado en BERT
class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        dropped = self.dropout(pooled_output)
        return self.fc(dropped)

# 3️⃣ Función de entrenamiento y evaluación
def train_and_evaluate_bert(X_train, y_train, X_test, y_test, model_name="bert-base-uncased", epochs=3, batch_size=8, lr=2e-5, pred=False):
    """
    Entrena un modelo basado en BERT y lo evalúa en el conjunto de prueba.
    
    Retorna:
    - accuracy: Precisión del modelo
    - f1: F1-score del modelo
    - bert_model: Modelo BERT entrenado
    - tokenizer: Tokenizador de BERT
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar tokenizer y dataset
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Inicializar modelo
    num_classes = len(set(y_train))  # Número de clases en el dataset
    model = BertClassifier(model_name, num_classes=num_classes).to(device)

    # Configurar optimizador y función de pérdida
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 🔥 Entrenamiento
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # 🎯 Evaluación
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    if pred:
        return None, None, all_preds
    accuracy = np.round(accuracy_score(all_labels, all_preds), 4)
    f1 = np.round(f1_score(all_labels, all_preds, average="weighted"), 4)

    return accuracy, f1, all_preds 