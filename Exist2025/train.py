from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, log_loss, accuracy_score
)
from datasets import Dataset

import torch
import numpy as np

from dataloader import load_data_json
from typing import List, Dict, Any
import json

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, log_loss
from datasets import Dataset
from collections import Counter, defaultdict
import torch, numpy as np, pandas as pd

class BertTrainerWrapper:
    def __init__(self, df=None, label_name="label1", soft=False,
                 model_name="bert-base-multilingual-cased"):
        if df is None:
            df = load_data_json("data/EXIST2025_training_videos.json", soft=soft)
        self.df = df.dropna(subset=[label_name]).reset_index(drop=True)
        self.label_name = label_name
        self.soft = soft
        self.model_name = model_name

        self.label_encoder = None
        self.keys = None
        self.num_labels = None
        self.problem_type = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.dataset = None
        self.trainer = None

    # ------------------------------------------------------------------
    # 1. PREPARAR ETIQUETAS
    # ------------------------------------------------------------------
    def prepare_labels(self, threshold=0.5):
        col = self.df[self.label_name]

        # -------- SOFT -------------------------------------------------
        if self.soft:
            # dict → vector
            if isinstance(col.iloc[0], dict):
                self.keys = sorted(set().union(*col.apply(lambda x: x.keys())))
                self.df["labels"] = col.apply(
                    lambda d: [d.get(k, 0.0) for k in self.keys])
                self.num_labels = len(self.keys)
                self.problem_type = "multi_label_classification"

            # float → regresión binaria
            else:
                self.df["labels"] = col.astype(float)
                self.num_labels = 1
                self.problem_type = "regression"

            return  # fin SOFT

        # -------- HARD -------------------------------------------------
        # HARD MULTI-LABEL  (lista de etiquetas)
        if isinstance(col.iloc[0], (list, set)):
            # multi-label hard  → vector multihot float
            self.keys = sorted(set().union(*col))

            def to_multihot(lst):
                return [float(1) if k in lst else float(0) for k in self.keys]  # 🆕 float

            self.df["labels"] = col.apply(to_multihot)
            self.num_labels   = len(self.keys)
            self.problem_type = "multi_label_classification"

        # HARD SINGLE-LABEL  (cadena)
        else:
            self.label_encoder = LabelEncoder()
            self.df["labels"] = self.label_encoder.fit_transform(col)
            self.num_labels = len(self.label_encoder.classes_)
            self.problem_type = "single_label_classification"

    # ------------------------------------------------------------------
    # 2. TOKENIZAR
    # ------------------------------------------------------------------
    def tokenize_dataset(self):
        ds = Dataset.from_pandas(self.df[["text", "labels"]])
        ds = ds.train_test_split(test_size=0.15, shuffle=True)
        ds = ds.map(lambda x: self.tokenizer(
            x["text"], truncation=True, padding="max_length",
            max_length=128), batched=False)
        self.dataset = ds

    # ------------------------------------------------------------------
    # 3. CONSTRUIR MODELO
    # ------------------------------------------------------------------
    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type=self.problem_type,
            ignore_mismatched_sizes=True
        )

    # ------------------------------------------------------------------
    # 4. ENTRENAR
    # ------------------------------------------------------------------
    def train(self, epochs=3, batch_size=8, lr=2e-5, eval_batch_size=2, 
              eval_accumulation_steps=32):
        self.prepare_labels()
        self.tokenize_dataset()
        self.build_model()

        args = TrainingArguments(
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            per_device_eval_batch_size = eval_batch_size,
            eval_accumulation_steps    = eval_accumulation_steps,
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

    # ------------------------------------------------------------------
    # 5. MÉTRICAS
    # ------------------------------------------------------------------
    def compute_metrics(self, eval_pred, th=0.5):
        logits, labels = eval_pred

        # ---------------- SOFT ----------------------------------------
        if self.soft:
            probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()

            # Soft binario
            if self.num_labels == 1:
                y_true = labels.squeeze()
                y_pred = probs.squeeze()
                y_true_h = (y_true >= th).astype(int)
                y_pred_h = (y_pred >= th).astype(int)

                return {
                    "mae":     float(np.mean(np.abs(y_pred - y_true))),
                    "logloss": float(log_loss(y_true_h, y_pred)),
                    "f1_micro": float(f1_score(y_true_h, y_pred_h, average="micro")),
                    "f1_macro": float(f1_score(y_true_h, y_pred_h, average="macro")),
                }

            # Soft multilabel
            y_true = labels
            y_pred = probs
            y_true_h = (y_true >= th).astype(int)
            y_pred_h = (y_pred >= th).astype(int)

            ce = np.mean([
                log_loss(y_true_h[:, k], y_pred[:, k])
                for k in range(self.num_labels)
            ])
            micro_f1 = f1_score(
                y_true_h.reshape(-1), y_pred_h.reshape(-1),
                average="micro", zero_division=0)
            macro_f1 = f1_score(
                y_true_h, y_pred_h, average="macro", zero_division=0)

            return {"mae": float(np.mean(np.abs(y_pred - y_true))),
                    "logloss": float(ce),
                    "f1_micro": float(micro_f1),
                    "f1_macro": float(macro_f1)}

        # ---------------- HARD ----------------------------------------
        if self.problem_type == "multi_label_classification":
            probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
            y_true = labels
            y_pred = (probs >= th).astype(int)

            micro_f1 = f1_score(
                y_true.reshape(-1), y_pred.reshape(-1),
                average="micro", zero_division=0)
            macro_f1 = f1_score(
                y_true, y_pred, average="macro", zero_division=0)

            return {"f1_micro": float(micro_f1), "f1_macro": float(macro_f1)}

        # HARD single-label
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(labels, preds))}

    # ------------------------------------------------------------------
    # 6. PREDICT FLEXIBLE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, texts, threshold=0.5, return_probabilities=True):
        if isinstance(texts, str):
            texts = [texts]

        toks = self.tokenizer(
            texts, return_tensors="pt",
            truncation=True, padding="max_length", max_length=128
        )
        device = next(self.model.parameters()).device
        toks = {k: v.to(device) for k, v in toks.items()}
        self.model.eval()

        logits = self.model(**toks).logits

        # ---------- soft ---------------------------------------------
        if self.soft:
            # binario soft
            if self.num_labels == 1:
                probs = torch.sigmoid(logits).squeeze().cpu().tolist()
                return probs if return_probabilities else [
                    int(p >= threshold) for p in probs
                ]
            # multilabel soft
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs = []
            for vec in probs:
                d = {k: float(v) for k, v in zip(self.keys, vec)}
                if return_probabilities:
                    outputs.append(d)
                else:
                    sel = [k for k, p in d.items() if p >= threshold] or \
                          [max(d, key=d.get)]
                    outputs.append(sel)
            return outputs

        # ---------- hard ---------------------------------------------
        if self.problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs = []
            for vec in probs:
                labels = [k for k, v in zip(self.keys, vec) if v >= threshold]
                outputs.append(labels or [self.keys[np.argmax(vec)]])
            return outputs
        else:
            pred_ids = logits.argmax(-1).cpu().numpy()
            return self.label_encoder.inverse_transform(pred_ids)


    def build_submission(
        self,
        threshold: float = 0.5,
        test_case: str = "EXIST2025",
        return_probabilities: bool = True,
        model_name: str = None,
        task_name: str = "task3_3",
        team_name: str = "ScalaR",
        run:int=1,
    ) -> List[Dict[str, Any]]:
        """
        label_name ..... 'label1', 'label2' o 'label3'
        threshold ...... umbral para convertir prob→etiqueta (solo hard multietiqueta)
        test_case ...... literal a incluir en cada JSON
        return_probabilities
            soft=True  -> dict de probabilidades (como tus ejemplos) si True,
                        lista de etiquetas si False
            soft=False -> se ignora; siempre lista/str de etiquetas duras
        """
        # ------------------------------------------------------
        # 1. Normalizar samples a lista ordenada [ {"id":..,"text":..}, ... ]
        # ------------------------------------------------------
        data=load_data_json("data/EXIST2025_training_videos.json")

        # DAta is a pandas dataframe

        texts = data['text'].tolist()
        ids = data['id'].tolist()
        # ------------------------------------------------------
        # 2. Inferencia
        # ------------------------------------------------------
        preds = [self.predict(
            p,
            threshold=threshold,
            return_probabilities=return_probabilities
        ) for p in texts]

        # ------------------------------------------------------
        # 3. Empaquetar resultado por muestra
        # ------------------------------------------------------
        outputs = []
        for s, p in zip(ids, preds):
            # p puede ser:
            #   • dict  -> probabilidades por clase
            #   • list  -> lista de etiquetas duras
            #   • int/str -> etiqueta única (hard-single)
            #   • float -> prob. binaria (soft escalar)
            value: Any
            if isinstance(p, list):
                value = p
                if isinstance(p[0], dict):
                    # soft multi-label
                    # if k=='-' convertimos a 'NO'
                    value = {k if k != "-" else 'NO': v for k, v in p[0].items()}
                        
            elif isinstance(p,  str):
                value = p
            elif isinstance(p, np.ndarray):
                value=  p[0]
                value = 'NO' if value  == '-' else value     

            outputs.append(
                {
                    "test_case": test_case,
                    "id": s,
                    "value": value,
                }
            )

        output_file = f'{task_name}_{"soft" if self.soft else "hard"}_{team_name}_{run}.json'
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
        print(f"Submission saved to {output_file}")
        return outputs
