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

class BertTrainerWrapper:
    def __init__(self, df=None, label_name="label1", soft=False, model_name="bert-base-multilingual-cased", split = True):

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
        self.is_regression = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.dataset = None
        self.split = split
    
    def prepare_labels(self):
        if self.soft:
            # --- soft binario como diccionario {"YES":p, "NO":1-p}
            if isinstance(self.df[self.label_name].iloc[0], dict):
                self.keys = sorted(set().union(
                    *self.df[self.label_name].apply(lambda x: x.keys())))
                self.df["labels"] = self.df[self.label_name].apply(
                    lambda d: [d.get(k, 0.0) for k in self.keys])
                self.num_labels = len(self.keys)
                self.problem_type = "multi_label_classification"
            else:
                # soft escalar (ej. 0.75)  ➞ regresión
                self.df["labels"] = self.df[self.label_name].astype(float)
                self.num_labels = 1
                self.problem_type = "regression"
                self.is_regression = True
        else:
            # --- hard: una sola clase string ➞ entero
            self.label_encoder = LabelEncoder()
            self.df["labels"] = self.label_encoder.fit_transform(
                self.df[self.label_name])
            self.num_labels = len(self.label_encoder.classes_)
            self.problem_type = "single_label_classification"

    def prepare_labels(self):
        if self.soft:
            # Soft: dict → vector
            self.keys = sorted(set().union(*self.df[self.label_name].apply(lambda x: x.keys())))
            def label_to_vec(d):
                return [d.get(k, 0.0) for k in self.keys]
            self.df["labels"] = self.df[self.label_name].apply(label_to_vec)
            self.num_labels = len(self.keys)
            self.problem_type = "multi_label_classification"
        else:
            # Hard: class → encoded int
            self.label_encoder = LabelEncoder()
            self.df["labels"] = self.label_encoder.fit_transform(self.df[self.label_name])
            self.num_labels = len(self.label_encoder.classes_)
            self.problem_type = "single_label_classification"

    def tokenize_dataset(self):
        ds = Dataset.from_pandas(self.df[["text", "labels"]])
        if self.split:
            ds= ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
        ds = ds.map(lambda x: self.tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=128), batched=False)
        self.dataset = ds

    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type=self.problem_type
        )

    def train(self, output_dir="./results", epochs=1, batch_size=8, lr=2e-5):
        self.prepare_labels()
        self.tokenize_dataset()
        self.build_model()

        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"] if self.split else self.dataset,
            eval_dataset=self.dataset['test'] if self.split else self.dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
    
    def compute_metrics(self, eval_pred, th=0.5):
        """
        th: umbral usado tanto para binarizar y_true como y_pred
            cuando calculamos F1 en modo soft
        """
        logits, labels = eval_pred

        # ---------------- SOFT ------------------------------------------
        if self.soft:
            probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()

            # -------- BINARIO soft (num_labels = 1) ----------
            if self.num_labels == 1:
                y_true = labels.squeeze()          # continua [0..1]
                y_pred = probs.squeeze()           # continua [0..1]

                y_true_hard = (y_true >= th).astype(int)
                y_pred_hard = (y_pred >= th).astype(int)

                return {
                    "mae":       float(np.mean(np.abs(y_pred - y_true))),
                    "logloss":   float(log_loss(y_true_hard, y_pred)),
                    "f1_micro":  float(f1_score(y_true_hard, y_pred_hard,
                                                average="micro")),
                    "f1_macro":  float(f1_score(y_true_hard, y_pred_hard,
                                                average="macro")),
                }

            # -------- MULTICLASE / MULTILABEL soft ------------
            y_true = labels                       # continua
            y_pred = probs

            y_true_hard = (y_true >= th).astype(int)
            y_pred_hard = (y_pred >= th).astype(int)

            micro_f1 = f1_score(
                y_true_hard.reshape(-1),
                y_pred_hard.reshape(-1),
                average="micro",
                zero_division=0,
            )
            macro_f1 = f1_score(
                y_true_hard,
                y_pred_hard,
                average="macro",
                zero_division=0,
            )
            ce = np.mean([
                log_loss(y_true_hard[:, k], y_pred[:, k])
                for k in range(self.num_labels)
            ])

            return {
                "mae":       float(np.mean(np.abs(y_pred - y_true))),
                "logloss":   float(ce),
                "f1_micro":  float(micro_f1),
                "f1_macro":  float(macro_f1),
            }

        # ---------------- HARD ------------------------------------------
        else:
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            return {"accuracy": float(acc)}

    
    @torch.no_grad()
    def predict(self, texts, threshold=0.5, return_probabilities=True):
        """
        texts: str o lista[str]
        threshold: umbral para binarizar (solo soft)
        return_probabilities:
            • soft  -> True  ⇒ devuelve probabilidades
            • soft  -> False ⇒ etiquetas (≥ threshold) / argmax
            • hard  -> ignora, siempre devuelve etiquetas de clase
        """
        if isinstance(texts, str):
            texts = [texts]

        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        device = next(self.model.parameters()).device       # NEW: detecta dónde está el modelo
        toks = {k: v.to(device) for k, v in toks.items()}   # NEW: mueve inputs a ese device
        self.model.eval()                                   # NEW: modo evaluación

        logits = self.model(**toks).logits

        # ---------- casos ----------
        if self.soft:
            if self.num_labels == 1:  # soft binario
                probs = torch.sigmoid(logits).squeeze().cpu().tolist()
                return probs if return_probabilities else [
                    int(p >= threshold) for p in probs
                ]

            probs = torch.sigmoid(logits).cpu().numpy()
            outputs = []
            for vec in probs:
                d = {k: float(v) for k, v in zip(self.keys, vec)}
                if return_probabilities:
                    outputs.append(d)
                else:
                    sel = [k for k, p in d.items() if p >= threshold] or [max(d, key=d.get)]
                    outputs.append(sel)
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
        task_name: str = "soft_3_1",
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

        # save to json
        if model_name is None:
            model_name = self.model_name.split("/")[-1]
        output_file = f"{model_name}_{task_name}.json"
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
        print(f"Submission saved to {output_file}")
        return outputs
