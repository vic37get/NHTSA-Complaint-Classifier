import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from typing import Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


class DataProcessor:
    def process(self, dataset) -> Dict:
        raise NotImplementedError


class TextProcessor(DataProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, dataset) -> Dict:
        def preprocess_data(examples):
            encoding = self.tokenizer(
                examples["summary"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
            encoding["label"] = examples["label"]
            return encoding
        return dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)


class ModelTrainer:
    def train(self, model, train_dataset, eval_dataset, training_args, compute_metrics):
        raise NotImplementedError


class HuggingFaceTrainer(ModelTrainer):
    def train(self, model, train_dataset, eval_dataset, training_args, compute_metrics):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        return trainer


def compute_metrics(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "hamming_loss": hamming_loss(labels, preds),
    }

def save_json(data: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train_model(cfg: DictConfig):
    """
    Treina o modelo de classificação de reclamações usando transformers.
    """
    print("Carregando o modelo e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    print("Carregando datasets...")
    datasets = load_dataset("csv", data_files={
        "train": cfg.paths.train_dataset,
        "eval": cfg.paths.eval_dataset,
        "test": cfg.paths.test_dataset
    })

    labels = list(set(datasets["train"]["label"]))
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    datasets = datasets.map(lambda example: {"label": label2id[example["label"]]}, num_proc=4)

    processor = TextProcessor(tokenizer)
    encoded_datasets = processor.process(datasets)
    encoded_datasets.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=cfg.paths.dir_save_model,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.model.learning_rate,
        per_device_train_batch_size=cfg.model.batch_size,
        per_device_eval_batch_size=cfg.model.batch_size,
        num_train_epochs=cfg.model.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    trainer = HuggingFaceTrainer()
    trainer.train(model, encoded_datasets["train"], encoded_datasets["eval"], training_args, compute_metrics)

    print("Avaliando o modelo...")
    metrics_eval = trainer.evaluate(encoded_datasets["eval"])
    metrics_test = trainer.evaluate(encoded_datasets["test"])
    
    save_json(metrics_eval, os.path.join(cfg.paths.dir_save_metrics, "metrics_eval.json"))
    save_json(metrics_test, os.path.join(cfg.paths.dir_save_metrics, "metrics_test.json"))

    trainer.save_model(cfg.paths.dir_save_model)
    print("Modelo treinado e salvo com sucesso!")

if __name__ == "__main__":
    train_model()