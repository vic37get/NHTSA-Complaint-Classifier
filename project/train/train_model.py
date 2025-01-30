from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer
from datasets import load_dataset
from datasets import Dataset
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score
import os
import json
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


class ClassifyComplaints:
    def __init__(self,
                 model_name: str,
                 path_train_dataset: str,
                 path_test_dataset: str,
                 path_eval_dataset: str,
                 batch_size: int,
                 dir_save_model: str,
                 dir_save_metrics: str,
                 learning_rate:  str,
                 epochs: str,
                 patience: int
                 ):
        
        self.model_name = model_name
        self.path_train_dataset = path_train_dataset
        self.path_test_dataset = path_test_dataset
        self.path_eval_dataset = path_eval_dataset
        self.batch_size = batch_size
        self.dir_save_model = dir_save_model
        self.dir_save_metrics = dir_save_metrics
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
    
    
    def train_model(self) -> None:
        """
            Realiza o treinamento e avaliação do modelo, além de salvar o modelo e suas métricas.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("Preparando os datasets..")
        train_dataset, eval_dataset, test_dataset = self.prepare_datasets()

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.id2label), 
            id2label=self.id2label, 
            label2id=self.label2id
        )
        model.resize_token_embeddings(len(self.tokenizer))
        
        args = TrainingArguments(
            output_dir=self.dir_save_model,
            logging_dir="../../models/logs",
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            greater_is_better=False,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        
        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
        )
        
        name_classifier = "complaints_classifier"
        trainer.train()
        
        print("Realizando o teste do modelo..")
        # Dataset de validação
        metrics_eval = trainer.evaluate(eval_dataset=eval_dataset)
        # Dataset de teste
        metrics_test = trainer.evaluate(eval_dataset=test_dataset)
        
        print("Salvando os arquivos...")
        dir_metrics_eval = os.path.join(self.dir_save_metrics, f"metrics_eval_{name_classifier}.json")
        dir_metrics_test = os.path.join(self.dir_save_metrics, f"metrics_test_{name_classifier}.json")
        dir_model = os.path.join(self.dir_save_model, name_classifier)
        
        json.dump(metrics_eval, open(dir_metrics_eval, "w"), indent=4, ensure_ascii=False)
        json.dump(metrics_test, open(dir_metrics_test, "w"), indent=4, ensure_ascii=False)
        print(f"As métricas foram salvas em: {dir_metrics_eval} e {dir_metrics_test}")
        
        trainer.save_model(dir_model)
        print(f"O modelo foi salvo em: {dir_model}")

        
    def prepare_datasets(self)-> tuple[Dataset]:
        """
            Realiza as transformações necessárias no dataset, deixando-o pronto para o treinamento e avaliação.
        """
        data = load_dataset("csv", data_files={"train": self.path_train_dataset, "eval": self.path_eval_dataset, "test": self.path_test_dataset})
        
        labels = list(set(data['train']['label']))
        self.id2label = {idx:label for idx, label in enumerate(labels)}
        self.label2id = {label:idx for idx, label in enumerate(labels)}
        
        data = data.map(lambda example: {'label': self.label2id[example['label']]})
        
        encoded_dataset = data.map(self.preprocess_data, batched=True, remove_columns=data['train'].column_names)
        encoded_dataset.set_format("torch")
        
        return encoded_dataset['train'], encoded_dataset['eval'], encoded_dataset['test']
        
        
    def preprocess_data(self, examples) -> dict:
        """
            Realiza a tokenização dos textos e retorna os encodings.
        """
        text = examples["summary"]
        encoding= self.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
        encoding['labels'] = examples['label']
        return encoding
    
    
    def compute_metrics(self, p) -> dict:
        """
            Computa as métricas para o treinamento e teste do modelo.
        """
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        recall = recall_score(labels, pred, average='weighted', zero_division=0)
        precision = precision_score(labels, pred, average='weighted', zero_division=0)
        f1 = f1_score(labels, pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, pred)
        hloss = hamming_loss(labels, pred)
        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "hamming_loss": hloss
        }
        return metrics


if __name__ == '__main__':
    
    params = {
        "model_name": "google-bert/bert-base-uncased", 
        "path_train_dataset": "../../data/csv/train.csv",
        "path_test_dataset": "../../data/csv/test.csv",
        "path_eval_dataset": "../../data/csv/eval.csv",
        "batch_size": 4,
        "dir_save_model": "../../models",
        "dir_save_metrics": "../../metrics",
        "learning_rate": 1e-5,
        "epochs": 30,
        "patience": 3
        }
    
    classifier = ClassifyComplaints(**params)
    classifier.train_model()
    print("Treinamento Finalizado")