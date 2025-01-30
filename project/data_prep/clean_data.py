import logging
import json
import regex as re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path("../../data/json")
CSV_DIR = Path("../../data/csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "ELECTRICAL SYSTEM": 0,
    "AIR BAGS": 1,
    "STRUCTURE": 2,
    "SERVICE BRAKES": 3,
    "OTHER": 4
}

def load_data(file_path: Path) -> pd.DataFrame:
    """Carrega os dados de um arquivo JSON para um DataFrame."""
    logging.info(f"Carregando os dados a partir do caminho: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra, limpa e prepara os dados para o treinamento."""
    logging.info("Pre-processando os dados..")
    df = df.loc[:, ("odiNumber", "dateComplaintFiled", "components", "summary")]
    df["dateComplaintFiled"] = pd.to_datetime(df["dateComplaintFiled"], format="%m/%d/%Y")
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['summary'], inplace=True)
    df["label"] = df["components"].apply(lambda x: x.strip() if x.strip() in CLASSES.keys() else "OTHER")
    return df

def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Balanceia as classes do dataset."""
    logging.info("Balanceando os dados para treinamento...")
    threshold_balanced = round(np.min(df['label'].value_counts()) / 2)
    balanced_data = [df[df['label'] == label].sample(n=threshold_balanced, random_state=42) for label in CLASSES.keys()]
    return pd.concat(balanced_data)

def clean_text(text: str) -> str:
    """Remove caracteres indesejados do texto."""
    text = re.sub(r'([•●▪•_·□»«#£¢¿&^~´`¨\t])', ' ', text)
    text = re.sub(r'(-)+', '-', text)
    text = re.sub(r'(\.)+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s\.$', '.', text)
    return text.lower().strip()

def split_and_save_data(train_df: pd.DataFrame, eval_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Salva os datasets de treino, validação e teste em arquivos CSV."""
    logging.info("Salvando os datasets...")
    for dataset, name in zip([train_df, eval_df, test_df], ["train", "eval", "test"]):
        dataset.reset_index(drop=True, inplace=True)
        dataset.to_csv(CSV_DIR / f"{name}.csv", index=False)

def main():
    df = load_data(BASE_DIR / "complaints_full.json")
    df = preprocess_data(df)
    
    df_train_eval = df[(df['dateComplaintFiled'].dt.year >= 2014) & (df["dateComplaintFiled"].dt.year <= 2024)]
    df_test = df[df['dateComplaintFiled'].dt.year < 2014]
    
    df_train_eval = balance_data(df_train_eval)
    df_test = df_test.sample(n=round(len(df_train_eval) * 0.2), random_state=42)
    
    logging.info("Realizando a limpeza dos textos...")
    df_train_eval["summary"] = df_train_eval["summary"].apply(clean_text)
    df_test["summary"] = df_test["summary"].apply(clean_text)
    
    train_df, eval_df = train_test_split(df_train_eval, test_size=0.2, random_state=42, stratify=df_train_eval['label'])
    
    split_and_save_data(train_df, eval_df, df_test)


if __name__ == "__main__":
    main()