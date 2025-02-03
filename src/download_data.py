import json
import requests
import logging
from tqdm.auto import tqdm
import os
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path("../data/json")
BASE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 10
HEADERS = {"User-Agent": "NHTSA-Data-Fetcher"}

def fetch_data(url: str) -> List[Dict]:
    """Faz uma requisição HTTP GET e retorna os dados em formato JSON."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao acessar {url}: {e}")
        return []

def save_json(data: List[Dict], file_path: Path) -> None:
    """Salva dados no formato JSON."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    logging.info(f"Dados salvos em {file_path}")

def get_all_model_years(output_file: Path) -> None:
    """Obtém todos os anos de modelos disponíveis."""
    url = "https://api.nhtsa.gov/products/vehicle/modelYears?issueType=c"
    model_years = fetch_data(url)
    years = [{"modelYear": year["modelYear"]} for year in model_years] if model_years else []
    save_json(years, output_file)

def get_all_makes_model_year(model_years: List[Dict], output_file: Path) -> None:
    """Obtém marcas associadas a cada ano de modelo."""
    url_template = "https://api.nhtsa.gov/products/vehicle/makes?modelYear={modelYear}&issueType=c"
    make_model_year = []
    
    for year in tqdm(model_years, desc="Obtendo marcas", colour="green"):
        url = url_template.format(modelYear=year["modelYear"])
        make_model_year.extend(fetch_data(url))
    
    save_json(make_model_year, output_file)

def get_all_models_make_year(make_model_years: List[Dict], output_file: Path) -> None:
    """Obtém modelos associados a cada combinação de marca e ano."""
    url_template = "https://api.nhtsa.gov/products/vehicle/models?modelYear={modelYear}&make={make}&issueType=c"
    model_make_year = []
    
    for item in tqdm(make_model_years, desc="Obtendo modelos", colour="yellow"):
        url = url_template.format(modelYear=item["modelYear"], make=item["make"])
        model_make_year.extend(fetch_data(url))
    
    save_json(model_make_year, output_file)

def get_all_complaints(make_model_year: List[Dict], output_file: Path) -> None:
    """Obtém reclamações por marca, modelo e ano."""
    url_template = "https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={modelYear}"
    complaints = []
    
    for item in tqdm(make_model_year, desc="Obtendo reclamações", colour="blue"):
        url = url_template.format(make=item["make"], model=item["model"], modelYear=item["modelYear"])
        complaints.extend(fetch_data(url))
    
    save_json(complaints, output_file)

if __name__ == "__main__":
    model_years_file = os.path.join(BASE_DIR, "model_years.json")
    make_model_year_file = os.path.join(BASE_DIR, "make_model_year.json")
    model_make_year_file = os.path.join(BASE_DIR, "model_make_year.json")
    complaints_file = os.path.join(BASE_DIR, "complaints.json")
    
    get_all_model_years(model_years_file)
    
    model_years = json.load(open(model_years_file))
    get_all_makes_model_year(model_years, make_model_year_file)
    
    make_model_years = json.load(open(make_model_year_file))
    get_all_models_make_year(make_model_years, model_make_year_file)
    
    model_make_years = json.load(open(model_make_year_file))
    get_all_complaints(model_make_years, complaints_file)