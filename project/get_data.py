import requests
from tqdm.auto import tqdm
import json
import pandas as pd
from datetime import datetime

def get_all_model_years() -> dict:
    """
    Obtem todos o ano de todos os modelos dispoíveis.
    """
    url = "https://api.nhtsa.gov/products/vehicle/modelYears?issueType=c"
    try:
        response = requests.get(url)
        results = response.json().get('results')
        if results:
            return [year['modelYear'] for year in results]
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    return None

def get_all_makes_model_year(list_model_year: list) -> dict:
    """
    A partir da lista de anos dos modelos disponíveis, obtem a marca daquele modelo.
    """
    url = "https://api.nhtsa.gov/products/vehicle/makes?modelYear={model_year}&issueType=c"
    # Lista com os modelos e anos.
    make_modelyear = []
    for model_year in tqdm(list_model_year, desc="Capturando modelo e ano", colour="green"):
        response = requests.get(url.format(model_year=model_year))
        results = response.json().get('results')
        if results:
            make_modelyear.extend(results)
    return make_modelyear

def get_all_models_make_year(list_make_year: list[dict]) -> list[dict]:
    """
    Obtem uma lista de dicionários que contém a marca, o modelo e o ano para cada carro.
    """
    url = "https://api.nhtsa.gov/products/vehicle/models?modelYear={modelYear}&make={make}&issueType=c"
    # Lista com modelos, marcas e anos.
    model_make_year = []
    for item in tqdm(list_make_year, desc="Capturando marca, modelo e ano", colour="yellow"):
        response = requests.get(url.format(modelYear=item.get('modelYear'), make=item.get("make")))
        results = response.json().get('results')
        if results:
            model_make_year.extend(results)
    return model_make_year

def get_all_complaints(make_model_year: list[dict]) -> list[dict]:
    """
    Obtem uma lista de dicionários que contém as reclamações por modelo, marca e ano.
    """
    url = "https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={modelYear}"
    # Lista das reclamações por modelo, marca e ano.
    complaints = []
    for item in tqdm(make_model_year, desc="Capturando as reclamações", colour="blue"):
        response = requests.get(url.format(make=item.get('make'), model=item.get('model'), modelYear=item.get('modelYear')))
        results = response.json().get('results')
        if results:
            for result in results:
                try:
                    date = result.get('dateComplaintFiled')
                    if date:
                        data_formatada = datetime.strptime(date, '%m/%d/%Y')
                        year = data_formatada.year
                        if year >= 2014 and year <= 2024:
                            complaints.extend(result)
                except Exception as e:
                    print(f"Erro ao processar o registro: {e}")
                    pass
    return complaints


list_model_year = get_all_model_years()
list_make_year = get_all_makes_model_year(list_model_year)
list_make_model_year = get_all_models_make_year(list_make_year)

make_model_year = pd.DataFrame(list_make_model_year)
make_model_year.drop_duplicates(inplace=True)
make_model_year.dropna(inplace=True)
make_model_year = make_model_year.to_dict(orient='records')

complaints = get_all_complaints(make_model_year)

json.dump(complaints, open("complaints.json", "w"), indent=4, ensure_ascii=False)
df_complaints = pd.DataFrame(complaints)
df_complaints.to_csv("df_complaints.csv", index=False)
