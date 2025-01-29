import json
import requests
from tqdm.auto import tqdm
import os


def get_all_complaints(make_model_year: list[dict], output_file: str = "complaints.json") -> list[dict]:
    """
    Obtém uma lista de dicionários contendo as reclamações por modelo, marca e ano.
    Os dados são salvos em um arquivo JSON a cada 100 registros processados.
    """
    url_template = "https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={modelYear}"
    complaints = []
    record_count = 0

    for item in tqdm(make_model_year, desc="Capturando as reclamações", colour="blue"):
        try:
            url = url_template.format(make=item.get('make'), model=item.get('model'), modelYear=item.get('modelYear'))
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('results', [])
            for result in results:
                complaints.append(result)
                record_count += 1

                if record_count >= 100:
                    with open(output_file, "w") as file:
                        json.dump(complaints, file, indent=4)
                        record_count = 0
        except Exception as e:
            print(e)
            continue
            
    if complaints:
        with open(output_file, "w") as file:
            json.dump(complaints, file, indent=4)

    print(f"Dados salvos em {output_file}")
    return complaints

def main() -> None:
    BASE_DIR = "../../data/json"
    model_make_year = json.load(open(os.path.join(BASE_DIR, "model_make_year.json"), 'r'))
    get_all_complaints(model_make_year, output_file=os.path.join(BASE_DIR, "complaints.json"))
    


if __name__ == '__main__':
    main()