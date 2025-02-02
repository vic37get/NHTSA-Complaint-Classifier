---
library_name: transformers
tags:
  - text-classification
  - nlp
  - automotive
  - nhtsa
  - bert
datasets:
  - nhtsa-complaints
language: en
---

# 🚗 Modelo de Classificação de Reclamações de Veículos do NHTSA

Este modelo foi treinado para classificar reclamações de veículos registradas no banco de dados da **NHTSA (National Highway Traffic Safety Administration)** entre **2014 e 2024**. Ele classifica textos em **cinco categorias** de componentes veiculares:

- **ELECTRICAL SYSTEM**  
- **AIR BAGS**  
- **STRUCTURE**  
- **SERVICE BRAKES**  
- **OTHER** (outras reclamações não categorizadas)  

## 📂 Dados e Pré-processamento

Os dados foram extraídos da API oficial da NHTSA e passaram por um pipeline de processamento de linguagem natural (NLP), incluindo:

- **Limpeza e normalização**: remoção de caracteres especiais, conversão para caixa baixa e remoção de duplicatas/nulos.  
- **Balanceamento das classes**: ajuste da distribuição de categorias para evitar viés no treinamento.  
- **Tokenização**: uso do tokenizer do `bert-base-uncased` para transformar o texto em tensores compatíveis com o modelo.  

📊 **Divisão dos Dados**:

| **Conjunto**   | **Amostras** |
|---------------|-------------|
| Treinamento   | 8.357       |
| Validação     | 2.090       |
| Teste         | 2.090       |

## ⚙️ Hiperparâmetros do Treinamento

| Parâmetro               | Valor                                |
|-------------------------|------------------------------------|
| **Modelo base**         | `bert-base-uncased`               |
| **Batch size**          | 4                                  |
| **Taxa de aprendizado** | 1e-5                               |
| **Épocas**              | 30 (com early stopping de 3 épocas sem melhora) |
| **Otimizador**          | AdamW                              |

## 📊 Desempenho do Modelo

### 🔍 Conjunto de Validação

| Métrica    | Valor  |
|-----------|--------|
| **Acurácia**  | 86.40% |
| **F1-Score**  | 85.78% |
| **Precisão**  | 85.96% |
| **Recall**    | 86.40% |

### 🔍 Conjunto de Teste

| Métrica    | Valor  |
|-----------|--------|
| **Acurácia**  | 69.94% |
| **F1-Score**  | 75.69% |
| **Precisão**  | 87.96% |
| **Recall**    | 69.94% |

A diferença de desempenho entre os conjuntos de validação e teste pode ser explicada pelo desbalanceamento e pela natureza ampla da classe **OTHER**, que agrupa diferentes tipos de reclamações.

## 🚀 Como Usar

Para carregar e utilizar o modelo:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar modelo e tokenizer
model_name = "vic35get/nhtsa_complaints_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Função de inferência
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

# Exemplo de uso
text = "The airbag did not deploy during the accident."
print(predict(text))
