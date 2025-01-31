# Classificação de Reclamações de veículos do NHTSA

 Esse projeto consiste em uma pipeline completo de processamento e modelagem de dados usando reclamações de veículos do NHTSA do período entre 2014 e 2024. O foco principal do projeto é a predição do tipo de componente do veículo relatado nas reclamações, utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning.

### Metodologia

A abordagem adotada está representada no fluxograma da imagem abaixo e segue as seguintes etapas:

![alt text](/data/img/fluxogram.png)

**Aquisição e pré-processamento dos dados:** Foram coletados os dados de reclamações de veículos do NHTSA, realizou-se limpeza textual, correções textuais e tokenização.

**Engenharia de Features:** Os textos foram transformados em representações numéricas utilizando técnicas de embeddings.

**Treinamento do Modelo:** Implementamos e avaliamos modelos de classificação usando o BERT.

**Avaliação:** Testamos diferentes hiperparâmetros para otimizar o desempenho.

**Deploy da API**: Criamos uma Api para tornar possível consumir o modelo e criamos um container Docker para garantir a portabilidade do modelo, permitindo integração com aplicações externas.

## 1. Aquisição dos Dados

### 1.1 Objetivo

O objetivo desta etapa foi coletar os dados de reclamações de veículos da [NHTSA](https://www.nhtsa.gov/nhtsa-datasets-and-apis#complaints) (National Highway Traffic Safety Administration) entre os anos de 2014 e 2024. Esses dados form essenciais para treinar um modelo de classificação que prediz o tipo de componente veicular relatado nas reclamações. A escolha desse problema como tarefa de classificação se deve ao fato de os dados já estarem previamente rotulados, além da ampla variedade de categorias de componentes disponíveis para análise.

### 1.2 Fontes de Dados

Os dados foram adquiridos diretamente da API oficial da **NHTSA**. Foram utilizados os seguintes endpoints:

- **Model Years**: Obtém os anos de modelo disponíveis.
- **Makes by Model Year**: Obtém as marcas de veículo associadas a cada ano de modelo.
- **Models by Make and Year**: Obtém os modelos de veículos vinculados a cada combinação de marca e ano.
- **Complaints by Vehicle**: Obtém as reclamações registradas para cada veículo, considerando marca, modelo e ano.

### 1.3 Processo de Aquisição

A extração foi realizada em etapas sequenciais:

1. **Obtenção dos Anos de Modelo**: Chamada ao endpoint para listar os anos de modelo disponíveis.
2. **Obtenção das Marcas por Ano**: Para cada ano coletado, buscamos as marcas correspondentes.
3. **Obtenção dos Modelos por Marca e Ano**: Para cada combinação de marca e ano, buscamos os modelos.
4. **Obtenção das Reclamações**: Para cada combinação de marca, modelo e ano, buscamos as reclamações na API.

### 1.4 Estrutura dos Dados

Os dados extraídos foram armazenados em arquivos JSON organizados dentro do diretório `../../data/json`. Os principais arquivos gerados foram:

- `model_years.json`: Lista de anos de modelo.
- `make_model_year.json`: Lista de marcas vinculadas a cada ano de modelo.
- `model_make_year.json`: Lista de modelos vinculados a cada marca e ano.
- `complaints.json`: Lista de reclamações contendo detalhes como o tipo de problema relatado, o veículo associado e outras informações relevantes.

### 1.5 Tratamento de Erros e Logging

Para garantir a robustez da aquisição de dados, foram adotadas as seguintes medidas:

- Uso de **timeout** nas requisições para evitar bloqueios indesejados.
- Implementação de **tratamento de exceções** (`try/except`) para lidar com falhas de conexão ou respostas inválidas.
- Utilização de **logging estruturado** para registrar informações sobre o progresso e possíveis erros durante a extração.
- Exibição de **barras de progresso** (`tqdm`) para facilitar o monitoramento da execução do script.

### 1.6 Conclusão

Na etapa de aquisição de dados foi garantida a obtenção de informações estruturadas sobre os veículos e suas respectivas reclamações. Os dados coletados foram utilizados na próxima etapa para pré-processamento e extração de features importantes.

---

## 2. Processamento e Limpeza de Dados

### 2.1 Objetivo

O objetivo desta etapa foi preparar os dados brutos adquiridos para que possam ser utilizados no treinamento do modelo de classificação. As principais ações incluíram a limpeza dos textos das reclamações, a remoção de dados irrelevantes e a criação de um conjunto de dados balanceado.

### 2.2 Etapas do Pré-processamento

1. **Filtragem de Dados**: Foram selecionadas apenas as colunas relevantes (`odiNumber`, `dateComplaintFiled`, `components`, `summary`).
2. **Conversão de Datas**: A coluna `dateComplaintFiled` foi convertida para o formato datetime para permitir análises temporais.
3. **Remoção de Dados Duplicados e Nulos**: Reclamações sem descrição (`summary`) ou sem categoria (`components`) foram descartadas.
4. **Classificação das Reclamações**: As reclamações foram agrupadas em cinco categorias principais de componentes (`ELECTRICAL SYSTEM`, `AIR BAGS`, `STRUCTURE`, `SERVICE BRAKES` e `OTHER`). Foram escolhidas as primeiras quatro categorias por serem as mais abrangentes no conjunto de dados. Todas as outras categorias diferentes das mesmas foram colocadas na categoria *OTHER*. 
5. **Balanceamento de Classes**: Como algumas categorias tinham muito mais exemplos do que outras, mais especificamente a categorias *OTHERS*, foi aplicada uma técnica de balanceamento para garantir uma distribuição mais uniforme das classes no conjunto de treino. Essa técnica consiste em reduzir a quantidade de amostras das classes majoritarias para o mesmo número de amostras da classe minoritária do conjunto de dados.
6. **Limpeza de Texto**: Foram removidos caracteres especiais, múltiplos espaços, e o texto foi normalizado para caixa baixa, já que muitas reclamações não seguiam o padrão, muitas estavam em caixa alta e outras em caixa baixa. Tendo em vista o melhor aprendizado e generalização do modelo, foi optado por converter todas as reclamações para caixa baixa.

### 2.3 Separação dos Conjuntos de Dados

Os dados foram divididos da seguinte forma:

- **Treinamento e Validação**: Reclamações entre 2014 e 2024 foram usadas para treino e validação do modelo.
- **Teste**: Um conjunto de dados adicional, contendo reclamações anteriores a 2014, foi separado para avaliar o desempenho do modelo em um período diferente, atendendo ao requisito opcional de detecção de data drift. Não foi feita nenhuma manipulação nesses dados, do tipo balanceamento, como foi feito na conjunto de dados de treinamento.

Os dados foram salvos em arquivos *CSV* organizados no diretório `../../data/csv`:

- `train.csv`: Conjunto de treinamento, composto por **8.357** amostras.
- `eval.csv`: Conjunto de validação, composto por **2.090** amostras.
- `test.csv`: Conjunto de teste, composto por **2.090** amostras.

A distribuição das classes de cada conjunto de dados pode ser vista na tabela a seguir:

| Conjunto de Dados  | ELETRICAL SYSTEM | AIR BAGS | STRUCTURE | SERVICE BRAKES | OTHER |
|:------------------:|:---------------:|:--------:|:---------:|:--------------:|:-----:|
| **train.csv**      |      1.672      |   1.671  |   1.671   |      1.671     | 1.671 |
| **eval.csv**       |       417       |    418   |    418    |       418      |  418  |
| **test.csv**       |       159       |    99    |    113    |       17       | 1.701 |




### 2.4 Conclusão

A etapa de processamento garantiu que os dados estivessem limpos, organizados e balanceados, prontos para serem utilizados no treinamento do modelo de machine learning.

---

## 3. Análise Descritiva do Conjunto de Dados

Realizamos uma análise descritiva dos conjunto de dados de reclamações selecionado anteriormente. Essa etapa é muito importante para verificar a distribuição das variáveis após o pré-processamento, avaliar o balanceamento das classes e identificar possíveis padrões que possam influenciar o desempenho do modelo, além de apoiar a escolha do modelo que será utilizado. 

A imagem a seguir, trata de uma análise descritiva dos textos das reclamações.

![alt text](/data/img/eda_complaints.png)

### 3.1. Distribuição do Número de Palavras Únicas
A maioria dos textos de reclamação contém um número relativamente pequeno de palavras únicas, com poucos textos apresentando uma grande diversidade de vocabulário. Esse padrão sugere que o conjunto de dados é composta por textos com um vocabulário mais simples, o que pode ser vantajoso para modelos como o BERT, que se beneficiam de contextos linguísticos mais diretos e frequentes. Contudo, a falta de diversidade pode limitar a capacidade de generalização do modelo para textos com vocabulários mais complexos ou técnicos, o que pode ser um desafio em domínios mais especializados.

### 3.2. Distribuição do Comprimento Médio das Palavras
A análise do comprimento médio das palavras indica que a maioria dos textos apresenta palavras com 4 a 5 caracteres. Esse padrão é adequado para o BERT, pois o modelo lida bem com palavras de tamanho médio, conseguindo capturar seus significados contextuais de forma eficaz. Entretanto, a presença de termos técnicos ou siglas, que costumam ser mais curtos ou mais longos, pode prejudicar a capacidade do BERT de interpretar esses elementos com precisão, especialmente quando não são representados corretamente durante a tokenização.

### 3.3. Distribuição da Proporção de Stopwords
Observa-se que a maioria dos textos possui entre 30% e 40% de palavras consideradas stopwords, ou seja, palavras comuns que não carregam significado semântico relevante, como "o", "a", "de". Embora o BERT seja projetado para lidar com stopwords sem perda de desempenho, essa alta proporção pode gerar ruído durante o treinamento e aumentar o custo computacional, pois essas palavras ocupam parte do modelo sem agregar informações úteis para a tarefa de classificação.

### 3.4. Distribuição do Número de Sentenças
A maior parte dos textos contém poucas sentenças, mas há uma quantidade considerável de textos com um número maior de sentenças. Isso indica que, em sua maioria, os textos são relativamente curtos, o que é vantajoso para o BERT, que é otimizado para lidar com textos curtos e moderados. No entanto, quando os textos se tornam mais longos, o modelo pode enfrentar dificuldades, pois há limitações no número de tokens que podem ser processados de uma vez. Nesse caso, o BERT pode precisar realizar truncamento ou segmentação, o que pode resultar na perda de contexto importante para a classificação.

### 3.5. Distribuição das classes de problema
A figura a seguir mostra a distribuição das classes de problema no conjunto de dados

![alt text](/data/img/eda_component.png)

Como ilustrado no gráfico, as classes mais representativas no dataset são: **ELECTRICAL SYSTEM, STRUCTURE, AIR BAGS e SERVICE BRAKES**. Essas classes, por serem as mais frequentes, foram selecionadas para serem tratadas diretamente pelo modelo. As demais classes, que possuem uma frequência significativamente menor, serão agrupadas sob a categoria OTHER. Essa abordagem permite que o modelo foque nas classes majoritárias, aproveitando a grande quantidade de dados disponíveis, o que pode resultar em uma aprendizagem mais robusta e precisa.

## 4. Engenharia de Features e Treinamento do Modelo  

### 4.1 Escolha do Modelo  

Para este projeto, optamos por utilizar o **BERT (Bidirectional Encoder Representations from Transformers)**, especificamente a versão [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) desenvolvida pela Google, por ser uma versão pré-treinada com volumosos conjuntos de dados na língua inglesa.

O **BERT** foi escolhido por ser um dos modelos mais avançados e eficazes para tarefas de Processamento de Linguagem Natural (NLP), especialmente para a **classificação de texto**. Sua arquitetura baseada em **transformers** permite capturar relações contextuais entre palavras de maneira bidirecional, o que é fundamental para a compreensão de textos de reclamações veiculares, que muitas vezes contêm ambiguidades, expressões informais e um vocabulário específico do domínio.

A arquitetura do BERT é composta por várias camadas de transformadores, que são redes neurais especializadas no processamento de sequências. O modelo é pré-treinado em grandes volumes de texto e, em seguida, ajustado para tarefas específicas, como a classificação de texto. 

O treinamento do BERT envolve duas tarefas principais: **Masked Language Modeling (MLM)** e **Next Sentence Prediction (NSP)**, que permitem ao modelo aprender representações profundas e contextualizadas das palavras, resultando em um entendimento mais preciso do significado.

A tarefa de modelagem escolhida foi a **classificação multiclasse**, em que cada reclamação é categorizada em um dos tipos de componentes previamente rotulados no dataset da **NHTSA**. A escolha do BERT visa não só melhorar a precisão na identificação do tipo de problema, mas também garantir que o modelo seja capaz de lidar com as nuances linguísticas presentes nas reclamações.

---  

### 4.2 Engenharia de Features  

A engenharia de features foi realizada com foco na tokenização eficiente dos textos das reclamações. Utilizamos o **tokenizador do BERT**, que converte os textos em embeddings compatíveis com a arquitetura do modelo. As etapas aplicadas foram:  

1. **Tokenização**: Cada reclamação foi tokenizada.
2. **Truncamento e Padding**: Para manter um tamanho fixo de entrada, os textos foram truncados ou preenchidos para terem até **512 tokens**.  
3. **Conversão para Tensores**: As sequências tokenizadas foram convertidas para tensores PyTorch para treinamento eficiente.  

---  

### 4.3 Hiperparâmetros e Configuração do Treinamento  

O modelo foi treinado com os seguintes hiperparâmetros:  

| Parâmetro                | Valor                 |
|--------------------------|----------------------|
| **Modelo Base**          | `bert-base-uncased`  |
| **Tamanho do Batch**     | 4                    |
| **Taxa de Aprendizado**  | 1e-5                 |
| **Épocas**               | 30                   |
| **Early Stopping**       | 3 épocas sem melhora |
| **Estratégia de Avaliação** | `epoch`           |
| **Métrica para Melhor Modelo** | `eval_loss`  |
| **Otimizador**           | AdamW                |

**Critério de parada antecipada (early stopping)**: Para evitar overfitting, foi utilizado um critério de parada antecipada, interrompendo o treinamento após **3 épocas consecutivas sem melhoria na métrica de validação** (`eval_loss`).  

**Versionamento e Salvamento**:  
- Durante o treinamento, os modelos foram salvos na pasta `../../models`, e as métricas registradas na pasta `../../metrics`.  

---  

### 4.4 Resultados do Treinamento  

#### 4.4.1 Resultados no Conjunto de Dados de Validação  

| Métrica    | Valor  |
|------------|--------|
| **Acurácia**  | 86.40% |
| **F1-Score**  | 85.78% |
| **Precisão**  | 85.96% |
| **Recall**    | 86.40% |

#### 4.4.2 Resultados no Conjunto de Dados de Teste  

| Métrica    | Valor  |
|------------|--------|
| **Acurácia**  | 69.94% |
| **F1-Score**  | 75.69% |
| **Precisão**  | 87.96% |
| **Recall**    | 69.94% |

O modelo apresentou **alto desempenho na validação**, mas uma queda significativa nos resultados para o conjuunto de dados de teste.
Isso já era esperado, visto que a classe OTHER abrange tudo que não envolve as 4 classes principais, que foram definidas previamente. Foi atestado que no conjunto de dados original, existem classes simultâneas, como: **TIRES/STRUCTURE**, isso dificulta a classificação do modelo, visto que o mesmo foi treinado para classificar separadamente **TIRES** e **STRUCTURE**. Além disso, o conjunto de dados de teste está muito desbalanceado, concentrando mais de **80%** das amostras na classe *OTHER*, que é a classe que o modelo mais erra. Acredita-se que o baixo desempenho no conjunto de dados de teste em relação ao conjunto de dados de validação se deve a essas razões.

---  

### 4.5 Evolução das Métricas por Época  

O modelo foi treinado por **4 épocas**, com o seguinte comportamento:  

| Época | Acurácia | F1-Score |
|-------|----------|----------|
| 1     | 86.40%  | 85.78%  |
| 2     | 86.88%  | 86.47%  |
| 3     | 86.74%  | 86.07%  |
| 4     | 86.07%  | 85.78%  |

**Análise**:  
- Observamos um aumento na **acurácia** e no **F1-score** até a segunda época.  
- A partir da terceira época, houve uma **leve degradação** no desempenho.  
- O critério de **early stopping** interrompeu o treinamento na **quarta época**, prevenindo um overfitting.  

---  

### 4.6 Conclusões

O modelo demonstrou **bom desempenho no conjunto de dados de validação**, mas uma **redução significativa no conjunto de dados de teste**, o que sugere que melhorias podem ser aplicadas. Algumas estratégias para mitigar esse efeito incluem:  

✅ **Aumento do tamanho do batch** para melhorar a estabilidade do treinamento.  
✅ **Data Augmentation**: Uso de técnicas como sinônimos e reformulação para aumentar a diversidade dos dados de treinamento.    
✅ **Fine-tuning com menos camadas congeladas** para permitir melhor adaptação ao domínio das reclamações veiculares.
✅ **Reformular a divisão de classes de reclamações** para abranger também aqueles registros que contém mais de uma classe. Poderia ser usada uma abordagem de classificação **multilabel**, em que o modelo pode classificar uma reclamação com mais de uma classe simultâneamente.

Com essas melhorias, é esperado alcançar **um modelo mais robusto e generalizável**, capaz de lidar melhor com dados fora da distribuição original.

## 5. Deploy do Modelo no Hugging Face

### 5.1 O que é o Hugging Face?

O [Hugging Face](https://huggingface.co) é uma plataforma líder no campo de processamento de linguagem natural (NLP), oferecendo uma vasta coleção de modelos pré-treinados e ferramentas para diversas tarefas, como análise de sentimentos, tradução, e classificação de textos. A plataforma permite que pesquisadores e desenvolvedores compartilhem seus modelos, facilitando o acesso e a colaboração na comunidade de machine learning.

### 5.2 Deploy do Modelo

O modelo de classificação de reclamações ([nhtsa_complaints_classifier](https://huggingface.co/vic35get/nhtsa_complaints_classifier)) foi disponibilizado na plataforma Hugging Face, permitindo seu acesso e utilização pela comunidade.

Após o treinamento, o modelo foi carregado e enviado para o repositório público do Hugging Face, onde qualquer usuário pode acessá-lo e utilizá-lo em seus próprios projetos. Isso garante que o modelo esteja disponível para uso em diversas aplicações e que a comunidade possa colaborar na melhoria e adaptação do modelo.

Com o deploy, o modelo agora pode ser facilmente integrado em sistemas de produção, onde é possível realizar classificações de novas reclamações de veículos em tempo real, aproveitando a infraestrutura escalável e robusta oferecida pela plataforma.

## 6. API para Consumo do Modelo

### 6.1 Objetivo

O objetivo desta etapa foi criar uma API simples que permita consumir o modelo treinado e fazer previsões com novas reclamações. A API foi implementada utilizando o framework **Flask**, sendo uma solução prática e eficiente para disponibilizar o modelo para uso em produção.

### 6.2 Detalhes da Implementação

A API foi projetada para expor quatro endpoints principais:

1. **GET /**: Um endpoint de verificação, retornando uma mensagem simples de boas-vindas.
2. **GET /status**: Endpoint para verificar o status da API, útil para monitoramento.
3. **POST /load**: Endpoint responsável pelo carregamento do modelo treinado e do tokenizador a partir de um modelo pré-existente (como o BERT). Ele permite que o modelo seja carregado na memória e esteja pronto para realizar previsões.
4. **POST /classify_complaints**: Endpoint principal da API, responsável por receber reclamações no formato JSON, limpar o texto e classificar o tipo de componente utilizando o modelo treinado.

### 6.3 Funcionamento

A estrutura da API é baseada no Flask, com a utilização da biblioteca `flask_classful` para facilitar a organização e modularização dos endpoints. A seguir, estão os detalhes de cada uma das rotas implementadas:

- **/ (GET)**: Endpoint básico para verificar se a API está funcionando corretamente, retornando uma mensagem de sucesso.
  
- **/status (GET)**: Endpoint de status que retorna uma mensagem de "sucesso", indicando que a API está ativa.

- **/load (POST)**: Este endpoint é utilizado para carregar o modelo e o tokenizador. Ele recebe um JSON contendo o nome do modelo, faz o carregamento do modelo pré-treinado a partir da Hugging Face, e prepara a pipeline de classificação. Isso permite que o modelo seja recarregado sem a necessidade de reiniciar a aplicação.
  - Exemplo de payload:
    ```json
    {
      "model": "vic35get/nhtsa_complaints_classifier"
    }
    ```

- **/classify_complaints (POST)**: Endpoint responsável por receber uma reclamação no formato JSON, limpar o texto utilizando uma função de limpeza de texto personalizada e passar o texto para o modelo de classificação. O resultado da classificação é então retornado no formato JSON. A resposta inclui o rótulo da classificação e a pontuação associada, que indica a confiança do modelo na classificação.
  - Exemplo de payload:
    ```json
    {
      "complaint": "I am facing issues with the airbag system."
    }
    ```
  - Exemplo de retorno:
    ```json
    {
        "message": "success",
        "output": [
            {
                "label": "AIR BAGS",
                "score": 0.9949865341186523
            }
        ],
        "status": 0
    }
    ```

### 6.4 Função de Limpeza de Texto

A função de limpeza de texto é uma etapa crucial para garantir que os dados estejam no formato adequado para a classificação. A função `clean_text` realiza as seguintes ações:
- **Remoção de caracteres especiais**: Qualquer caractere indesejado, como pontuações, símbolos e caracteres não alfanuméricos, é substituído por espaços.
- **Normalização de texto**: O texto é convertido para minúsculas e os múltiplos espaços são reduzidos a um único espaço.
  
Esse processo garante que o texto esteja limpo e sem ruídos, proporcionando melhores resultados de classificação.

### 6.5 Configurações e Execução

A API foi configurada para rodar na porta 5009, mas permite a personalização de parâmetros como a porta e o host via argumentos de linha de comando. A execução da aplicação pode ser feita diretamente com o comando:
```bash
python3 service.py
```
Para o uso adequado da API, deve-se executar primeiramente o endpoint load, que carrega o modelo em memória, passando o nome do modelo de classificação `vic35get/nhtsa_complaints_classifier`, assim como visto na seção 5.3.

## 7. Deploy da API com Docker

### 7.1 Objetivo

Para garantir a portabilidade, escalabilidade e facilidade de deploy, a API foi containerizada utilizando **Docker**. A criação do container permite que a API seja executada de forma consistente em qualquer ambiente, seja em desenvolvimento ou produção, sem dependências externas ou configurações específicas de sistema operacional. A utilização do **Docker Compose** facilita a orquestração e o gerenciamento do container, garantindo que todos os serviços necessários (como a execução da API) estejam devidamente configurados e isolados.

### 7.2 Arquivos de Configuração

Foram criados dois arquivos principais para a configuração do Docker:

1. **docker-compose.yml**: Este arquivo define o serviço da API, especificando como o container será construído, quais portas serão expostas e as variáveis de ambiente necessárias para rodar a aplicação em produção. Além disso, ele garante que o container seja reiniciado automaticamente em caso de falha.

2. **Dockerfile**: O Dockerfile descreve o processo de construção da imagem Docker, incluindo a instalação das dependências necessárias para rodar a aplicação e a configuração da execução do servidor Flask.

### 7.3 Vantagens do Deploy com Docker

- **Portabilidade**: A aplicação containerizada pode ser executada em qualquer máquina que tenha o Docker instalado, sem a necessidade de configurações adicionais. Isso garante que a API funcionará de maneira idêntica em diferentes ambientes.
- **Facilidade de Deploy**: Com o Docker Compose, o processo de deploy da API se torna simples e direto. Um único comando é suficiente para construir e rodar o container, eliminando a complexidade de configuração manual de ambientes.
- **Escalabilidade**: Docker permite que a aplicação seja facilmente escalada, criando múltiplos containers em diferentes servidores ou nuvens. Isso facilita o crescimento da infraestrutura à medida que a demanda aumenta.

### 7.4 Como Executar a API

Para executar a API containerizada, siga os seguintes passos:

1. Certifique-se de que o **Docker** e o **Docker Compose** estão instalados em sua máquina.
2. Va até o diretório `/project/api`, que é onde os arquivos **docker-compose.yml** e **Dockerfile** estão localizados.
3. Execute o comando abaixo para construir e iniciar o container:

   ```bash
   docker-compose up -d --build

Para o uso adequado da API, deve-se executar primeiramente o endpoint load, que carrega o modelo em memória, passando o nome do modelo de classificação `vic35get/nhtsa_complaints_classifier`, assim como visto na seção 5.3.