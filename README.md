## PD - Projeto da Disciplina - Engenharia de Machine Learning [25E1_3]
#### Aluna: Rosana Ribeiro Lima


## Item 1.
Link para o projeto no Github: 
https://github.com/roriblim/kobe-PD

## Item 2.
#### Diagramas com as etapas do projeto:
- Pipeline de preparação dos dados:

![Pipeline de preparação dos dados](docs/pipeline-01-preparacao-dados.png)

- Pipeline de treinamento dos modelos

![Pipeline de treinamento dos modelos](docs/pipeline-02-treinamento.png)

- Pipeline de aplicação dos modelos aos dados de produção

![Pipeline de aplicação dos modelos aos dados de produção](docs/pipeline-03-aplicacao_prod.png)

- Servindo o melhor modelo via MLflow e monitorando via Streamlit

![Pipeline de preparação dos dados](docs/servindo-e-monitorando-modelo.png)

## Item 3.


## Overview

Para rodar este projeto, foi criado um ambiente com Python na versão 3.11


## Arquivos e Diretórios Importantes


- `mlruns/`: Diretório para armazenar artefatos do MLflow
- `conf/local/mlflow.yml`: Configuração do MLflow 

(...)


## Executando de forma local 

### Pré-requisitos

- Python 3.11
- pip (gerenciador de pacotes Python)

### Configuração do ambiente

1. Crie um ambiente virtual com Python 3.11 (recomendado):
Exemplo:
```bash
conda create --name PD_env_1 python=3.11 --no-default-packages -y
conda activate PD_env_1 # sempre que for necessário entrar na env para executar comandos no projeto!
```

2. Instale as dependências:

```bash
# dentro da env, no diretório kobe/
pip install -r requirements.txt
```

### Visualizando experimentos com MLflow

Para iniciar o servidor MLflow localmente:

```bash
# dentro da env, no diretório kobe/
mlflow server --host 0.0.0.0 --port 5000
```

### Executando o projeto

Para executar as pipelines Kedro (pré-processamentos, gerar as métricas, modelos e resultados, e inferências com os dados de produção):

```bash
# dentro da env, no diretório kobe/
kedro run
```

### Servindo o modelo gerado com MLflow

É possível subir o melhor modelo encontrado com o seguinte comando:

```bash
# dentro da env, no diretório kobe/
MLFLOW_TRACKING_URI=file://$PWD/mlruns mlflow models serve -m models:/best_model/latest --env-manager=local --port 5002
```

### Monitorando API com Streamlit (fazendo inferências)

É possível subir o melhor modelo encontrado com o seguinte comando:

```bash
# dentro da env, no diretório kobe/
cd streamlit
streamlit run main_API.py 
```

### JupyterLab

O projeto inclui suporte para JupyterLab. Para usar o JupyterLab:

```bash
# dentro da env, no diretório kobe/
kedro jupyter lab --no-browser
```






