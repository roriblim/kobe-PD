## PD - Projeto da Disciplina - Engenharia de Machine Learning

(...)


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
streamlit run main.py
```

### JupyterLab

O projeto inclui suporte para JupyterLab. Para usar o JupyterLab:

```bash
# dentro da env, no diretório kobe/
kedro jupyter lab --no-browser
```






