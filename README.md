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

1. Crie um ambiente virtual (recomendado):
Exemplo:
```bash
python3.11 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```
### Visualizando experimentos com MLflow

Para iniciar o servidor MLflow localmente:

```bash
mlflow server --host 0.0.0.0 --port 5001
```

### Executando o projeto

Para executar o pipeline Kedro:

```bash
kedro run
```
### Servindo o modelo gerado com MLflow

Após a criação do modelo, é possível subi-lo com o seguinte comando:

```
MLFLOW_TRACKING_URI=file://$PWD/mlruns mlflow models serve -m models:/model_DT/latest --env
-manager=local --port 5002
```

### JupyterLab

O projeto inclui suporte para JupyterLab. Para usar o JupyterLab:

```bash
kedro jupyter lab --no-browser
```

> **Nota**: Ao usar `kedro jupyter`, as seguintes variáveis estarão disponíveis no escopo: `context`, `session`, `catalog`, e `pipelines`.





