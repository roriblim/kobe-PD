
## PD - Projeto da Disciplina - Engenharia de Machine Learning

## Overview

Para rodar este projeto, foi criado um ambiente com Python na versão 3.11

## Executando com Docker

O projeto está configurado para ser executado usando Docker e Docker Compose. Esta é a maneira recomendada de executar o projeto, pois garante um ambiente consistente e isolado.

### Pré-requisitos

- Docker
- Docker Compose

### Executando o projeto

Para construir e iniciar os containers:

```bash
docker-compose up --build
```

Para executar em background:

```bash
docker-compose up -d --build
```

Para parar os containers:

```bash
docker-compose down
```

### Serviços disponíveis

O projeto inclui três serviços:

1. **kobe**: O serviço principal que mantém o ambiente rodando
   - Porta 5000: MLflow UI
   - Este serviço mantém o ambiente ativo para execução de comandos

2. **jupyter**: Serviço dedicado para notebooks
   - Porta 8888: Interface do Jupyter Notebook
   - Acessível em `http://localhost:8888`

3. **mlflow**: Interface do MLflow para visualização de experimentos
   - Porta 5001: Interface web do MLflow
   - Acessível em `http://localhost:5001`

### Executando comandos no container

Para executar comandos no container principal (como rodar o pipeline), você pode usar:

```bash
# Executar o pipeline
docker-compose exec kobe kedro run

# Executar outros comandos kedro
docker-compose exec kobe kedro [comando]
```

### Volumes e persistência

O projeto está configurado com os seguintes volumes:

1. **Código Fonte**: Todo o diretório do projeto está montado como volume (`.:/app`), permitindo alterações no código sem precisar reconstruir a imagem.

2. **Dados Persistentes**: Diretórios específicos são montados como volumes separados para garantir persistência e melhor performance:
   - `./data`: Armazena os dados do projeto
   - `./mlruns`: Armazena as métricas e experimentos do MLflow

Para aplicar alterações no código:
1. Faça as modificações nos arquivos do projeto e salve
2. As alterações serão refletidas automaticamente no container
3. Execute novamente o pipeline se necessário

> **Nota**: Embora os diretórios `data` e `mlruns` já estejam incluídos no volume principal do projeto, eles são montados separadamente para garantir persistência e otimização de performance.

## Executando de forma local sem Docker

### Pré-requisitos

- Python 3.11
- pip (gerenciador de pacotes Python)

### Configuração do ambiente

1. Crie um ambiente virtual (recomendado):
Exemplo:
```bash
# Criar ambiente virtual com Python 3.11
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

### Executando o projeto

Para executar o pipeline Kedro:

```bash
kedro run
```

### Trabalhando com notebooks

O projeto inclui suporte para JupyterLab. Para usar o JupyterLab:

#### JupyterLab
```bash
kedro jupyter lab --no-browser
```



> **Nota**: Ao usar `kedro jupyter`, as seguintes variáveis estarão disponíveis no escopo: `context`, `session`, `catalog`, e `pipelines`.

### Visualizando experimentos com MLflow

Para iniciar o servidor MLflow localmente:

```bash
mlflow server --host 0.0.0.0 --port 5001
```

A interface do MLflow estará disponível em `http://localhost:5001`







