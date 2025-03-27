# imagem base do Python 3.11
FROM python:3.11-slim

# variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# diretório de trabalho
WORKDIR /app



# instalação das dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# código fonte
COPY . .

# porta do serviço
EXPOSE 5000

# comando padrão para executar o projeto
CMD ["kedro", "run"] 