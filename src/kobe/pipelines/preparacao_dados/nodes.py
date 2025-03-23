"""
This is a boilerplate pipeline 'preparacao_dados'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd

def prepare_data(raw_dev):

    mlflow.log_param("num_linhas", len(raw_dev))

    # Processamento dos dados
    processed_data = raw_dev.dropna()
    
    # Registrar métricas
    mlflow.log_metric("linhas_antes", len(raw_dev))
    mlflow.log_metric("linhas_depois", len(processed_data))

    # Salvar um artefato
    processed_data.to_csv("processed_data.csv", index=False)
    mlflow.log_artifact("processed_data.csv")

    return processed_data