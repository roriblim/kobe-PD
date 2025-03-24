"""
This is a boilerplate pipeline 'preparacao_dados'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(raw_dev):

    raw_data_unique = raw_dev.drop_duplicates()

    data = (
    raw_data_unique[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance', 'shot_made_flag']]
    .dropna()
    .assign(playoffs = lambda x: x['playoffs'].astype(bool))
    .assign(shot_made_flag = lambda x: x['shot_made_flag'].astype(bool))
    )

    # Registrar métricas
    mlflow.log_metric("num_linhas_raw", len(raw_dev))
    mlflow.log_metric("num_linhas_primary", len(data))

    # Salvar um artefato
    data.to_csv("primary_dev.csv", index=False)
    mlflow.log_artifact("primary_dev.csv")

    return data

def feature_engineering(data):

   # a análise para chegar nesses parâmetros foi feita no notebook feature_engineering.ipynb
   data['lat_quadra'] = data['lat'] - 34.0443
   data['lon_quadra'] = data['lon'] + 118.2698
   data = data.drop(columns=["lat", "lon"])
   
   return data

def separacao_treino_teste(data, random_state_param):

    x_data = data.drop('shot_made_flag', axis=1)
    y_data = data[['shot_made_flag']]

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.2,
    random_state=random_state_param, stratify=y_data)

    data_train = pd.concat([x_data_train, y_data_train], axis=1)
    data_test = pd.concat([x_data_test, y_data_test], axis=1)

    return data_train, data_test