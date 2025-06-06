"""
This is a boilerplate pipeline 'preparacao_dados'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(raw_dev):

    raw_data_unique = raw_dev.drop_duplicates()

    data = (
    raw_data_unique[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance', 'shot_made_flag']]
    .dropna()
    .assign(playoffs = lambda x: x['playoffs'].astype(bool))
    )

    mlflow.log_metric("num_linhas_raw_dev", len(raw_dev))
    mlflow.log_metric("num_linhas_primary_dev", len(data))

    data.to_csv("data/03_primary/primary_dev.csv", index=False)
    mlflow.log_artifact("data/03_primary/primary_dev.csv")

    return data

def feature_engineering(data):

    # a análise para chegar nesses parâmetros foi feita no notebook feature_engineering.ipynb
    data['lat_quadra'] = data['lat'] - 34.0443
    data['lon_quadra'] = data['lon'] + 118.2698
    data = data.drop(columns=["lat", "lon"])

    data.to_csv("data/04_feature/data_filtered_dev.csv", index=False)
    mlflow.log_artifact("data/04_feature/data_filtered_dev.csv")
   
    return data

def separacao_treino_teste(data, random_state_param, test_size):

    # parâmetro de proporção de teste
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state_param)

    x_data = data.drop('shot_made_flag', axis=1)
    y_data = data[['shot_made_flag']]

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=test_size,
    random_state=random_state_param, stratify=y_data)

    data_train = pd.concat([x_data_train, y_data_train], axis=1)
    data_test = pd.concat([x_data_test, y_data_test], axis=1)

    plota_distribuicao_features(x_data_train,"train")
    plota_distribuicao_features(x_data_test,"test")

    # métricas sobre o tamanho das bases
    mlflow.log_metric("tamanho_dados_pós_preparacao_linhas", data.shape[0])
    mlflow.log_metric("tamanho_dados_pós_preparacao_colunas", data.shape[1])
    mlflow.log_metric("tamanho_treino_linhas", data_train.shape[0])
    mlflow.log_metric("tamanho_treino_colunas", data_train.shape[1])
    mlflow.log_metric("tamanho_teste_linhas", data_test.shape[0])
    mlflow.log_metric("tamanho_teste_colunas", data_test.shape[1])
    mlflow.log_metric("proporcao_treino", len(data_train) / len(data))
    mlflow.log_metric("proporcao_teste", len(data_test) / len(data))

    # distribuição das classes
    mlflow.log_metric("proporcao_positivos_total", data['shot_made_flag'].mean())
    mlflow.log_metric("proporcao_positivos_treino", data_train['shot_made_flag'].mean())
    mlflow.log_metric("proporcao_positivos_teste", data_test['shot_made_flag'].mean())

    data_train.to_csv("data/05_model_input/base_train.csv", index=False)
    mlflow.log_artifact("data/05_model_input/base_train.csv")
    data_test.to_csv("data/05_model_input/base_test.csv", index=False)
    mlflow.log_artifact("data/05_model_input/base_test.csv")

    return data_train, data_test, data_train, data_test

def plota_distribuicao_features(x_data_train, name):
    num_features = x_data_train.shape[1]
    cols = 3
    rows = (num_features // cols) + int(num_features % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(x_data_train.columns):
        sns.violinplot(y=x_data_train[col], ax=axes[i])
        axes[i].set_title(f"Violinplot - {col}")
        axes[i].set_xlabel("")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()    
    plt.savefig(f'data/08_reporting/distribuicao_features_{name}.png')
    plt.close()




