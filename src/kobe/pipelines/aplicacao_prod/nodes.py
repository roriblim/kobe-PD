"""
This is a boilerplate pipeline 'aplicacao_prod'
generated using Kedro 0.19.12
"""
from sklearn.metrics import log_loss, f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from kobe.pipelines.preparacao_dados.nodes import plota_distribuicao_features
import mlflow

def pre_process_predict(data):

    
    data_processed = data.dropna()
    data_processed_x = (data_processed[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance']]
        .assign(playoffs = lambda x: x['playoffs'].astype(bool))
       )   
    data_processed_x['lat_quadra'] = data_processed_x['lat'] - 34.0443
    data_processed_x['lon_quadra'] = data_processed_x['lon'] + 118.2698
    data_processed_x = data_processed_x.drop(columns=["lat", "lon"])

    plota_distribuicao_features(data_processed_x,"prod")

    data_processed_y = data_processed[['shot_made_flag']]

    return data_processed_x, data_processed_y


def predict_DT(data_x, data_y_actual, model_DT):

    model = pickle.load(open('data/06_models/DT_model.pkl', 'rb'))
    data_y_proba = model.predict_proba(data_x)
    data_y_pred = model.predict(data_x)


    metrics_DT = get_metrics_prod(data_y_proba, data_y_pred, data_y_actual, "DT")

    data_y_pred_df = pd.DataFrame(data_y_pred)
    data_y_proba_df = pd.DataFrame(data_y_proba)

    data_y_pred_df.to_parquet("data/07_model_output/DT_data_prod_y_pred.parquet", index=False)
    mlflow.log_artifact("data/07_model_output/DT_data_prod_y_pred.parquet")
    data_y_proba_df.to_parquet("data/07_model_output/DT_data_prod_y_proba.parquet", index=False)
    mlflow.log_artifact("data/07_model_output/DT_data_prod_y_proba.parquet")

    return data_y_proba_df, data_y_pred_df, metrics_DT

def predict_RL(data_x, data_y_actual, model_RL):

    model = pickle.load(open('data/06_models/RL_model.pkl', 'rb'))
    data_y_proba = model.predict_proba(data_x)
    data_y_pred = model.predict(data_x)

    metrics_RL = get_metrics_prod(data_y_proba, data_y_pred, data_y_actual, "RL")

    data_y_pred_df = pd.DataFrame(data_y_pred)
    data_y_proba_df = pd.DataFrame(data_y_proba)

    data_y_pred_df.to_parquet("data/07_model_output/RL_data_prod_y_pred.parquet", index=False)
    mlflow.log_artifact("data/07_model_output/RL_data_prod_y_pred.parquet")
    data_y_proba_df.to_parquet("data/07_model_output/RL_data_prod_y_proba.parquet", index=False)
    mlflow.log_artifact("data/07_model_output/RL_data_prod_y_proba.parquet")

    return data_y_proba_df, data_y_pred_df, metrics_RL

def get_metrics_prod(data_y_proba, data_y_pred, data_y_actual, model_name):

    log_loss_prod = log_loss(data_y_actual, np.array(data_y_proba)[:, 1])
    f1_score_prod = f1_score(data_y_actual, data_y_pred)
    acuracia_prod = accuracy_score(data_y_actual, data_y_pred),
    precisao_prod = precision_score(data_y_actual, data_y_pred),
    recall_prod = recall_score(data_y_actual, data_y_pred)
    roc_auc_prod = salvar_curva_roc(data_y_actual, np.array(data_y_proba), model_name)

    metrics_prod = pd.DataFrame({
        "Metrica": ["Log Loss", "F1 Score", "Acurácia", "Precisão", "Recall", "AUC"],
        "Valor": [log_loss_prod, f1_score_prod, acuracia_prod, precisao_prod, recall_prod, roc_auc_prod]
    })

    mlflow.log_metric(f"{model_name}_prod_roc_auc", roc_auc_prod)
    mlflow.log_metric(f"{model_name}_prod_log_loss", log_loss_prod)
    mlflow.log_metric(f"{model_name}_prod_f1_score", f1_score_prod)

    return metrics_prod

def salvar_curva_roc(y_true, y_pred_proba, model_name):

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    plt.xlabel('Taxa Falsos Positivos')
    plt.ylabel('Taxa Verdadeiros Positivos')
    plt.title(f'Curva ROC produção - {model_name}')
    plt.legend(loc="lower right")
    
    plt.savefig(f'data/08_reporting/roc_curve_prod_{model_name}.png')
    plt.close()
    
    return roc_auc