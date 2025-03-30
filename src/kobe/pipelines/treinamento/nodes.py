"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
# from kobe.pipelines.treinamento.CustomMLflowModel import CustomMLflowModel
from kobe.pipelines.treinamento.CustomMLflowModel import CustomMLflowModel
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import ClassificationExperiment
import pickle
from pathlib import Path

def train_model_pycaret_DT(data_train, data_test, session_id):

    exp = ClassificationExperiment()
    # desabilitando o log_experiment do Pycaret para evitar conflito com o MLflow
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id, log_experiment=False)

    mlflow.log_param("DT_model_type", "Decision Tree")
    mlflow.log_param("DT_data_train_shape", str(data_train.shape))
    mlflow.log_param("DT_data_test_shape", str(data_test.shape))
    

    dt_model = exp.create_model('dt')
    randcv_model = exp.tune_model(dt_model, n_iter=100, optimize='F1')

    test_log_loss, test_f1, test_predictions_binary, test_probs = get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model)
    
    metrics = exp.pull()

    DT_predictions_parquet = "data/07_model_output/DT_test_predictions.parquet"
    pd.DataFrame(test_predictions_binary).to_parquet(DT_predictions_parquet, index=False)

    DT_predictions_csv = "data/07_model_output/DT_test_predictions.csv"
    pd.DataFrame(test_predictions_binary).to_csv(DT_predictions_csv, index=False)

    # Registros
    mlflow.log_metric("DT_test_log_loss", test_log_loss)
    mlflow.log_metric("DT_test_f1_score", test_f1)
    mlflow.log_artifact(DT_predictions_parquet)
    mlflow.log_artifact(DT_predictions_csv)
    
    salvar_modelo_pickle(randcv_model,"DT_model.pkl")

    return randcv_model, metrics, pd.DataFrame(test_probs)


def train_model_pycaret_RL(data_train, data_test, session_id):

    exp = ClassificationExperiment()
    # desabilitando o log_experiment do Pycaret para evitar conflito com o MLflow
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id, log_experiment=False)

    mlflow.log_param("RL_model_type", "Regressão Logística")
    mlflow.log_param("RL_data_train_shape", str(data_train.shape))
    mlflow.log_param("RL_data_test_shape", str(data_test.shape))
    
    rl_model = exp.create_model('lr')
    randcv_model = exp.tune_model(rl_model, n_iter=100, optimize='F1')
    
    test_log_loss, test_f1, test_predictions_binary, test_probs = get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model)
    
    metrics = exp.pull()

    RL_predictions_parquet = "data/07_model_output/RL_test_predictions.parquet"
    pd.DataFrame(test_predictions_binary).to_parquet(RL_predictions_parquet, index=False)

    RL_predictions_csv = "data/07_model_output/RL_test_predictions.csv"
    pd.DataFrame(test_predictions_binary).to_csv(RL_predictions_csv, index=False)

    # Registros
    mlflow.log_metric("RL_test_log_loss", test_log_loss)
    mlflow.log_metric("RL_test_f1_score", test_f1)
    mlflow.log_artifact(RL_predictions_parquet)
    mlflow.log_artifact(RL_predictions_csv)
    
    salvar_modelo_pickle(randcv_model,"RL_model.pkl")

    return randcv_model, metrics, pd.DataFrame(test_probs)


def compare_models(model1, model2, test_metrics_1, test_metrics_2):
        
    if test_metrics_1.loc[0, "F1"]  > test_metrics_2.loc[0, "F1"] :
        best_model=model1
        best_model_test_metrics=test_metrics_1
    else:
        best_model=model2
        best_model_test_metrics=test_metrics_2

    salvar_modelo_pickle(best_model,"best_model.pkl")
    
    return best_model, best_model_test_metrics

def get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model):
    
    test_predictions = exp.predict_model(randcv_model, data=data_test)
    test_predicted_target = test_predictions['prediction_label']

    test_probs = randcv_model.predict_proba(data_test.drop(columns=["shot_made_flag"]))
    test_actual_target = data_test['shot_made_flag']

    # Log Loss
    test_log_loss = log_loss(test_actual_target, test_probs)
    
    # F1 score
    test_f1 = f1_score(test_actual_target, test_predicted_target)

    return test_log_loss, test_f1,test_predicted_target, test_probs

def create_API_model(best_model, model_test_metrics):

    api_model = CustomMLflowModel(best_model)
    salvar_modelo_pickle(api_model, "api_model.pkl")
    
    return api_model

def salvar_modelo_pickle(randcv_model,nome_modelo):
    save_path = Path("data/06_models/")
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / nome_modelo
    with open(model_path, "wb") as file:
        pickle.dump(randcv_model, file)



