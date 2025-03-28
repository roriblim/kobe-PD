"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import ClassificationExperiment

def train_model_pycaret_DT(data_train, data_test, session_id):

    exp = ClassificationExperiment()
    # desabilitando o log_experiment do Pycaret para evitar conflito com o MLflow
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id, log_experiment=False)

    mlflow.log_param("DT_model_type", "Decision Tree")
    mlflow.log_param("DT_data_train_shape", str(data_train.shape))
    mlflow.log_param("DT_data_test_shape", str(data_test.shape))
    
    dt_model = exp.create_model('dt')
    randcv_model = exp.tune_model(dt_model, n_iter=100, optimize='F1')

    test_log_loss, test_f1, test_predictions_binary = get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model)
    
    DT_predictions_parquet = "data/06_model/DT_test_predictions.parquet"
    pd.DataFrame(test_predictions_binary).to_parquet(DT_predictions_parquet, index=False)

    DT_predictions_csv = "data/06_model/DT_test_predictions.csv"
    pd.DataFrame(test_predictions_binary).to_csv(DT_predictions_csv, index=False)

    # Registro no MLFlow
    mlflow.log_metric("DT_test_log_loss", test_log_loss)
    mlflow.log_metric("DT_test_f1_score", test_f1)
    mlflow.log_artifact(DT_predictions_parquet)
    mlflow.log_artifact(DT_predictions_csv)

    return randcv_model

def train_model_pycaret_RL(data_train, data_test, session_id):

    exp = ClassificationExperiment()
    # desabilitando o log_experiment do Pycaret para evitar conflito com o MLflow
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id, log_experiment=False)

    mlflow.log_param("RL_model_type", "Regressão Logística")
    mlflow.log_param("RL_data_train_shape", str(data_train.shape))
    mlflow.log_param("RL_data_test_shape", str(data_test.shape))
    
    rl_model = exp.create_model('lr')
    randcv_model = exp.tune_model(rl_model, n_iter=100, optimize='F1')
    
    test_log_loss, test_f1, test_predictions_binary = get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model)
    
    RL_predictions_parquet = "data/06_model/RL_test_predictions.parquet"
    pd.DataFrame(test_predictions_binary).to_parquet(RL_predictions_parquet, index=False)

    RL_predictions_csv = "data/06_model/RL_test_predictions.csv"
    pd.DataFrame(test_predictions_binary).to_csv(RL_predictions_csv, index=False)

    # Registro no MLFlow
    mlflow.log_metric("RL_test_log_loss", test_log_loss)
    mlflow.log_metric("RL_test_f1_score", test_f1)
    mlflow.log_artifact(RL_predictions_parquet)
    mlflow.log_artifact(RL_predictions_csv)
    return randcv_model


def compare_models(model1, model2):

    # TO DO
    best_model=model1
    
    return best_model

def get_f1_and_log_loss_and_predictions(data_test, exp, randcv_model):
    
    test_predictions = exp.predict_model(randcv_model, data=data_test)
    test_predicted_target = test_predictions['prediction_label']

    test_probs = randcv_model.predict_proba(data_test.drop(columns=["shot_made_flag"]))
    test_actual_target = data_test['shot_made_flag']

    # Log Loss
    test_log_loss = log_loss(test_actual_target, test_probs)
    
    # F1 score
    test_f1 = f1_score(test_actual_target, test_predicted_target)

    return test_log_loss, test_f1,test_predicted_target


