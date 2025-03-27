"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import *

def train_model_pycaret_DT(data_train, data_test, session_id):

    exp = ClassificationExperiment()
    # desabilitando o log_experiment do Pycaret para evitar conflito com o MLflow
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id, log_experiment=False)

    mlflow.log_param("DT_model_type", "Decision Tree")
    mlflow.log_param("DT_data_train_shape", str(data_train.shape))
    mlflow.log_param("DT_data_test_shape", str(data_test.shape))
    
    # Treinar o modelo
    dt_model = exp.create_model('dt')


    # Versão mais nova do PyCaret pode retornar diretamente o modelo sklearn
    randcv_model = exp.tune_model(dt_model, n_iter=100, optimize='F1')
    # Verificar se o objeto é um modelo sklearn ou um container do PyCaret
    if hasattr(randcv_model, 'model'):
        model_obj = randcv_model.model
    else:
        model_obj = randcv_model
   
            
    # Avaliar o modelo e obter as métricas
    train_predictions = exp.predict_model(model_obj, data=data_train)
    test_predictions = exp.predict_model(model_obj, data=data_test)
    
    # Extrair as previsões de probabilidade
    train_probs = train_predictions['prediction_score'].values
    test_probs = test_predictions['prediction_score'].values
    
    # Extrair os valores reais
    train_actual = data_train['shot_made_flag'].astype(int).values
    test_actual = data_test['shot_made_flag'].astype(int).values
    
    # Calcular métricas - log loss
    train_log_loss = log_loss(train_actual, train_probs)
    test_log_loss = log_loss(test_actual, test_probs)
    
    # Calcular métricas - F1 score
    train_predictions_binary = (train_probs >= 0.5).astype(int)
    test_predictions_binary = (test_probs >= 0.5).astype(int)
    train_f1 = f1_score(train_actual, train_predictions_binary)
    test_f1 = f1_score(test_actual, test_predictions_binary)
    
    # Registrar métricas no MLflow
    mlflow.log_metric("DT_train_log_loss", train_log_loss)
    mlflow.log_metric("DT_test_log_loss", test_log_loss)
    mlflow.log_metric("DT_train_f1_score", train_f1)
    mlflow.log_metric("DT_test_f1_score", test_f1)
    
    # Registrar o modelo no MLflow
    mlflow.sklearn.log_model(
        model_obj, 
        "decision_tree_model",
        registered_model_name="DecisionTreeModel"
    )
    
    # # Registrar parâmetros importantes do modelo
    # try:
    #     # Tentar acessar diretamente os parâmetros do modelo
    #     params = model_obj.get_params()
    #     for param_name, param_value in params.items():
    #         if not isinstance(param_value, (list, dict, set)) and param_value is not None:
    #             mlflow.log_param(f"model_{param_name}", str(param_value))
    # except:
    #     # Se não conseguir acessar os parâmetros diretamente, registrar uma string
    #     mlflow.log_param("model_info", str(model_obj))
    
    return model_obj


