"""
This is a boilerplate pipeline 'aplicacao_prod'
generated using Kedro 0.19.12
"""
from sklearn.metrics import log_loss, f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

def pre_process_predict(data):

    data_processed = data.dropna()
    data_processed_x = (data_processed[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance']]
        .assign(playoffs = lambda x: x['playoffs'].astype(bool))
       )   
    data_processed_x['lat_quadra'] = data_processed_x['lat'] - 34.0443
    data_processed_x['lon_quadra'] = data_processed_x['lon'] + 118.2698
    data_processed_x = data_processed_x.drop(columns=["lat", "lon"])

    data_processed_y = data_processed[['shot_made_flag']]

    return data_processed_x, data_processed_y


def predict(model, data_x):

    data_y_proba = model.predict_proba(data_x)
    data_y_pred = model.predict(data_x)

    return pd.DataFrame(data_y_proba), pd.DataFrame(data_y_pred)

def get_metrics_prod(data_y_proba, data_y_pred, data_y_actual):

    log_loss_prod = log_loss(data_y_actual, np.array(data_y_proba)[:, 1])
    f1_score_prod = f1_score(data_y_actual, data_y_pred)
    acuracia_prod = accuracy_score(data_y_actual, data_y_pred),
    precisao_prod = precision_score(data_y_actual, data_y_pred),
    recall_prod = recall_score(data_y_actual, data_y_pred)


    metrics_prod = pd.DataFrame({
        "Metrica": ["Log Loss", "F1 Score", "Acurácia", "Precisão", "Recall"],
        "Valor": [log_loss_prod, f1_score_prod, acuracia_prod, precisao_prod, recall_prod]
    })
    return metrics_prod


# def get_f1_and_log_loss_and_predictions(data_prod, randcv_model):
    
#     test_predictions = exp.predict_model(randcv_model, data=data_test)
#     test_predicted_target = test_predictions['prediction_label'].astype(int)

#     test_probs = randcv_model.predict_proba(data_test.drop(columns=["shot_made_flag"]))
#     test_actual_target = data_test['shot_made_flag'].astype(int)

#     # Log Loss
#     test_log_loss = log_loss(test_actual_target, test_probs[:, 1])
    
#     # F1 score
#     test_f1 = f1_score(test_actual_target, test_predicted_target)

#     return test_log_loss, test_f1,test_predicted_target, test_probs