"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score, roc_curve, auc
from pycaret.classification import ClassificationExperiment
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



def train_model_pycaret_DT(data_train, data_test, session_id):

    test_actual_target = data_test['shot_made_flag'].astype(int)

    exp = ClassificationExperiment()
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id)

    mlflow.log_param("DT_model_type", "Decision Tree")
    mlflow.log_param("DT_data_train_shape", str(data_train.shape))
    mlflow.log_param("DT_data_test_shape", str(data_test.shape))

    dt_model = exp.create_model('dt', max_depth=3)
    randcv_model = exp.tune_model(dt_model, n_iter=400, optimize='F1')
    model="DT"

    test_predicted_target, test_probs = get_and_save_predictions(data_test, exp, randcv_model, model)
    metrics = get_and_save_metrics(test_actual_target, exp, test_predicted_target, test_probs, model)

    salvar_modelo_pickle(randcv_model,"DT_model.pkl")

    if hasattr(randcv_model, "get_params"):
        params = randcv_model.get_params()
        mlflow.log_params({f"hiperparametro_DT__{k}": v for k, v in params.items()})

    importances = randcv_model.feature_importances_
    feature_names = data_train.drop(columns=["shot_made_flag"]).columns

    plotar_importancia_features(model, importances, feature_names)

    return randcv_model, metrics, pd.DataFrame(test_probs)


def train_model_pycaret_RL(data_train, data_test, session_id):

    test_actual_target = data_test['shot_made_flag'].astype(int)

    exp = ClassificationExperiment()
    
    exp.setup(data=data_train, target='shot_made_flag', session_id=session_id)

    mlflow.log_param("RL_model_type", "Regressão Logística")
    mlflow.log_param("RL_data_train_shape", str(data_train.shape))
    mlflow.log_param("RL_data_test_shape", str(data_test.shape))
    
    rl_model = exp.create_model('lr')
    randcv_model = exp.tune_model(rl_model, n_iter=100, optimize='F1')
    model="RL"

    test_predicted_target, test_probs = get_and_save_predictions(data_test, exp, randcv_model, model)
    metrics = get_and_save_metrics(test_actual_target, exp, test_predicted_target, test_probs, model)

    salvar_modelo_pickle(randcv_model,"RL_model.pkl")

    if hasattr(randcv_model, "get_params"):
        params = randcv_model.get_params()
        mlflow.log_params({f"hiperparametro_RL__{k}": v for k, v in params.items()})

    coefs = randcv_model.coef_[0]
    importances = np.abs(coefs)
    feature_names = data_train.drop(columns=["shot_made_flag"]).columns

    plotar_importancia_features(model, importances, feature_names)

    return randcv_model, metrics, pd.DataFrame(test_probs)

def plotar_importancia_features(model, importances, feature_names):
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df)
    plt.title(f'Importância das Features - {model}')
    plt.tight_layout()
    plt.savefig(f'data/08_reporting/{model}_feature_importance.png')
    plt.close()

def get_and_save_metrics(test_actual_target, exp, test_predicted_target, test_probs, model):

    test_log_loss = log_loss(test_actual_target, test_probs[:, 1])
    test_f1 = f1_score(test_actual_target, test_predicted_target)
    roc_auc = salvar_curva_roc(test_actual_target, test_probs, model)
    
    metrics = exp.pull()
    
    metrics['roc_auc'] = roc_auc
    metrics['test_log_loss'] = test_log_loss
    metrics['test_f1'] = test_f1

    mlflow.log_metric(f"{model}_test_roc_auc", roc_auc)
    mlflow.log_metric(f"{model}_test_log_loss", test_log_loss)
    mlflow.log_metric(f"{model}_test_f1_score", test_f1)

    return metrics

def get_and_save_predictions(data_test, exp, randcv_model, model):
    test_predictions = exp.predict_model(randcv_model, data=data_test)
    test_predicted_target = test_predictions['prediction_label'].astype(int)

    test_probs = randcv_model.predict_proba(data_test.drop(columns=["shot_made_flag"]))

    predictions_parquet = f"data/07_model_output/{model}_test_predictions.parquet"
    pd.DataFrame(test_predicted_target).to_parquet(predictions_parquet, index=False)

    predictions_csv = f"data/07_model_output/{model}_test_predictions.csv"
    pd.DataFrame(test_predicted_target).to_csv(predictions_csv, index=False)

    mlflow.log_artifact(predictions_parquet)
    mlflow.log_artifact(predictions_csv)

    return test_predicted_target,test_probs


def compare_models(model1, model2, test_metrics_1, test_metrics_2):
        
    if test_metrics_1.loc[0, "roc_auc"]  > test_metrics_2.loc[0, "roc_auc"] :
        best_model=model1
        best_model_test_metrics=test_metrics_1
    else:
        best_model=model2
        best_model_test_metrics=test_metrics_2

    salvar_modelo_pickle(best_model,"best_model.pkl")
    
    return best_model, best_model_test_metrics

def salvar_modelo_pickle(randcv_model,nome_modelo):
    save_path = Path("data/06_models/")
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / nome_modelo
    with open(model_path, "wb") as file:
        pickle.dump(randcv_model, file)
    mlflow.sklearn.log_model(randcv_model, nome_modelo)

def salvar_curva_roc(y_true, y_pred_proba, modelo_nome):

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    plt.xlabel('Taxa Falsos Positivos')
    plt.ylabel('Taxa Verdadeiros Positivos')
    plt.title(f'Curva ROC - {modelo_nome}')
    plt.legend(loc="lower right")
    
    plt.savefig(f'data/08_reporting/roc_curve_test_{modelo_nome}.png')
    plt.close()
    
    return roc_auc
