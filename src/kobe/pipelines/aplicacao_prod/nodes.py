"""
This is a boilerplate pipeline 'aplicacao_prod'
generated using Kedro 0.19.12
"""



def pre_process(data):

    return None

def teste(model):

    return None



def get_f1_and_log_loss_and_predictions(data_prod, randcv_model):
    
    test_predictions = exp.predict_model(randcv_model, data=data_test)
    test_predicted_target = test_predictions['prediction_label'].astype(int)

    test_probs = randcv_model.predict_proba(data_test.drop(columns=["shot_made_flag"]))
    test_actual_target = data_test['shot_made_flag'].astype(int)

    # Log Loss
    test_log_loss = log_loss(test_actual_target, test_probs[:, 1])
    
    # F1 score
    test_f1 = f1_score(test_actual_target, test_predicted_target)

    return test_log_loss, test_f1,test_predicted_target, test_probs