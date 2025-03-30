"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.train_model_pycaret_DT,
            inputs=["data_train", "data_test", "params:DT_session_id"],
            outputs=["model_DT", "test_metrics_DT", "test_probs_DT"],
            name="train_model_pycaret_DT_node"
        ),
        node(
            nodes.train_model_pycaret_RL,
            inputs=["data_train", "data_test", "params:RL_session_id"],
            outputs=["model_RL", "test_metrics_RL", "test_probs_RL"],
            name="train_model_pycaret_RL_node"
        ),
        node(
            nodes.compare_models,
            inputs=["model_DT", "model_RL", "test_metrics_DT", "test_metrics_RL"],
            outputs=["best_model","best_model_test_metrics"],
            name="compare_models_node"
        ),
        node(
            nodes.create_API_model,
            inputs=["best_model", "best_model_test_metrics"],
            outputs="api_model",
            name="create_API_model_node"
        )
    ])
