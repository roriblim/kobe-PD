"""
This is a boilerplate pipeline 'preparacao_dados'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
            nodes.prepare_data,
            inputs="raw_dev",
            outputs="primary_dev",
            name="process_data_node"
        ),
        node(
            nodes.feature_engineering,
            inputs="primary_dev",
            outputs="feature_dev",
            name="feature_engineering_node"
        ),
        node(
            nodes.separacao_treino_teste,
            inputs=["feature_dev", "params:stratified_random_state", "params:test_size"],
            outputs=["data_train", "data_test", "data_train_csv", "data_test_csv"],
            name="separacao_treino_teste_node"
        )
    ])
