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
            outputs="feature_dev",
            name="process_data_node"
        )
    ])
