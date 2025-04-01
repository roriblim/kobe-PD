"""
This is a boilerplate pipeline 'aplicacao_prod'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.teste,
            inputs=["best_model"],
            outputs=None,
            name="teste"
        )
    ])
