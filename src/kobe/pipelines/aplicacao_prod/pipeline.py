"""
This is a boilerplate pipeline 'aplicacao_prod'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.pre_process_predict,
            inputs=["raw_prod"],
            outputs=["data_prod_x","data_prod_y"],
            name="pre_process_prod_predict_node"
        ),
        node(
            nodes.predict,
            inputs=["model_DT","data_prod_x"],
            outputs=["DT_data_prod_y_proba","DT_data_prod_y_pred"],
            name="DT_prod_predict_node"
        ),
         node(
            nodes.predict,
            inputs=["model_RL","data_prod_x"],
            outputs=["RL_data_prod_y_proba","RL_data_prod_y_pred"],
            name="RL_prod_predict_node"
        ),  
        node(
            nodes.get_metrics_prod,
            inputs=["DT_data_prod_y_proba","DT_data_prod_y_pred","data_prod_y"],
            outputs="DT_metrics_prod",
            name="DT_prod_metrics_node"
        ),
        node(
            nodes.get_metrics_prod,
            inputs=["RL_data_prod_y_proba","RL_data_prod_y_pred","data_prod_y"],
            outputs="RL_metrics_prod",
            name="RL_prod_metrics_node"
        ),                
    ])

