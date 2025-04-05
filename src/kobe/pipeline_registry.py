"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines

from kobe.pipelines import preparacao_dados, treinamento, aplicacao_prod  


from kedro.pipeline import Pipeline



def register_pipelines() -> dict[str, Pipeline]:

    preparacao = preparacao_dados.create_pipeline()
    treino = treinamento.create_pipeline()
    aplicacao = aplicacao_prod.create_pipeline()

    pipelines = {
        "preparacao_dados": preparacao,
        "treinamento": treino,
        "aplicacao_prod": aplicacao,
        "__default__": preparacao + treino + aplicacao,
    }
    return pipelines
