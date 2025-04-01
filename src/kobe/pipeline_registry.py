"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines

from kobe.pipelines import preparacao_dados, treinamento, aplicacao_prod  


from kedro.pipeline import Pipeline




def register_pipelines() -> dict[str, Pipeline]:

    pipelines = {
        "preparacao_dados": preparacao_dados.create_pipeline(),
        "treinamento": treinamento.create_pipeline(),
        "aplicacao_prod": aplicacao_prod.create_pipeline(),
        "__default__": preparacao_dados.create_pipeline() + treinamento.create_pipeline() + aplicacao_prod.create_pipeline(),
    }
    return pipelines


