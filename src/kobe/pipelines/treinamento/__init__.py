"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""

from .pipeline import create_pipeline
from .CustomMLflowModel import CustomMLflowModel

__all__ = ["create_pipeline", "CustomMLflowModel"]

__version__ = "0.1"
