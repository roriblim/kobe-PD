import mlflow
import mlflow.pyfunc
import pickle
import pandas as pd

class CustomMLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def load_context(self, context):
        print("Modelo carregado com sucesso")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = (data[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance']]
        # .assign(playoffs = lambda x: x['playoffs'].astype(bool))
        )        
        if "playoffs" in data.columns:
            data["playoffs"] = data["playoffs"].astype(bool)
        
        if {"lat", "lon"}.issubset(data.columns):
            data["lat_quadra"] = data["lat"] - 34.0443
            data["lon_quadra"] = data["lon"] + 118.2698
            data = data.drop(columns=["lat", "lon"])
        
        return data

    def predict(self, context, model_input: pd.DataFrame):

        processed_data = self.preprocess(model_input)
        predict_result = self.model.predict(processed_data)
        predict_proba_result = self.model.predict_proba(processed_data)

        return {"predictions": predict_result, "predict_proba": predict_proba_result}
