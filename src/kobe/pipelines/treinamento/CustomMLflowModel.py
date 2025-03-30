
import mlflow
import mlflow.pyfunc
import pickle
import pandas as pd

import mlflow.pyfunc
import pandas as pd

class CustomMLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model  #

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:

        data = (data[['lat', 'lon','minutes_remaining','period','playoffs','shot_distance', 'shot_made_flag','playoffs']]
        .assign(playoffs = lambda x: x['playoffs'].astype(bool))
        )
        # if "playoffs" in data.columns:
        #     data["playoffs"] = data["playoffs"].astype(bool)

        if {"lat", "lon"}.issubset(data.columns):
            data["lat_quadra"] = data["lat"] - 34.0443
            data["lon_quadra"] = data["lon"] + 118.2698
            data = data.drop(columns=["lat", "lon"])
        
        print("printando para debug")
        print(data)
        return data

    def predict(self, context, model_input: pd.DataFrame):

        processed_data = self.preprocess(model_input)
        return self.model.predict(processed_data) 
