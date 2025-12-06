"""Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

python3 -m make_predictions.py

"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del experimento que se desea cargar
## Se puede obtener el run_id desde la UI de MLflow

loaded_model = "runs:/29cd9a7864024515b55d68e70bef3f83/model"
loaded_model = mlflow.pyfunc.load_model(loaded_model)
y = loaded_model.predict(x)

print(y)
