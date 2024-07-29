import os
import pickle
import numpy as np

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

os.environ["MODEL_FILE_NAME"] = 'iris_model.pkl'

model_file_name = os.getenv("MODEL_FILE_NAME")

# Load the trained model
with open(model_file_name, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()


# Define input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Define output data model
class IrisResponse(BaseModel):
    prediction: str


# Define the prediction function
def predict_iris_species(input_data):
    # Convert the input data into a numpy array
    input_array = np.array([
        input_data.sepal_length, input_data.sepal_width,
        input_data.petal_length, input_data.petal_width
    ]).reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = model.predict(input_array)[0]

    # Map the integer prediction to the corresponding species name
    species_map = {
        0: 'Iris Setosa',
        1: 'Iris Versicolour',
        2: 'Iris Virginica'
    }
    prediction_name = species_map[prediction]

    # Return the prediction result
    return IrisResponse(prediction=prediction_name)


# Define the endpoint for single prediction
@app.post("/predict")
def predict_single(data: IrisData):
    return predict_iris_species(data)


# Define the endpoint for batch prediction
@app.post("/predict_batch")
def predict_batch(data: List[IrisData]):
    predictions = [predict_iris_species(item) for item in data]
    return predictions
