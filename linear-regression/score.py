#%%writefile score.py
import json
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.core.model import Model

def init():
    global model
    # retreive the path to the model file using the model name
    #model_path = Model.get_model_path('toyota-lin-regression')
    #model = joblib.load(model_path)
    model = joblib.load("./linear-regression.pkl")

def run(input_df):
    y_hat = model.predict(input_df)
    return json.dumps(str(y_hat[0]))

def main():
    #json_string = '{"Id":1, "Price": 13500}'
    df = pd.DataFrame(data=[[1, 13500]], columns=['Id', 'Price'])
    init()
    print("Result: " + run(df))

    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

    # Generate the service_schema.json
    generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')
    print("Schema generated")


if __name__ == "__main__":
    main()