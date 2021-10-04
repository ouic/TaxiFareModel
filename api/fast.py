
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return dict(greeting="hello")


@app.get("/predict")
def predict(pickup_datetime,  # 2013-07-06 17:18:00
            lon1,             # -73.950655
            lat1,             # 40.783282
            lon2,             # -73.984365
            lat2,             # 40.769802
            passcount):       # 1

    # ⚠️ no timezone conversion here: the user is assumed to provide an UTC datetime

    # step 1 : convert params to dataframe
    X_pred = pd.DataFrame({
        "key": ["truc"],
        "pickup_datetime": [pickup_datetime + " UTC"],
        "pickup_longitude": [float(lon1)],
        "pickup_latitude": [float(lat1)],
        "dropoff_longitude": [float(lon2)],
        "dropoff_latitude": [float(lat2)],
        "passenger_count": [int(passcount)]})

    # print(X_pred)
    # print(X_pred.columns)
    # print(X_pred.dtypes)

    # step 2 : load the trained model
    pipeline = joblib.load("model.joblib")
    # print(pipeline)

    # step 3 : make a prediction
    y_pred = pipeline.predict(X_pred)
    # print(type(y_pred))

    # step 4 : return the prediction (extract the prediction value from the ndarray)
    # print(y_pred)
    prediction = y_pred[0]

    return {"pred": prediction}
