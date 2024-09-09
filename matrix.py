import pandas as pd
import numpy as np
import torch

file = "pkl/uber-raw-data-combined.csv.pkl"
df = pd.read_pickle(file)

if (df["Date/Time"].dtypes != np.datetime64):
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
print(f"Shape: {df.shape}")

lat = df["Lat"]
lon = df["Lon"]
time = df["Date/Time"]


def bucket(lat, lon, time, lat_bucket_size, lon_bucket_size, time_bucket_period):
    df["lat_bucket"] = (lat // lat_bucket_size) * lat_bucket_size
    df["lon_bucket"] = (lon // lon_bucket_size) * lon_bucket_size
    df["time_bucket"] = time.dt.to_period(time_bucket_period).dt.start_time
    matrix = df.groupby([df["time_bucket"], df["lat_bucket"], df["lon_bucket"]]).size().reset_index(name="count")

    return matrix

def preprocess():
    matrix = bucket(lat, lon, time, 0.1, 0.1, "min")
    matrix["year"] = matrix["time_bucket"].dt.year
    matrix["month"] = matrix["time_bucket"].dt.month
    matrix["day"] = matrix["time_bucket"].dt.day
    
    matrix["hour"] = matrix["time_bucket"].dt.hour
    matrix["minute"] = matrix["time_bucket"].dt.minute
    matrix["second"] = matrix["time_bucket"].dt.second

    matrix["total_seconds"] = matrix["hour"] * 3600 + matrix["minute"] * 60 + matrix["second"]
    matrix = matrix.drop(columns=["time_bucket", "hour", "minute", "second"])
    
    print(matrix)
    matrix.to_pickle("torch_process.pkl")
    
def create_tensor():
    preprocess()
    matrix = pd.read_pickle("torch_process.pkl")
    tensor = torch.tensor(matrix.values, dtype=torch.float32)
    return tensor