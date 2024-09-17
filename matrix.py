import pandas as pd
import numpy as np
import torch
import random
from datetime import timedelta

file = "pkl/uber-raw-data-combined.csv.pkl" # read pkl file
df = pd.read_pickle(file)
main_matrix = 0

if (df["Date/Time"].dtypes != np.datetime64):
    df["Date/Time"] = pd.to_datetime(df["Date/Time"]) # ensures all dates are in datetime format
print(f"Shape: {df.shape}")

lat = df["Lat"]
lon = df["Lon"]
time = df["Date/Time"]


def bucket(lat, lon, time, lat_bucket_size, lon_bucket_size, time_bucket_period):
    df["lat_bucket"] = (lat // lat_bucket_size) * 10 * lat_bucket_size # buckets values specified
    df["lon_bucket"] = (lon // lon_bucket_size) * 10 * lon_bucket_size
    df["time_bucket"] = time.dt.to_period(time_bucket_period).dt.start_time
    matrix = df.groupby([df["time_bucket"], df["lat_bucket"], df["lon_bucket"]]).size().reset_index(name="count") #used to get the count

    return matrix

def preprocess(matrix):
    '''
    if randomize: # creates a randomized time matrix window
        range1 = matrix["time_bucket"].max() + timedelta(seconds=random.randint(0, int((matrix["time_bucket"].min() # gets random datetime value in range of dataset
                                                                                        - matrix["time_bucket"].max())
                                                                                       .total_seconds())))
        range2  = matrix["time_bucket"].max() + timedelta(seconds=random.randint(0, int((range1 -  #gets random datetime value in range of range1 to max
                                                                                         matrix["time_bucket"].max())
                                                                                        .total_seconds())))
        matrix = df[(df["Date/Time"] >= range1) & (df["Date/Time"] <= range2)] # filters dataset to only have between range1 and range2 values
    '''
    
    # break down matrix into time sections
    matrix["year"] = matrix["time_bucket"].dt.year
    matrix["month"] = matrix["time_bucket"].dt.month
    matrix["day"] = matrix["time_bucket"].dt.day
    
    matrix["hour"] = matrix["time_bucket"].dt.hour
    matrix["minute"] = matrix["time_bucket"].dt.minute
    matrix["second"] = matrix["time_bucket"].dt.second

    matrix["total_seconds"] = matrix["hour"] * 3600 + matrix["minute"] * 60 + matrix["second"] #converts hour, min, sec all into seconds
    matrix = matrix.drop(columns=["time_bucket", "hour", "minute", "second"])
    
    print(matrix)
    matrix.to_pickle("torch_process.pkl")
    
def create_tensor():
    if main_matrix == 0: # creates the full matrix once, to allow loops on function
        global main_matrix
        main_matrix = bucket(lat, lon, time, 0.1, 0.1, "min") # matrix is now bucketed, containing a count column
    preprocess(main_matrix) # creates matrix, with true = randomize
    matrix = pd.read_pickle("torch_process.pkl")
    tensor = torch.tensor(matrix.values, dtype=torch.float32) # creates torch tensor from new matrix
    return tensor

create_tensor()