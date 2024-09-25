import pandas as pd
import numpy as np
import torch
from datetime import timedelta
from torch.utils.data import TensorDataset

file = "pkl/uber-raw-data-combined.csv.pkl" # read pkl file
print(f"Reading {file}")
df = pd.read_pickle(file)
main_matrix, ftensor = None, None

if (df["Date/Time"].dtypes != np.datetime64):
    df["Date/Time"] = pd.to_datetime(df["Date/Time"]) # ensures all dates are in datetime format
print(f"Shape: {df.shape}")

lat = df["Lat"]
lon = df["Lon"]
time = df["Date/Time"]


def bucket(lat, lon, time, lat_bucket_size, lon_bucket_size, time_bucket_period):
    print("Bucketing values")
    df["lat"] = (lat // lat_bucket_size).round() * lat_bucket_size # buckets values specified
    df["lon"] = (lon // lon_bucket_size).round() * lon_bucket_size
    df["time"] = time.dt.to_period(time_bucket_period).dt.start_time
    # matrix = df.groupby([df["time_bucket"], df["lat_bucket"], df["lon_bucket"]]).size().reset_index(name="count") #used to get the count
    df["time"] = (df["time"] - df["time"].min()).dt.total_seconds()
    
    return df[["lat", "lon", "time"]].copy()

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
    print("Matrix column split")
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
    print("Making global tensor")
    global main_matrix
    if main_matrix is None or main_matrix.empty: # creates the full matrix once, to allow loops on function
        main_matrix = bucket(lat, lon, time, 0.1, 0.1, "min") # matrix is now bucketed, containing a count column
    print(main_matrix)
    main_matrix.to_pickle("torch_process.pkl")
    matrix = pd.read_pickle("torch_process.pkl")
    global ftensor
    ftensor = TensorDataset(torch.tensor(matrix.values, dtype=torch.float32)) # creates torch tensor from new matrix
    print("Global tensor created")
    
# get stack data from a subset
def get_subset_tensor(subset):
    return torch.stack([data[0] for data in subset])

def split(train, test, validation):
    if (train + test + validation != 1):
        print("Make sure percents add up to 100")
        return
    print("Creating Tensor")
    create_tensor()
    print("Spliting tensor")
    train_x, test_x, validation_x = torch.utils.data.random_split(ftensor, [train, test, validation])
    print("Split tensor done")
    print("Subset to tensor")
    # torch tensor -> tensor for logging
    train_x = get_subset_tensor(train_x)
    test_x = get_subset_tensor(test_x)
    validation_x = get_subset_tensor(validation_x)
    print("Subset to tensor done")
    # log tensors in console
    print("Training Tensor:\n", train_x, train_x.shape)
    print("Testing Tensor:\n", test_x, test_x.shape)
    print("Validation Tensor:\n", validation_x, validation_x.shape)
    return train_x, test_x, validation_x