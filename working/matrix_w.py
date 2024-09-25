import pandas as pd
import numpy as np
import torch
from datetime import timedelta
from torch.utils.data import TensorDataset

file = "pkl/uber-raw-data-combined.csv.pkl" # read pkl file
print(f"Reading {file}")
df = pd.read_pickle(file)
ftensor = None
time_steps = 0

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
    df["time"] = (df["time"] - df["time"].min()).dt.total_seconds()
    # matrix = df.groupby([df["time_bucket"], df["lat_bucket"], df["lon_bucket"]]).size().reset_index(name="count") #used to get the count
    lat_min = df["lat"].min()  # Assuming latitude is in the first column
    lat_max = df["lat"].max()

    long_min = df["lon"].min()  # Assuming longitude is in the second column
    long_max = df["lon"].max()

    print(f"Latitude range: {lat_min} to {lat_max}")
    print(f"Longitude range: {long_min} to {long_max}")
    lat_dim = int((lat_max - lat_min) / 0.1)  # Define your lat_resolution
    long_dim = int((long_max - long_min) / 0.1)  # Define your long_resolution
    print(lat_dim, long_dim)
    
    
    return df[["lat", "lon", "time"]].copy()

# def preprocess(matrix): discontinued, save for future use if needed
#     '''
#     if randomize: # creates a randomized time matrix window
#         range1 = matrix["time_bucket"].max() + timedelta(seconds=random.randint(0, int((matrix["time_bucket"].min() # gets random datetime value in range of dataset
#                                                                                         - matrix["time_bucket"].max())
#                                                                                        .total_seconds())))
#         range2  = matrix["time_bucket"].max() + timedelta(seconds=random.randint(0, int((range1 -  #gets random datetime value in range of range1 to max
#                                                                                          matrix["time_bucket"].max())
#                                                                                         .total_seconds())))
#         matrix = df[(df["Date/Time"] >= range1) & (df["Date/Time"] <= range2)] # filters dataset to only have between range1 and range2 values
#     '''
#     print("Matrix column split")
#     # break down matrix into time sections
#     matrix["year"] = matrix["time_bucket"].dt.year
#     matrix["month"] = matrix["time_bucket"].dt.month
#     matrix["day"] = matrix["time_bucket"].dt.day
    
#     matrix["hour"] = matrix["time_bucket"].dt.hour
#     matrix["minute"] = matrix["time_bucket"].dt.minute
#     matrix["second"] = matrix["time_bucket"].dt.second

#     matrix["total_seconds"] = matrix["hour"] * 3600 + matrix["minute"] * 60 + matrix["second"] #converts hour, min, sec all into seconds
#     matrix = matrix.drop(columns=["time_bucket", "hour", "minute", "second"])

#     print(matrix)
#     matrix.to_pickle("torch_process.pkl")

def create_tensor():
    print("Making global tensor")
    main_matrix = bucket(lat, lon, time, 0.1, 0.1, "min") # matrix is now bucketed, containing a count column
    # print(main_matrix)
    
    # --------------method padding------------------
    global time_steps
    time_steps = main_matrix["time"].value_counts().max() # found using the most reoccurances of a value

    result_list = []
    # Group the data by time and iterate efficiently
    for time_value, group in main_matrix.groupby("time"):
        group_size = len(group)
        if group_size < time_steps: # If the group is smaller than the time_steps, pad with zeros
            padding = np.zeros((time_steps - group_size, group.shape[1]))
            padded_group = np.vstack([group.to_numpy(), padding])
        else:
            padded_group = group.to_numpy()
        
        result_list.append(padded_group)

    final_result = np.vstack(result_list) # Stack all the results together into a final numpy array

    padded_matrix = pd.DataFrame(final_result, columns=main_matrix.columns) # convert back into df
    print(padded_matrix)
    
    padded_matrix.to_pickle("torch_process.pkl")
    data_array = padded_matrix.to_numpy()
    print(data_array)
    # --------------------------------------------
    batch_size = data_array.shape[0] // time_steps
    spatial_dim = 2  # lat and long
    channels = 1 

    # [:, :2] is there to only take lat and long columns
    # THIS IS TENSOR CONVERSION, PLEASE CHECK
    # (batch_size, channels=1, time_steps, spatial_dim)
    tensor_input = np.reshape(data_array[:, :2],(batch_size, channels, time_steps, spatial_dim))
    global ftensor
    ftensor = torch.from_numpy(tensor_input)
    print(ftensor)
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
    # return validation_x