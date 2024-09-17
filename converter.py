# FOLLOWING IS REQUIRED TO USE KAGGLE API
import os
kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

if not os.path.exists(kaggle_dir): #checks if .kaggle folder exists
    os.makedirs(kaggle_dir)
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
if not os.path.exists(kaggle_json_path):
    while True:
    # Get user input for Kaggle username and API key
        kaggle_username = input("Enter your Kaggle username: ")
        kaggle_key = input("Enter your Kaggle API key or type 1 for help: ")
        if kaggle_key == "1":
            print(
                "Please download your Kaggle API key.\n"
                "1. Visit https://www.kaggle.com/account\n"
                "2. Go to the 'API' section and click on 'Create New API Token'.\n"
                "3. This will download a file named 'kaggle.json'.\n"
                "4. Open the file in a reader and copy the value for \"key\""
            )
        else: break

    # Create the kaggle.json file based on given username and API key
    with open(kaggle_json_path, 'w') as f:
        f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')
    f.close()
        
    # file perms to owner read/write only
    os.chmod(kaggle_json_path, 0o600)
# ----------------------------------------------------
# DOWNLOAD DATASET

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# Authenticate using the Kaggle API
api = KaggleApi()
api.authenticate()
dataset_url = "mukaffimoin/potato-diseases-datasets"
path = "./kaggle test"

try:
    print("Please wait, this will take a while depending on dataset size")
    kaggle.api.dataset_download_files(dataset_url, path=path, unzip=True)
    print(f'Dataset downloaded to: {os.path.abspath(path)}')
except Exception as e:
    print(f'An error occurred: {e}')

import csv
import glob
import pandas as pd

def tdrive_conv():
    directory = "release/taxi_log_2008_by_id"
    column_names = ["taxi id", "date time", "longitude", "latitude"]
    if not os.path.isfile("release/taxi_log_2008_by_id.csv"):
        with open("release/taxi_log_2008_by_id.csv", "w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=column_names)
            writer.writeheader()
            for path in glob.glob(f"{directory}/*.txt"):
                with open(path, newline="") as source:
                    reader = csv.DictReader(source, delimiter=",", fieldnames=column_names)
                    writer.writerows(reader)
    else:
        print("release/taxi_log_2008_by_id.csv exists")

    if not os.path.isfile("release/taxi_log_2008_by_id.pkl"):
        df = pd.read_csv("release/taxi_log_2008_by_id.csv", encoding="latin1")
        pkl_file = "taxi_log_2008_by_id.pkl"
        print(f"Converting to {pkl_file}")
        df.to_pickle("release/" + pkl_file)
    else:
        print("release/taxi_log_2008_by_id.pkl exists")

def uber_conv():
    directory = "archive"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith(".csv") and not os.path.isfile(f"pkl/{filename}.pkl"):
            print(f"Reading {f}")
            try:
                df = pd.read_csv(f, encoding="latin1")
                pkl_file = f"pkl/{filename}.pkl"
                print(f"Converting {f} to {pkl_file}")
                df.to_pickle(pkl_file)
                # print(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        else:
            print(f"{f} exists")



def uber_combine():
    if os.path.isfile("archive/uber-raw-data-combined.csv"):
        print("archive/uber-raw-data-combined.csv exists")
        return
    months = ["apr", "may", "jun", "jul", "aug", "sep"]
    
    output = "archive/uber-raw-data-combined.csv"
    
    first = True
    
    for file in months:
        file = f"archive/uber-raw-data-{file}14.csv"
        print(f"combining {file}")
        for chunk in pd.read_csv(file, chunksize=10000):
            if first:
                chunk.to_csv(output, index=False, mode="w", header=True)
                first = False
            else:
                chunk.to_csv(output, index=False, mode="a", header=False)
    print("CSV files have been combined")
    
# print("Converting T-drive dataset: txt -> csv -> pkl")
# tdrive_conv()

print("Combining all uber-raw-data csvs together")
uber_combine()

print("Converting Uber dataset: csv -> pkl")
uber_conv()