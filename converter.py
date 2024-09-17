import csv
import glob
import pandas as pd
import os

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