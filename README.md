## Please use "model-modifications" branch for GAN model modifications
# Setup
1. Load the preset conda environment (check below on instructions)
2. `git clone https://github.com/dankmrpanda/USDOT-UTC.git`
3. run converter.py in VSC, make sure to select your interpreter as the preset conda environment

## Preset Conda Environment
1. download myenv.yml
2. `conda create --name <myenv>`
3. `conda activate myenv`
4. `conda env update -f myenv.yml --prune`

## Export Conda Environment
1. `conda activate myenv`
2. `conda env export > myenv.yml`

# File Directory
working/: this model works, **HOWEVER**, is not based on provided code, so might not function properly

setup.py: run after setup is done

main.py: runs gan model, based on provided code

matrix.py: tensor generator

jupyter files/: not maintained

# File Structure
![image](https://github.com/user-attachments/assets/b1286b27-0cd0-49ab-b9f7-f5f51e36b5ab)
## archive/
- contains extracted csvs from: https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city

looks like:

![image](https://github.com/user-attachments/assets/f7690b2d-acbd-4b10-9f1c-18ce7b1ffc7b)

## release/
- contains extracted data from: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

looks like (the 10,357 files should be all txts numbered):

![image](https://github.com/user-attachments/assets/d2c2db35-6b3d-47ec-b090-7ae4a6ef954e)

## pkl/
- empty folder


# IGNORE
```
%windir%\System32\cmd.exe "/K" C:\ProgramData\anaconda3\Scripts\activate.bat C:\ProgramData\anaconda3  && conda activate tf && cd C:\Users\raymo\Desktop\ucr\USDOT UTC && jupyter lab
```
Empty txts here
```
Error reading release/taxi_log_2008_by_id\10115.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\10352.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\1089.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\1497.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\1947.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\2929.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\2945.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\295.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\3050.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\3160.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\3194.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\3950.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\5972.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\6030.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\6236.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\6322.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\6717.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\7583.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\8209.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\8424.txt: No columns to parse from file
Error reading release/taxi_log_2008_by_id\9874.txt: No columns to parse from file
```
