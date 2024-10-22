import numpy as np
import pandas as pd
import os

input_dir="../Data/input/train/dicom_files/"
files = [".".join(file.split(".")[:-1]) for file in os.listdir(input_dir) if file.endswith('.dcm')]

print(files[:5])
# train_data=pd.read_csv("../Data/input/train/train-rle.csv")
# image_ids_list = train_data['ImageId'].tolist()
train_data=pd.read_csv("../Data-Output/metadata.csv")
image_ids_list = train_data['DICOM'].tolist()
missing_files = [file for file in files if file not in image_ids_list]
print(len(missing_files))

with open("../Data-Output/missing_files2.txt", "w") as file:
    for missing_file in missing_files:
        file.write(missing_file + "\n")
