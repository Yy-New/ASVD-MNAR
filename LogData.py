import os
import numpy as np
from utils import get_excel_data, merge_heading, is_folder_exists

file_list_path = "../PublicData"
file_list = os.listdir(file_list_path)

for file in file_list:
    file_path = f"{file_list_path}/{file}"
    class_label, metabolites, file_data = get_excel_data(file_path)
    file_data = file_data.values.astype("float")
    log_data = np.where(np.isnan(file_data), np.nan, np.log(file_data + 1))
    scaled_data = merge_heading(class_label, metabolites, log_data)
    is_folder_exists("../LogPublicData")
    scaled_data.to_excel(f"../LogPublicData/{file}", header=None, index=False)

