import os

import pandas as pd

from utils import get_csv_data, get_excel_data, get_NRMSE, is_folder_exists

RealData_path = "../../LogPublicData"
Imputation_list = ["MultipleImputation", "ASVDImputation"]
# Imputation_list = ["ASVDImputation"]
for imputation_type in Imputation_list:
    FillData_path = f"../../FillData/{imputation_type}"
    if not os.path.exists(FillData_path):
        continue
    file_list = os.listdir(FillData_path)
    for file in file_list:
        real_data = get_excel_data(RealData_path + '/' + file + '.xlsx')[2]
        threshold_concentration_path = f"{FillData_path}/{file}"
        threshold_concentration_files = os.listdir(threshold_concentration_path)
        for threshold_concentration_file in threshold_concentration_files:
            gamma_prop_files_path = f"{threshold_concentration_path}/{threshold_concentration_file}"
            gamma_prop_files = os.listdir(gamma_prop_files_path)
            for gamma_prop_file in gamma_prop_files:
                num_files_path = f"{gamma_prop_files_path}/{gamma_prop_file}"
                num_files = os.listdir(num_files_path)
                for num_file in num_files:
                    fill_files_path = f"{num_files_path}/{num_file}"
                    fill_files = os.listdir(fill_files_path)
                    NRMSE_prop_list = []
                    NRMSE_list = []
                    methods = []
                    temp_NRMSE_list = []
                    save_dir_path = num_files_path.replace("FillData", "Results/NRMSE")
                    is_folder_exists(save_dir_path)
                    save_path = f"{save_dir_path}/{file}_NRMSE_{num_file}.xlsx"
                    if os.path.exists(save_path):
                        now_NRMSE = pd.read_excel(save_path, index_col=0)
                    for fill_file in fill_files:
                        method_name = fill_file.split(f"_{file}_")[0]
                        miss_prop = float(fill_file.split(f"_{file}_")[1].split('%')[0])
                        if miss_prop not in NRMSE_prop_list:
                            NRMSE_prop_list.append(miss_prop)
                        fill_file_path = f"{fill_files_path}/{fill_file}"
                        fill_data = get_csv_data(fill_file_path)[2]
                        NRMSE_result = get_NRMSE(real_data, fill_data)
                        if method_name not in methods:
                            if len(temp_NRMSE_list) != 0:
                                NRMSE_list.append(temp_NRMSE_list)
                            methods.append(method_name)
                            temp_NRMSE_list = []
                        temp_NRMSE_list.append(NRMSE_result)
                    NRMSE_list.append(temp_NRMSE_list)
                    NRMSE_list = pd.DataFrame(NRMSE_list, index=methods, columns=NRMSE_prop_list).sort_index(axis=1)
                    if os.path.exists(save_path):
                        NRMSE_list = pd.concat([now_NRMSE, NRMSE_list], axis=0)
                    NRMSE_list.to_excel(save_path)
                    print(f"{fill_files_path} processing completed.")
