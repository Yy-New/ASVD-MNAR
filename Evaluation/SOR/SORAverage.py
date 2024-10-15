import os

import numpy as np
import pandas as pd

from utils import is_folder_exists

file_path = "../Results/SOR/ASVD_MNAR"

file_list = os.listdir(file_path)
for file in file_list:
    threshold_concentration_path = f"{file_path}/{file}"
    threshold_concentration_files = os.listdir(threshold_concentration_path)
    for threshold_concentration_file in threshold_concentration_files:
        gamma_prop_files_path = f"{threshold_concentration_path}/{threshold_concentration_file}"
        gamma_prop_files = os.listdir(gamma_prop_files_path)
        for gamma_prop_file in gamma_prop_files:
            num_files_path = f"{gamma_prop_files_path}/{gamma_prop_file}"
            num_files = os.listdir(num_files_path)
            result_list = []
            for num_file in num_files:
                sor_file_path = f"{num_files_path}/{num_file}"
                SOR_result = pd.read_excel(sor_file_path, index_col=0)
                result_list.append(SOR_result)
            average = np.zeros(result_list[0].shape)
            for result in result_list:
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        average[i][j] += result.values[i][j]
            average = average / len(result_list)
            average = pd.DataFrame(average)
            average.index = result_list[0].index
            average.columns = result_list[0].columns
            save_path = f"{gamma_prop_files_path.replace('Results', 'AverageResults')}/"
            is_folder_exists(save_path)
            save_path = f"{save_path}/{file}_{gamma_prop_file}.xlsx"
            average.to_excel(save_path)
            print(f"{num_files_path} processing completed.")
