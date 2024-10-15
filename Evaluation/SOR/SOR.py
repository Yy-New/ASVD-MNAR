import os

import numpy as np
import pandas as pd

from utils import get_csv_data, get_excel_data, get_NRMSE, is_folder_exists

RealData_path = "../../LogPublicData"
MMData_path = "../../MMData"
file_list = os.listdir(MMData_path)
for file in file_list:
    real_data = get_excel_data(RealData_path + '/' + file + '.xlsx')[2].values.astype('float')
    threshold_concentration_path = f"{MMData_path}/{file}"
    threshold_concentration_files = os.listdir(threshold_concentration_path)
    for threshold_concentration_file in threshold_concentration_files:
        gamma_prop_files_path = f"{threshold_concentration_path}/{threshold_concentration_file}"
        gamma_prop_files = os.listdir(gamma_prop_files_path)
        for gamma_prop_file in gamma_prop_files:
            num_files_path = f"{gamma_prop_files_path}/{gamma_prop_file}"
            num_files = os.listdir(num_files_path)
            for num_file in num_files:
                missing_files_path = f"{num_files_path}/{num_file}"
                missing_files = os.listdir(missing_files_path)
                CWSVDImputation_path = missing_files_path.replace("MMData", "FillData/ASVDImputation")
                MultipleImputation_path = missing_files_path.replace("MMData", "FillData/MultipleImputation")
                methods = ["rf", "svd", "QRILC", "knn", "ns_knn", "iter_svd", "ASVD-MNAR"]
                rank_list = []
                missing_prop_list = []
                for missing_file in missing_files:
                    if 'target' in missing_file:
                        continue
                    miss_prop = float(missing_file.split(f"{file}_")[1].split('%')[0])
                    if miss_prop not in missing_prop_list:
                        missing_prop_list.append(miss_prop)
                    missing_path = f"{missing_files_path}/{missing_file}"
                    rf_path = f"{MultipleImputation_path}/rf_{missing_file}"
                    iter_svd_path = f"{MultipleImputation_path}/iter_svd_{missing_file}"
                    svd_path = f"{MultipleImputation_path}/svd_{missing_file}"
                    QRILC_path = f"{MultipleImputation_path}/QRILC_{missing_file}"
                    knn_path = f"{MultipleImputation_path}/knn_{missing_file}"
                    ns_knn_path = f"{MultipleImputation_path}/ns_knn_{missing_file}"
                    ASVD_MNAR_path = f"{CWSVDImputation_path}/ASVD-MNAR_{missing_file}"

                    missing_data = get_csv_data(missing_path)[2].values.astype('float')
                    rf_data = get_csv_data(rf_path)[2].values.astype('float')
                    iter_svd_data = get_csv_data(iter_svd_path)[2].values.astype('float')
                    svd_data = get_csv_data(svd_path)[2].values.astype('float')
                    QRILC_data = get_csv_data(QRILC_path)[2].values.astype('float')
                    knn_data = get_csv_data(knn_path)[2].values.astype('float')
                    ns_knn_data = get_csv_data(ns_knn_path)[2].values.astype('float')
                    ASVD_MNAR_data = get_csv_data(ASVD_MNAR_path)[2].values.astype('float')

                    rank = [0, 0, 0, 0, 0, 0, 0]
                    for i in range(missing_data.shape[0]):
                        if np.sum(np.isnan(missing_data[i])) == 0:
                            continue
                        rf_nrmse = get_NRMSE(real_data[i], rf_data[i])
                        iter_svd_nrmse = get_NRMSE(real_data[i], iter_svd_data[i])
                        svd_nrmse = get_NRMSE(real_data[i], svd_data[i])
                        QRILC_nrmse = get_NRMSE(real_data[i], QRILC_data[i])
                        knn_nrmse = get_NRMSE(real_data[i], knn_data[i])
                        ns_knn_nrmse = get_NRMSE(real_data[i], ns_knn_data[i])
                        ASVD_MNAR_nrmse = get_NRMSE(real_data[i], ASVD_MNAR_data[i])
                        rank += np.argsort(np.argsort([rf_nrmse, iter_svd_nrmse, svd_nrmse, QRILC_nrmse, knn_nrmse,
                                                       ns_knn_nrmse, ASVD_MNAR_nrmse])) + 1
                    rank_list.append(rank)
                rank_list = np.array(rank_list).T
                rank_list = pd.DataFrame(rank_list, index=methods, columns=missing_prop_list).sort_index(axis=1)
                save_dir_path = num_files_path.replace("MMData", "Results/SOR/ASVD_MNAR")
                is_folder_exists(save_dir_path)
                save_path = f"{save_dir_path}/{file}_SOR_{num_file}.xlsx"
                rank_list.to_excel(save_path)
                print(f"{missing_files_path} processing completed.")
