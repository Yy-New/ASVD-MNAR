import os

from ImputationAlgorithm import random_forest_imputation, soft_svd_imputation, iterative_svd_imputation, \
    knn_imputation, ns_knn_imputation, QRILC_imputation
from utils import get_csv_data, is_folder_exists, merge_heading


files_path = "../MMData"
files = os.listdir(files_path)
for file in files:
    threshold_concentration_path = files_path + f"/{file}"
    threshold_concentration_files = os.listdir(threshold_concentration_path)
    for threshold_concentration_file in threshold_concentration_files:
        gamma_prop_files_path = threshold_concentration_path + f"/{threshold_concentration_file}"
        gamma_prop_files = os.listdir(gamma_prop_files_path)
        for gamma_prop_file in gamma_prop_files:
            num_files_path = gamma_prop_files_path + f"/{gamma_prop_file}"
            num_files = os.listdir(num_files_path)
            for num_file in num_files:
                missing_files_path = num_files_path + f"/{num_file}"
                missing_files = os.listdir(missing_files_path)
                result_path = missing_files_path.replace("MMData", "FillData/MultipleImputation")
                is_folder_exists(result_path)
                for missing_file in missing_files:
                    prop = float(missing_file.split('_')[1].split('%')[0]) / 100
                    missing_file_path = missing_files_path + f"/{missing_file}"
                    class_label, metabolites, missing_data = get_csv_data(missing_file_path)
                    missing_data = missing_data.astype('float').T

                    _data = knn_imputation(missing_data, 6).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/knn_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} KNN completion of imputation.")

                    _data = soft_svd_imputation(missing_data).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/svd_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} svd completion of imputation.")

                    _data = iterative_svd_imputation(missing_data).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/iter_svd_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} iter_svd completion of imputation.")

                    _data = random_forest_imputation(missing_data).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/rf_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} rf completion of imputation.")

                    _data = ns_knn_imputation(missing_data, k=6, dist_choice=2).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/ns_knn_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} ns_knn completion of imputation.")

                    _data = QRILC_imputation(missing_data).T
                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + '/QRILC_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} QRILC completion of imputation.")

