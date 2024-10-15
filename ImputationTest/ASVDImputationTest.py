import os

from ASVD import ClassificationAdaptiveSVDSolver
from ClassificationAlgorithm import xgb_construction_classifier, xgb_prediction_type
from PSOAlgorithm import pso_xyz_gradient
from utils import is_folder_exists, get_csv_data, get_complete_subset_data, class_MM_generate, get_init_miss_data, \
    get_min_data, merge_heading

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
                result_path = missing_files_path.replace("MMData", "FillData/ASVDImputation")
                is_folder_exists(result_path)
                for missing_file in missing_files:
                    prop = float(missing_file.split('_')[1].split('%')[0]) / 100
                    missing_file_path = missing_files_path + f"/{missing_file}"

                    class_label, metabolites, missing_data = get_csv_data(missing_file_path)
                    missing_data = missing_data.astype('float')

                    x_complete = get_complete_subset_data(missing_data)
                    alpha, beta, gamma = pso_xyz_gradient(metabolites, x_complete, missing_data, prop, 10)

                    index_i = round(len(x_complete) * alpha)
                    index_j = round(len(x_complete) * beta)
                    x_imposeds = []
                    x_miss_types = []
                    for i in range(5):
                        x_imposed, x_miss_type = class_MM_generate(metabolites, x_complete, prop, alpha, beta, gamma)
                        x_imposeds.append(x_imposed)
                        x_miss_types.append(x_miss_type)

                    classifier, train_accuracy = \
                        xgb_construction_classifier(metabolites, x_imposeds, x_miss_types, index_i, index_j)

                    predict_target_type = \
                        xgb_prediction_type(metabolites, missing_data, classifier, index_i, index_j)

                    init_miss_data = get_init_miss_data(missing_data, predict_target_type).T

                    min_data = get_min_data(missing_data.T)

                    missing_data = missing_data.T
                    predict_target_type = predict_target_type.T

                    _data = ClassificationAdaptiveSVDSolver().\
                        fit_transform(missing_data, init_miss_data, predict_target_type, min_data).T

                    tmp_data = merge_heading(class_label, metabolites, _data)
                    tmp_data_path = result_path + f'/ASVD-MNAR_' + missing_file
                    tmp_data.to_csv(tmp_data_path, index=False, header=None)
                    print(f"{missing_file_path} ASVD-MNAR completion of imputation.")

                    svd_lower_limit_list = [0, 0.5, 0.75, 1, 1.25]
                    svd_upper_limit_list = [0.25, 0.75, 1, 1.25, 1.5]
                    index = 1
                    for i in range(5):
                        _data = ClassificationAdaptiveSVDSolver(svd_lower_limit=svd_lower_limit_list[i],
                                                              svd_upper_limit=svd_upper_limit_list[i]). \
                            fit_transform(missing_data, init_miss_data, predict_target_type, min_data).T

                        tmp_data = merge_heading(class_label, metabolites, _data)
                        tmp_data_path = result_path + f'/ASVD-MNAR_{index + i}_' + missing_file
                        tmp_data.to_csv(tmp_data_path, index=False, header=None)
                        print(f"{missing_file_path} ASVD-MNAR_{index + i} completion of imputation.")

                    index += 5
                    soft_threshold = [0.8, 0.85, 0.9, 0.95, 0.99]
                    for i in range(5):
                        _data = ClassificationAdaptiveSVDSolver(soft_threshold_ratio=soft_threshold[i]). \
                            fit_transform(missing_data, init_miss_data, predict_target_type, min_data).T

                        tmp_data = merge_heading(class_label, metabolites, _data)
                        tmp_data_path = result_path + f'/ASVD-MNAR_{index + i}_' + missing_file
                        tmp_data.to_csv(tmp_data_path, index=False, header=None)
                        print(f"{missing_file_path} ASVD-MNAR_{index + i} completion of imputation.")

                    index += 5
                    mnar_lower_limit_list = [0, 0.4, 0.6, 0.8, 0.5, 1]
                    mnar_upper_limit_list = [0.2, 0.6, 0.8, 1, 0.5, 1]
                    for i in range(6):
                        _data = ClassificationAdaptiveSVDSolver(mnar_lower_limit=mnar_lower_limit_list[i],
                                                              mnar_upper_limit=mnar_upper_limit_list[i]). \
                            fit_transform(missing_data, init_miss_data, predict_target_type, min_data).T

                        tmp_data = merge_heading(class_label, metabolites, _data)
                        tmp_data_path = result_path + f'/ASVD-MNAR_{index + i}_' + missing_file
                        tmp_data.to_csv(tmp_data_path, index=False, header=None)
                        print(f"{missing_file_path} ASVD-MNAR_{index + i} completion of imputation.")
