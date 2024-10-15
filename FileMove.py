import os

threshold_concentration_path = "./MMData/ST000419"
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
            for missing_file in missing_files:
                if "target" in missing_file:
                    source_file = missing_files_path + f"/{missing_file}"
                    os.remove(source_file)

