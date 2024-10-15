import os
import random
from math import floor

import numpy as np
import pandas as pd
from utils import is_folder_exists, get_excel_data


def MM_generate(metabolites, class_label, data, mis_prop, k, file, alpha, beta, gamma):

    total_num = data.size
    low_mnar_percentage = gamma * mis_prop
    mid_mnar_percentage = 0.5 * gamma * mis_prop

    low_abundance_missing = round(low_mnar_percentage * total_num)
    mid_abundance_missing = round(mid_mnar_percentage * total_num)
    mv_num = round(mis_prop * total_num) - low_abundance_missing - mid_abundance_missing

    data = data.values.astype(float)
    mean_concentrations = np.mean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    low_abundance_num = round(alpha * data.shape[0])
    mid_abundance_num = round(beta * data.shape[0])
    low_abundance_metabolites = sorted_metabolites[:low_abundance_num]
    mid_abundance_metabolites = sorted_metabolites[low_abundance_num:mid_abundance_num]

    data_target = np.full(data.shape, "O", dtype='object')

    tmp_num = low_abundance_missing % len(low_abundance_metabolites)
    for metabolite in low_abundance_metabolites:
        num_missing = floor(low_abundance_missing / len(low_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        low_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:round(num_missing*0.8)]
        all_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][low_indices_missing] = np.nan
        data_target[kk][low_indices_missing] = "MNAR"
        tmp = np.setdiff1d(all_indices_missing, low_indices_missing)
        if floor(num_missing*0.2) > 0:
            tmp = list(tmp)
            low_indices_missing = random.sample(tmp, floor(num_missing*0.2))
            data[kk][low_indices_missing] = np.nan
            data_target[kk][low_indices_missing] = "MNAR"

    tmp_num = mid_abundance_missing % len(mid_abundance_metabolites)
    for metabolite in mid_abundance_metabolites:
        num_missing = floor(mid_abundance_missing / len(mid_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        mid_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:round(num_missing * 0.8)]
        all_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][mid_indices_missing] = np.nan
        data_target[kk][mid_indices_missing] = "MNAR"
        tmp = np.setdiff1d(all_indices_missing, mid_indices_missing)
        if floor(num_missing * 0.2) > 0:
            tmp=list(tmp)
            mid_indices_missing = random.sample(tmp, floor(num_missing * 0.2))
            data[kk][mid_indices_missing] = np.nan
            data_target[kk][mid_indices_missing] = "MNAR"

    non_nan_coordinates = np.column_stack(np.where(~np.isnan(data)))
    non_nan_flat = [item[0]*data.shape[1]+item[1] for item in non_nan_coordinates]
    mcar_missing = random.sample(non_nan_flat, mv_num)
    data.flat[mcar_missing] = np.nan
    data_target.flat[mcar_missing] = "MCAR"

    data = np.insert(data, 0, metabolites, axis=1)
    df = pd.DataFrame(data)
    df = pd.concat([class_label, df], axis=0).reset_index(drop=True)

    data_target = np.insert(data_target, 0, metabolites, axis=1)
    target_df = pd.DataFrame(data_target)
    target_df = pd.concat([class_label, target_df], axis=0).reset_index(drop=True)

    save_path_dir = f"./MMData/{file}/alpha={round(alpha*100, 2)}%_beta={round(beta*100, 2)}%/gamma={round(gamma*100, 2)}/{k}/"
    is_folder_exists(save_path_dir)
    df_path = save_path_dir + f"{file}_{round(prop * 100, 2)}%.csv"
    df.to_csv(df_path, index=False, header=None)
    # target_df.to_csv(save_path_dir + f"{file}_{round(prop * 100, 2)}%_target.csv", index=False, header=None)
    print(f"{df_path} processing completed." )


if __name__ == "__main__":
    dir_path = "./LogPublicData"
    files = os.listdir(dir_path)
    """
    alpha: Separation of low concentrations from the rest of the metabolites (proportion of low concentrations)
    beta: Separate high concentrations from the rest of the metabolites (medium to low concentration ratio)
    gamma: The loss rate of low concentration as a percentage of the total loss rate
    gamma ranges from 33% to 66%, and the step size is 3%, while the proportion of MNAR ranges from 49.5% to 99%
    """
    threshold_concentration = [[0.1,0.2,0.7], [0.2, 0.3, 0.5], [0.1,0.6,0.3], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.25, 0.45, 0.3]]
    gamma_list = [0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.63, 0.66]

    for item in files:
        file_name = item.split('.')[0]
        class_label, metabolites, data = get_excel_data(dir_path + '/' + item)
        for i in range(len(threshold_concentration)):
            for j in range(len(gamma_list)):
                for k in range(1, 11):
                    for prop in range(50, 401, 25):
                        prop = prop / 1000
                        alpha = threshold_concentration[i][2]
                        beta = threshold_concentration[i][1] + threshold_concentration[i][2]
                        MM_generate(metabolites, class_label, data, prop, k, file_name, alpha, beta, gamma_list[j])
