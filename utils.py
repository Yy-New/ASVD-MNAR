import copy
import os
import random
from math import floor

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_csv_data(data_url):
    """
    :param data_url: csv file address
    :return: Sample label, metabolite name, two-dimensional matrix (row: metabolite, column: sample)
    """
    data = pd.read_csv(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


def get_excel_data(data_url):
    """
    :param data_url: address of the excel file
    :return: Sample label, metabolite name, two-dimensional matrix (row: metabolite, column: sample)
    """
    data = pd.read_excel(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


def is_folder_exists(save_path_dir):
    """
    :param save_path_dir: indicates the folder address
    :return: If the folder does not exist, it is created
    """
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)


def merge_heading(class_label, metabolites, data):
    """
    :param class_label: sample label
    :param metabolites: name of a metabolite
    :param data: indicates the data matrix
    :return: Restore the original data format
    """
    tmp_data = np.insert(data, 0, metabolites.tolist(), axis=1)
    df = pd.DataFrame(tmp_data)
    df = pd.concat([class_label, df], axis=0).reset_index(drop=True)
    return df


def get_NRMSE(x_true, x_pred):
    """
    param x_true: indicates the actual data
    param x_pred: indicates forecast data
    :return: NRMSE of real data and predicted data
    """
    x_true = x_true.astype('float')
    x_pred = x_pred.astype('float')
    mse = mean_squared_error(x_true, x_pred)
    rmse = np.sqrt(mse)
    mean_std = np.mean(x_true.std())
    return rmse / mean_std


def get_MAPE(x_true, x_pred):
    """
    :param x_true: indicates the actual data
    :param x_pred: indicates forecast data
    :return: MAPE of real data and predicted data
    """
    x_true = x_true.astype('float')
    x_pred = x_pred.astype('float')
    absolute_percentage_errors = np.abs((x_pred - x_true) / x_true)
    return np.mean(absolute_percentage_errors)


def nan_permutation(x):
    new_data = []
    max_num = 0
    for item in x:
        nan_data = []
        tmp_data = []
        nan_count = 0
        for val in item:
            if np.isnan(val):
                nan_data.append(val)
                nan_count += 1
            else:
                tmp_data.append(val)
        new_data.append(tmp_data + nan_data)
        max_num = max(max_num, nan_count)
    return new_data, len(x[0]) - max_num


def get_complete_subset_data(data):
    """
    :param data: missing data set
    :return: obtains the maximum data subset
    """
    data = data.values.astype("float")
    data, max_num = nan_permutation(data)
    data = pd.DataFrame(data).iloc[:, :max_num]
    return data


def nan_distance(x, y, p):
    """
    :param x:
    :param y:
    :param p: p is 1 for Manhattan distance p is 2 for Euclidean distance p is 3 for Chebyshev distance
    :return:
    """
    n, m = x.shape
    ans = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            nan_mask = ~np.isnan(x[i]) & ~np.isnan(y[j])
            num_cnt = np.sum(nan_mask)
            if num_cnt > 0:
                diff = np.abs(x[i, nan_mask] - y[j, nan_mask]) ** p
                sum_diff = np.sum(diff)
                ans[i, j] = (m / num_cnt * sum_diff) ** (1 / p)
                ans[j, i] = ans[i, j]
    return ans


def get_dynamic_k(dist):
    n = dist.shape[0]
    temp_dist = np.sort(dist, axis=1)
    dynamic_k = []
    for i in range(n):
        sum = 0
        for j in range(2, n):
            sum += temp_dist[i][j] - temp_dist[i][1]
        sum /= ((n-2)*np.sqrt(len(dist)))
        cnt = 1
        for j in range(2, n):
            if temp_dist[i][j] - temp_dist[i][1] < sum:
                cnt += 1
        dynamic_k.append(cnt)
    return dynamic_k


def class_MM_generate(metabolites, data, mis_prop, alpha, beta, gamma):
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

    return data, data_target


def get_init_miss_data(miss_data, miss_data_type):
    miss_data = miss_data.values
    res = copy.deepcopy(miss_data)
    for i in range(len(miss_data)):
        num = 1000
        min_non_nan = np.nanmin(miss_data[i])
        data_filled = np.where(np.isnan(miss_data[i]), np.nanmin(miss_data[i]), miss_data[i])
        mean_estimation = np.nanmean(data_filled)
        std_estimation = np.nanstd(data_filled)
        norm_dist = np.random.normal(mean_estimation, std_estimation, num)
        min_filtered_list = norm_dist[(norm_dist < min_non_nan) & (0 < norm_dist)].tolist()
        mnar_index = np.where(miss_data_type[i]=="MNAR")[0]
        mcar_index = np.where(miss_data_type[i] == "MCAR")[0]
        if min_non_nan == 0:
            min_filtered_list = [min_non_nan] * len(mnar_index)
        else:
            while len(min_filtered_list) < len(mnar_index):
                num *= 10
                norm_dist = np.random.normal(mean_estimation, std_estimation, num)
                min_filtered_list = norm_dist[(norm_dist < min_non_nan) & (0 < norm_dist)].tolist()
                if num >= 10000000:
                    if len(min_filtered_list) < len(mnar_index):
                        min_filtered_list.extend([min_non_nan]*(len(mnar_index)-len(min_filtered_list)))
                    break
        selected_numbers = random.sample(min_filtered_list, len(mnar_index))
        for j, idx in enumerate(mnar_index):
            res[i][idx] = selected_numbers[j]
        selected_numbers = random.sample(norm_dist.tolist(), len(mcar_index))
        for j, idx in enumerate(mcar_index):
            res[i][idx] = selected_numbers[j]
    return res


def get_min_data(miss_data):
    data = copy.deepcopy(miss_data)
    data.fillna(data.min(), inplace=True)
    return data.values
