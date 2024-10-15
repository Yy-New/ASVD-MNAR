import pandas as pd
import os
from utils import nan_distance
os.environ['R_HOME'] = "D:/PROGRA~1/R/R-44~1.1"
import copy
import numpy as np
from fancyimpute import SoftImpute, IterativeSVD
from missingpy import MissForest
from sklearn.impute import KNNImputer
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri


def zero_imputation(miss_data):
    data = copy.deepcopy(miss_data)
    data.fillna(0, inplace=True)
    return data.values


def hm_imputation(miss_data):
    data = copy.deepcopy(miss_data)
    data.fillna(data.min()/2, inplace=True)
    return data.values


def mean_imputation(miss_data):
    data = copy.deepcopy(miss_data)
    data.fillna(data.mean(), inplace=True)
    return data.values


def median_imputation(miss_data):
    data = copy.deepcopy(miss_data)
    data.fillna(data.median(), inplace=True)
    return data.values


def knn_imputation(miss_data, k):
    knn_impute = KNNImputer(n_neighbors=k)
    imputed_data = knn_impute.fit_transform(miss_data)
    return imputed_data


def random_forest_imputation(miss_data):
    impute = MissForest(random_state=0, n_jobs=-1)
    imputed_data = impute.fit_transform(miss_data)
    return imputed_data


def soft_svd_imputation(miss_data):
    """
    The matrix is filled by iterative soft threshold processing of SVD decomposition
    """
    miss_data = miss_data.values
    imputed_data = SoftImpute().fit_transform(miss_data)
    return imputed_data


def iterative_svd_imputation(miss_data):
    """
    The matrix is populated by iterative low-rank SVD decomposition
    """
    miss_data = miss_data.values
    imputed_data = IterativeSVD().fit_transform(miss_data)
    return imputed_data


def replace_min(x):
    """
    :param x:list type
    :return: array replaces each missing value in x with the minimum value for that column
    """
    x = np.array(x)
    min_values = np.nanmin(x, axis=0)
    nan_mask = np.isnan(x)
    x[nan_mask] = np.tile(min_values, (x.shape[0], 1))[nan_mask]
    return x


def NS_KNN(x, dist, k):
    """
    :param x: missing data
    :param dist: distance
    :param K: indicates the value of K in KNN
    :return: Returns the result of NS_KNN filling
    """
    temp_x = copy.deepcopy(x)
    temp_x = replace_min(temp_x)
    miss_x = copy.deepcopy(x)
    for i in range(len(x)):
        nan_j = np.isnan(x[i])
        if not nan_j.any():
            continue
        sorted_indices = np.argsort(dist[i])
        top_k_indices = sorted_indices[1: k+1]
        for j in np.where(nan_j)[0]:
            miss_x[i][j] = np.average(temp_x[top_k_indices, j])
    return miss_x


def ns_knn_imputation(miss_data, k, dist_choice):
    miss_data = miss_data.values.astype(float)
    nan_euclidean_distance = nan_distance(miss_data, miss_data, dist_choice)
    imputed_data = NS_KNN(miss_data, nan_euclidean_distance, k)
    imputed_data = pd.DataFrame(imputed_data)
    return imputed_data


def QRILC_imputation(miss_data):
    """
    QRILC interpolates the behavior characteristics listed in the samples This is assuming that each sample is conforming to a normal distribution
    nFeatures = dim(dataSet.mvs)[1]
    nSamples = dim(dataSet.mvs)[2]
    """
    # Enable automatic conversion between R objects and Pandas DataFrame
    pandas2ri.activate()
    # Enable automatic conversion of numpy array to R array
    numpy2ri.activate()
    # Import R's imputeLCMD package
    imputeLCMD = importr('imputeLCMD')
    miss_data = miss_data.values
    imputed_data = imputeLCMD.impute_QRILC(miss_data, tune_sigma=1)[0]
    return imputed_data