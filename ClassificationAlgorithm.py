from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


def xgb_construction_classifier(metabolites, x_imposeds, x_miss_types, alpha, beta):
    """
    :param metabolites: Metabolite name
    :param x_imposed: Generated missing data
    :param x_miss_type: Missing type of generated data
    :param alpha: Separation of low concentrations from remaining metabolites (proportion of low concentrations)
    :param beta: Separation of high concentrations from remaining metabolites (medium to low concentration ratio)
    :return: Classifier, verify set accuracy
    """
    metabolites = metabolites.tolist()
    train_data = []
    for x_imposed, x_miss_type in zip(x_imposeds, x_miss_types):
        mean_concentrations = np.nanmean(x_imposed, axis=1)

        sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

        low_abundance_metabolites = sorted_metabolites[:alpha]
        mid_abundance_metabolites = sorted_metabolites[alpha:beta]
        high_abundance_metabolites = sorted_metabolites[beta:]

        for i in range(x_imposed.shape[0]):
            tmp_mis_prop = np.count_nonzero(np.isnan(x_imposed[i])) / len(x_imposed[i])
            for j in range(x_imposed.shape[1]):
                if not np.isnan(x_imposed[i][j]):
                    continue
                tmp_train_data = []
                sample_nan_cnt = 0
                tmp_j = j
                while tmp_j >= 0 and np.isnan(x_imposed[i][tmp_j]):
                    tmp_j -= 1
                    sample_nan_cnt += 1
                tmp_j = j+1
                while tmp_j < x_imposed.shape[1] and np.isnan(x_imposed[i][tmp_j]):
                    tmp_j += 1
                    sample_nan_cnt += 1
                metabolites_nan_cnt = 0
                tmp_i = i
                while tmp_i >= 0 and np.isnan(x_imposed[tmp_i][j]):
                    tmp_i -= 1
                    metabolites_nan_cnt += 1
                tmp_i = i + 1
                while tmp_i < x_imposed.shape[0] and np.isnan(x_imposed[tmp_i][j]):
                    tmp_i += 1
                    metabolites_nan_cnt += 1
                tmp_train_data.append(sample_nan_cnt)
                tmp_train_data.append(metabolites_nan_cnt)
                tmp_train_data.append(np.nanmax(x_imposed[i]))
                tmp_train_data.append(np.nanmin(x_imposed[i]))
                tmp_train_data.append((np.nanmedian(x_imposed[i])))
                tmp_train_data.append((np.nanmean(x_imposed[i])))
                tmp_train_data.append(tmp_mis_prop)
                tmp_train_data.append(metabolites[i])
                if metabolites[i] in low_abundance_metabolites:
                    tmp_train_data.append(1)
                elif metabolites[i] in mid_abundance_metabolites:
                    tmp_train_data.append(2)
                elif metabolites[i] in high_abundance_metabolites:
                    tmp_train_data.append(3)
                tmp_train_data.append(x_miss_type[i][j])
                train_data.append(tmp_train_data)

    train_data = np.array(train_data)
    train_data[train_data == "MNAR"] = 0
    train_data[train_data == "MCAR"] = 1
    train_data = train_data.astype('float')

    X_train, X_test, y_train, y_test = train_test_split(train_data[:, :9], train_data[:, 9], test_size=0.3)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'seed': 42
    }

    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    y_pred = model.predict(dtest)
    y_pred = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


def xgb_prediction_type(metabolites, data, classifier, alpha, beta):
    """
    :param metabolites: Metabolite name
    :param data: The original data
    :param classifier: classifier
    :param alpha: Separation of low concentrations from remaining metabolites (proportion of low concentrations)
    :param beta: Separation of high concentrations from remaining metabolites (medium to low concentration ratio)
    :return: Missing data prediction type
    """
    data = data.values.astype('float')

    mean_concentrations = np.nanmean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    low_abundance_metabolites = sorted_metabolites[:alpha]
    mid_abundance_metabolites = sorted_metabolites[alpha:beta]
    high_abundance_metabolites = sorted_metabolites[beta:]
    x_test = []
    for i in range(data.shape[0]):
        tmp_mis_prop = np.count_nonzero(np.isnan(data[i])) / len(data[i])
        for j in range(data.shape[1]):
            if not np.isnan(data[i][j]):
                continue
            tmp_test_data = []
            metabolites_nan_cnt = 0
            tmp_i = i
            while tmp_i >= 0 and np.isnan(data[tmp_i][j]):
                tmp_i -= 1
                metabolites_nan_cnt += 1
            tmp_i = i + 1
            while tmp_i < data.shape[0] and np.isnan(data[tmp_i][j]):
                tmp_i += 1
                metabolites_nan_cnt += 1
            tmp_test_data.append(metabolites_nan_cnt)
            tmp_test_data.append(np.nanmax(data[i]))
            tmp_test_data.append(np.nanmin(data[i]))
            tmp_test_data.append((np.nanmedian(data[i])))
            tmp_test_data.append((np.nanmean(data[i])))
            tmp_test_data.append(tmp_mis_prop)
            tmp_test_data.append(metabolites[i])
            if metabolites[i] in low_abundance_metabolites:
                tmp_test_data.append(1)
            elif metabolites[i] in mid_abundance_metabolites:
                tmp_test_data.append(2)
            elif metabolites[i] in high_abundance_metabolites:
                tmp_test_data.append(3)
            x_test.append(tmp_test_data)
    x_test = np.array(x_test, dtype='float')
    dtest = xgb.DMatrix(x_test)
    y_pred = classifier.predict(dtest)
    y_pred = [round(value) for value in y_pred]
    y_pred = np.array(np.array(y_pred), dtype='str')
    y_pred[y_pred == '0'] = "MNAR"
    y_pred[y_pred == '1'] = "MCAR"

    target_data_type = deepcopy(data)
    target_data_type = target_data_type.astype("str")
    target_cnt = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i][j]):
                target_data_type[i][j] = y_pred[target_cnt]
                target_cnt += 1
            else:
                target_data_type[i][j] = "O"
    return target_data_type
