import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_tabnet
import pickle
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

def get_files(N, p, d):
    file = os.path.join(os.getcwd(), "Syn_data/Datasets/N=" + str(N) + "_p=" + str(p) + "_d_random=" + str(d) + "/data.pkl")
    with open(file, "rb") as input_file:
        data_file = pickle.load(input_file)
    file = os.path.join(os.getcwd(), "Syn_data/Datasets/N=" + str(N) + "_p=" + str(p) + "_d_random=" + str(d) + "/dag_info.pkl")
    with open(file, "rb") as input_file:
        info_file = pickle.load(input_file)
    return data_file, info_file

def reorder_data(data, info):
    relevant_nodes = info['relevant_nodes']
    irrelevant_nodes = info['irrelevant_nodes']
    random_nodes = info['random_nodes']
    target_num = info['target']
    max_col = len(data.columns) - 1
    col_list = list(data)
    col_list[target_num], col_list[max_col] = col_list[max_col], col_list[target_num]
    data = data[col_list]
    if max_col in irrelevant_nodes:
        irrelevant_nodes.remove(max_col)
    else:
        if target_num in relevant_nodes:
            relevant_nodes.remove(target_num)
        elif target_num in random_nodes:
            random_nodes.remove(target_num)
        else:
            irrelevant_nodes.remove(target_num)
        relevant_nodes = [target_num if num == max_col else num for num in relevant_nodes]
        random_nodes = [target_num if num == max_col else num for num in random_nodes]

    return data, relevant_nodes, irrelevant_nodes, random_nodes

def preprocess(data):
    train = data
    target = 'T'
    if "Set" not in train.columns:
        # Ensure that both valid and test sets have at least two elements and they do not contain all the same values
        while True:
            train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))
            train_indices = train[train.Set=="train"].index
            valid_indices = train[train.Set=="valid"].index
            test_indices = train[train.Set=="test"].index
            
            y_train = train[target].values[train_indices]
            y_valid = train[target].values[valid_indices]
            y_test = train[target].values[test_indices]

            if (len(y_valid) > 3) and (len(y_test) > 3):
                if (not all(c == y_valid[0] for c in y_valid)) and (not all(c == y_test[0] for c in y_test)):
                    break

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            # print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    unused_feat = ['Set']
    features = [ col for col in train.columns if col not in unused_feat+[target]] 
    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    X_train = train[features].values[train_indices]
    X_valid = train[features].values[valid_indices]
    X_test = train[features].values[test_indices]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, cat_idxs, cat_dims

def generate_classifier(cat_idxs, cat_dims, lr = 0.01, optimizer = torch.optim.SGD, grouped_features = None):
    tabnet_params = {"cat_idxs": cat_idxs,
                     "cat_dims": cat_dims,
                     "cat_emb_dim": 2,
                     "optimizer_fn":optimizer,
                     "optimizer_params":dict(lr=lr),
                     "scheduler_params": {"step_size":50,
                                          "gamma":0.9},
                     "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                     "mask_type":'entmax', # "sparsemax"
                    #  "grouped_features" : grouped_features,
                }

    return TabNetClassifier(**tabnet_params)

def train(clf, X_train, y_train, X_valid, y_valid, batch_size = 32, virtual_batch_size = 8, max_epochs = 300):
    aug = ClassificationSMOTE(p=0.2)
    sparse_X_train = scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
    sparse_X_valid = scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid

    # Fitting the model
    clf.fit(
        X_train=sparse_X_train, y_train=y_train,
        eval_set=[(sparse_X_train, y_train), (sparse_X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs , patience=max_epochs,
        batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        num_workers=0,
        weights=1,
        drop_last=False,
        augmentations=aug,
    )

    return clf

def compute_importance(clf, relevant_nodes, irrelevant_nodes, random_nodes):
    feature_importances = clf.feature_importances_
    relevant = feature_importances[relevant_nodes]
    irrelevant = feature_importances[irrelevant_nodes]
    random = feature_importances[random_nodes]

    max_length = max(len(relevant), len(irrelevant), len(random))
    relevant_padded = np.pad(relevant, (0, max_length - len(relevant)), mode='constant', constant_values=np.nan)
    irrelevant_padded = np.pad(irrelevant, (0, max_length - len(irrelevant)), mode='constant', constant_values=np.nan)
    random_padded = np.pad(random, (0, max_length - len(random)), mode='constant', constant_values=np.nan)
    data = pd.DataFrame({'Relevant': relevant_padded,
                                'Irrelevant': irrelevant_padded,
                                'Random': random_padded})
    return feature_importances, relevant, irrelevant, random, sns.violinplot(data=data)

def compute_auc(clf, X_train, y_train, X_valid, y_valid, X_test, y_test):
    preds_train = clf.predict_proba(X_train)
    train_auc = roc_auc_score(y_score=preds_train[:,1], y_true=y_train)

    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

    preds_test = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds_test[:,1], y_true=y_test)

    return train_auc, valid_auc, test_auc

def false_rates(feature_importances, relevant, irrelevant, random):
    median = np.median(feature_importances)
    false_negative = 0
    false_positive_i = 0
    false_positive_r = 0
    for i in relevant:
        if i < median:
            false_negative += 1
    for i in irrelevant:
        if i > median:
            false_positive_i += 1
    for i in random:
        if i > median:
            false_positive_r += 1

    fn_relevant = false_negative / len(relevant)
    fp_irrelevant = false_positive_i / len(irrelevant)
    fp_random = false_positive_r / len(random)

    return fn_relevant, fp_irrelevant, fp_random