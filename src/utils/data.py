from typing import Union

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random,
                          coo_matrix,
                          csr_matrix,
                          vstack)
import os
import json
import datetime
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from os.path import join
from utils.fm import my_load, my_save
from typing import Optional, List, Tuple, Union
from tesseract import temporal
from utils.utils import set_all_seed


DS_ROOT_PATH = '../data/extended-features'

## Loading features
def load_dataset(ds_root_path: str = DS_ROOT_PATH,
                 updated: bool = True,
                 reduced=None):
    """
    The function to load features in the Tesseract dataset. Please note that you have to parametrize the names of the files opened, to load the right file.
    """

    X_fname = 'extended-features-X'
    y_fname = 'extended-features-y'
    meta_fname = 'extended-features-meta'

    dataset_mode = ''

    if reduced is not None:
        updated = True

    if updated:
        dataset_mode += '-updated'
    y_fname = f"{y_fname}{dataset_mode}"
    meta_fname = f"{meta_fname}{dataset_mode}"

    if reduced is not None:
        assert reduced in ('1k', '10k')
        dataset_mode += f'-reduced-{reduced}'
    X_fname = f"{X_fname}{dataset_mode}"

    X_fname = f"{X_fname}.json"
    y_fname = f"{y_fname}.json"
    meta_fname = f"{meta_fname}.json"


    vec_path = join(ds_root_path, f"vec{dataset_mode}.pkl")
    ds_path = join(ds_root_path, f"dataset{dataset_mode}.pkl")

    if not os.path.exists(ds_path):
        print(f'Loading and processing dataset...')
        with open(join(ds_root_path, X_fname), 'r') as f:
            X = json.load(f)

        print('Loading labels...')
        with open(join(ds_root_path, y_fname), 'rt') as f:
            y = json.load(f)

        print('Loading metadata and timestamps...')
        with open(join(ds_root_path, meta_fname), 'rt') as f:
            meta = json.load(f)
        T = [o['dex_date'] for o in meta]
        T = np.array([datetime.datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o
                      else datetime.datetime.strptime(o, '%Y-%m-%d %H:%M:%S') for o in T])

        # Convert to numpy array and get feature names
        vec = DictVectorizer()
        X = vec.fit_transform(X).astype("float32")
        y = np.asarray(y)
        feature_names = vec.get_feature_names_out()

        my_save(vec, vec_path)

        # Get time index of each sample for easy reference
        time_index = {}
        for i in range(len(T)):
            t = T[i]
            if t.year not in time_index:
                time_index[t.year] = {}
            if t.month not in time_index[t.year]:
                time_index[t.year][t.month] = []
            time_index[t.year][t.month].append(i)

        data = (X, y, time_index, feature_names, T, meta)
        data_names = ('X', 'y', 'time_index', 'feature_names', 'T', 'meta')
        data_dict = {k: v for k, v in zip(data_names, data)}
        my_save(data_dict, ds_path)
    else:
        data_dict = my_load(ds_path)



    return data_dict



def load_and_preprocess_dataset(ds_root_path: str = DS_ROOT_PATH,
                                updated=True,
                                reduced=None,
                                train_size=12,
                                test_size=1,
                                use_only_train_features=True,
                                max_train_samples=None):

    data_dict = load_dataset(ds_root_path=ds_root_path,
                             updated=updated,
                             reduced=reduced)
    X, y, time_index, feature_names, T, meta = tuple(data_dict.values())
    splits = temporal.time_aware_train_test_split(
        X, y, T, meta, train_size=train_size, test_size=test_size, granularity='month')
    X_train, X_tests, y_train, y_tests, t_train, t_tests, meta_train, meta_tests = splits


    print(f"### Initial X_train shape: {X_train.shape}")
    if use_only_train_features:
        # remove features that do not appear in the train set
        idx_tr_features = np.array(X_train.sum(axis=0)).flatten() > 0
        X_train = X_train[:, idx_tr_features]
        X_tests = [X_test[:, idx_tr_features] for X_test in X_tests]
        feature_names = np.array(feature_names)[idx_tr_features]
        print(f"### features present only in train set: {idx_tr_features.sum()}")

        # remove samples with all-zero features from the train set
        idx_non_zero_tr_samples = np.array(X_train.sum(axis=1)).flatten() > 0
        X_train = X_train[idx_non_zero_tr_samples, :]
        y_train = y_train[idx_non_zero_tr_samples]
        t_train = t_train[idx_non_zero_tr_samples]
        meta_train = meta_train[idx_non_zero_tr_samples]

        print(f"### number of all-zero samples removed from train set: "
              f"{idx_non_zero_tr_samples.size - idx_non_zero_tr_samples.sum()}")

    print(f"### Number of features not present in train set: {X_train.shape[1]}")

    if isinstance(max_train_samples, int):
        X_train = X_train[:max_train_samples, :]
        y_train = y_train[:max_train_samples]
        t_train = t_train[:max_train_samples]
        meta_train = meta_train[:max_train_samples]

    return X_train, y_train, t_train, meta_train, X_tests, y_tests, t_tests, meta_tests, feature_names


def create_dataloaders(X_train, y_train, t_train, meta_train,
                       X_tests, y_tests, t_tests, meta_tests,
                       train_batch_size=64, test_batch_size=1000,
                       random_seed=0):
    set_all_seed(random_seed)
    sparse_dataset = SparseDataset(X_train, y_train, t_train, meta_train)
    train_loader = DataLoader(sparse_dataset, batch_size=train_batch_size, shuffle=True,
                              collate_fn=dense_batch_collate)

    test_loaders = []
    for X_test, y_test, t_test, meta_test in zip(X_tests, y_tests, t_tests, meta_tests):
        sparse_dataset = SparseDataset(X_test, y_test, t_test, meta_test)
        test_loader_i = DataLoader(sparse_dataset, batch_size=test_batch_size, shuffle=False,
                                   collate_fn=dense_batch_collate)
        test_loaders.append(test_loader_i)

    return train_loader, test_loaders

class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data: Union[np.ndarray, coo_matrix, csr_matrix],
                 targets: Union[np.ndarray, coo_matrix, csr_matrix],
                 timestamps: Optional[Union[np.ndarray, coo_matrix, csr_matrix]],
                 metadata: Optional[Union[np.ndarray, coo_matrix, csr_matrix]],
                 transform: bool = None,
                 get_meta: bool = False):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

        self.timestamps = timestamps
        self.metadata = metadata
        self.get_meta = get_meta
        self.transform = transform  # Can be removed

    def __getitem__(self, index: int):
        if not self.get_meta:
            return self.data[index], self.targets[index]
        else:
            return self.data[index], self.targets[index], self.timestamps[index], self.metadata[index]

    def __len__(self):
        return self.data.shape[0]


def sparse_coo_to_tensor(coo: coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    s = torch.Size(shape)

    return torch.sparse_coo_tensor(i, v, s)


def sparse_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.DoubleTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.DoubleTensor(targets_batch)
    return data_batch, targets_batch


def dense_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    data_batch = torch.tensor(vstack(data_batch).todense(), dtype=torch.double)
    targets_batch = torch.tensor(targets_batch, dtype=torch.int8)
    return data_batch, targets_batch

def OOD_arr(arr, perc):
    # Set the seed for reproducibility
    np.random.seed(132)

    # Calculate the number of elements to replace with -1 (20% of total elements)
    num_elements_to_replace = int(perc * len(arr))

    # Generate random indices to replace
    indices_to_replace = np.random.choice(len(arr), size=num_elements_to_replace, replace=False)

    # Replace elements with -1 at random indices
    arr[indices_to_replace] = -1

    print(arr)
    return arr


def reject_per(l, s, ood_s, per,pred_l):

  '''

  ood_s = [v1, v2, v3, v4, v5, .. ,v10]
  assume v2, v5 are the values with the highest OOD scores
  highest_indices = [1, 4]

  we need min of ood_s[highest_indices] = min([ood_s[i]for i in highest_indices])

  min([ood_s[i]for i in highest_indices])

  '''
  num_samples_to_reject = int(len(ood_s) * per)

  # Sort the OOD scores and get the indices of the highest scores
  sorted_indices = sorted(range(len(ood_s)), key=lambda i: ood_s[i], reverse=True)
  highest_indices = sorted_indices[:num_samples_to_reject]  # Select the indices of the two highest OOD scores

  # Reject the samples with the highest OOD scores

  rej_scores = [s[i] for i in range(len(s)) if i in highest_indices]
  rej_labels = [l[i] for i in range(len(l)) if i in highest_indices]

  n_miss_total = len([l[i] for i in range(len(l)) if i in highest_indices and pred_l[i] != l[i]])
  n_miss_malware = len([l[i] for i in range(len(l)) if i in highest_indices and pred_l[i] != l[i] and l[i] == 1])



  remaining_scores = [s[i] for i in range(len(s)) if i not in highest_indices]
  remaining_labels = [l[i] for i in range(len(l)) if i not in highest_indices]

  # print("Remaining scores:", remaining_scores)
  # print("Remaining labels:", remaining_labels)

  return remaining_labels, remaining_scores, rej_labels, rej_scores, min([ood_s[i]for i in highest_indices]) ,n_miss_total , n_miss_malware

def reject_thr(l, s, ood_s, thr):

  # Reject the samples with the highest OOD scores

  rej_scores = s[ood_s >= thr]
  rej_labels = l[ood_s >= thr]

  remaining_scores = s[ood_s < thr]
  remaining_labels = l[ood_s < thr]

  # print("Remaining scores:", remaining_scores)
  # print("Remaining labels:", remaining_labels)

  return remaining_labels, remaining_scores, rej_labels, rej_scores


if __name__ == '__main__':
    data_dict = load_dataset()
    X, y, time_index, feature_names, T = tuple(data_dict.values())
    print("")