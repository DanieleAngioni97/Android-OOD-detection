from typing import Union

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random,
                          coo_matrix,
                          csr_matrix,
                          vstack)
from tqdm import tqdm


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data: Union[np.ndarray, coo_matrix, csr_matrix],
                 targets: Union[np.ndarray, coo_matrix, csr_matrix],
                 transform: bool = None):

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

        self.transform = transform  # Can be removed

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

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
