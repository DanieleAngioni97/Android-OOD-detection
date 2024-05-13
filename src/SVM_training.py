import torch
from utils import fm
import os
from sklearn.svm import LinearSVC
from models.base.base_torch_svm import LinearSVM
from utils import data as ut_data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack
import numpy as np
from pytorch_ood.loss import EnergyRegularizedLoss, EntropicOpenSetLoss, CrossEntropyLoss
from pytorch_ood.detector import MaxSoftmax, EnergyBased, Entropy, OpenMax
from sklearn.metrics import roc_auc_score
from utils.data import load_dataset
from itertools import chain


if __name__ == '__main__':
    
    path_0 = "data/datasets_and_clf/"
    path_1 = "data/ital-IA_clf_and_datasets/"
    path_2 = "data/extended-features/"
    # train_features = fm.my_load(os.path.join(path_2, "train.pkl"))
    
    """
    test_features = fm.my_load(os.path.join(path_1, "test-1.pkl"))
    test_features_w = fm.my_load(os.path.join(path_0, "tests_monthly.pkl"))
    """

    ds = load_dataset(os.path.join(path_2))
    print(ds.keys())
    #print(ds['time_index'][2016][1])

    vs = [v for k,v in ds['time_index'][2014].items()]
    vs = list(chain.from_iterable(vs))

    # print(ds["X"][vs][0])

    X_train = ds["X"][vs]
    y_train = ds["y"][vs]

    vs_t = [[v for k,v in ds['time_index'][2015].items()],
            [v for k,v in ds['time_index'][2016].items()],
            [v for k,v in ds['time_index'][2017].items()],
            [v for k,v in ds['time_index'][2018].items()]]
    
    X_tests = []
    y_tests = []
    t_tests = []
    for a in vs_t:
        for i in range(12):
            X_tests.append(ds["X"][a[i]])
            y_tests.append(ds["y"][a[i]])
            t_tests.append(ds["T"][a[i]])

    print(len(X_tests))

    max_n_samples = -1
    b_size = 1000

    n_test_months = 4*12
    n_months_per_test = 1
    n_slots = n_test_months // n_months_per_test
    
    test_set_list = []
    # X_tests, y_tests, t_tests = test_features_w
    for i in range(n_slots):
        start = i * n_months_per_test
        end = (i + 1) * n_months_per_test
        test_set_i = {}
        test_set_i['X'] = vstack(X_tests[start:end])
        test_set_i['y'] = np.concatenate(y_tests[start:end])
        test_set_i['t'] = np.concatenate(t_tests[start:end])
        test_set_list.append(test_set_i)

    test_loader_list = []
    for test_set_i in test_set_list:
        sparse_dataset = ut_data.SparseDataset(test_set_i['X'][:max_n_samples],
                                       test_set_i['y'][:max_n_samples])
        test_loader_i = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                                 collate_fn=ut_data.dense_batch_collate)
        test_loader_list.append(test_loader_i)

    # X_train, y_train, t_train = train_features

    sparse_dataset = ut_data.SparseDataset(X_train[:max_n_samples], y_train[:max_n_samples])
    train_loader = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                             collate_fn=ut_data.dense_batch_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Main Training for the SVM
    """
    classifier = LinearSVC(loss='hinge', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_tests[0])

    # Baseline Linear Layer
    linear_b = LinearSVM(n_features=classifier.n_features_in_,
                        bias=classifier.fit_intercept)
    linear_b.linear.weight = torch.nn.Parameter(torch.tensor(classifier.coef_))
    if classifier.fit_intercept:
        linear_b.linear.bias = torch.nn.Parameter(
            torch.tensor(classifier.intercept_))

    # Testing
    scores_torch = np.array([])
    y_pred_torch = np.array([])
    y_ts = np.array([])

    with torch.no_grad():
        for x, y in test_loader_list[0]:
            out = linear_b(x).numpy()
            # print(out)
            scores_torch = np.append(scores_torch, out[:,1])
            y_pred_torch = np.append(y_pred_torch, np.argmax(out, axis=1))
            y_ts = np.append(y_ts, y)

    print("Torch model accuracy", (y_pred_torch == y_ts).mean())
    print("AUROC", roc_auc_score(y_ts, scores_torch))

    # OOD Detection methods applied
    overwrite_detectors = True
    path = path_2
    detectors_path = os.path.join(path, f"detectors.pkl")

    if overwrite_detectors or (not os.path.isfile(detectors_path)):
        # The detector that we want to test from the paper [1] - [6] - [26] (in addition to the MaxSoftMax baseline detector)
        print("Creating OOD Detectors")

        detectors = {}
        detectors["MaxSoftmax"] = MaxSoftmax(linear_b)
        detectors["Entropy"] = Entropy(linear_b)
        detectors["EnergyBased"] = EnergyBased(linear_b)
        detectors["OpenMax"] = OpenMax(linear_b)

        print(f"> Fitting {len(detectors)} detectors")

        for name, detector in detectors.items():
            print(f"--> Fitting {name}")
            detector.fit(train_loader, device=device)
        fm.my_save(detectors, os.path.join(path, f"detectors.pkl"))
    else:
        detectors = fm.my_load(detectors_path)

    import json
    from utils.evaluation import detector_scores

    datasets = {}

    results_list = []
    for i, loader in enumerate(test_loader_list):
      datasets["OOD test set "] = loader
      results = detector_scores(detectors, datasets)
      results_list.append(results)

    # Saving the results in a json file
    fm.my_save(results_list, os.path.join(path, f'test_monthly_results.pkl'))