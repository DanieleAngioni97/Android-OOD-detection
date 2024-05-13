from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from plots import plot_roc, get_metrics
import gdown
import os
from zipfile import ZipFile, ZIP_DEFLATED
import json
import pandas as pd
import numpy as np


def load_features(features_path):
    """

    Parameters
    ----------
    features_path :
        Absolute path of the features compressed file.

    Returns
    -------
    generator of list of strings
        Iteratively returns the textual feature vector of each sample.
    """
    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        for filename in z.namelist():
            with z.open(filename) as fp:
                js = json.load(fp)
                yield [f"{k}::{v}" for k in js for v in js[k] if js[k]]


def load_labels(features_path, ds_data_path, i=1):
    """

    Parameters
    ----------
    features_path : str
        Absolute path of the features compressed file.
    ds_data_path : str
        Absolute path of the data file (json or compressed csv) containing
        the labels.
    i : int
        If a json file is provided, specify the index to select.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) containing the class labels.
    """
    if ds_data_path.endswith(".json"):
        with open(ds_data_path, "r") as f:
            labels_json = {k: v for k, v in json.load(f)[i].items()}
    else:
        with ZipFile(ds_data_path, "r", ZIP_DEFLATED) as z:
            ds_csv = pd.concat(
                [pd.read_csv(z.open(f))[["sha256", "label"]]
                 for f in z.namelist()], ignore_index=True)
            labels_json = {k: v for k, v in zip(ds_csv.sha256.values,
                                                ds_csv.label.values)}

    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        labels = [labels_json[f.split(".json")[0].lower()]
                  for f in z.namelist()]
    return np.array(labels)


def main(clf, p, n_tr, n_ts):
    train_features_path = "training_set_features.zip"
    train_csv_path = "training_set.zip"

    if not os.path.exists(train_features_path):
        gdown.download(id="1roRQj1fZS8RT_PisoeXACEnjhViRph3H",
                       output=train_features_path)
    if not os.path.exists(train_csv_path):
        gdown.download(id="1T1Tp7Fsz4Gf0IVnX4DURG2Nu5gXbHxgk",
                       output=train_csv_path)

    features_tr = load_features(train_features_path)
    y_tr = load_labels(train_features_path, train_csv_path)
    vectorizer = CountVectorizer(
        input="content", lowercase=False, tokenizer=lambda x: x,
        binary=True, token_pattern=None)
    X_tr = vectorizer.fit_transform(features_tr)

    # feature selection
    if p != 0:
        sel = VarianceThreshold(threshold=(p * (1 - p)))
        X_tr = sel.fit_transform(X_tr)
        selected_features = (vectorizer.get_feature_names_out())[
            sel.get_support()]
        vectorizer = CountVectorizer(
            input="content", lowercase=False, tokenizer=lambda x: x,
            token_pattern=None, binary=True, vocabulary=selected_features)
        vectorizer.fixed_vocabulary_ = False
        vectorizer.stop_words_ = set()

    X_tr, X_ts, y_tr, y_ts = train_test_split(
        X_tr, y_tr, stratify=y_tr, train_size=n_tr, test_size=n_ts)

    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_ts)
    scores = clf.decision_function(X_ts)

    get_metrics(y_ts, y_pred)
    plot_roc(y_ts, scores[:, 1])


if __name__ == "__main__":
    from sparse_inf_svm import SparseInfSVM
    from sparse_inf_svm_2 import SparseInfSVM2

    # clf = SparseInfSVM(C=0.1)
    clf = SparseInfSVM2(t=0.7)

    # 0 -> 1e-4 -> 34636 features; 1e-3 -> 6788 features; 1e-2 -> 1102 features
    p = 1e-3
    n_tr = 10000
    n_ts = 10000

    main(clf, p, n_tr, n_ts)
