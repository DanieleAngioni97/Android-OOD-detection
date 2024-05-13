if __name__ == "__main__":
    from inf_svm import InfSVM
    from sparse_inf_svm import SparseInfSVM
    from sklearn.datasets import make_moons
    import numpy as np

    x, y = make_moons(n_samples=2000, noise=0.2)

    dense = InfSVM(C=0.1)
    dense.fit(x, y)
    y_pred_dense = dense.predict(x)
    scores_dense = dense.decision_function(x)

    sparse = SparseInfSVM(C=0.1)
    sparse.fit(x, y)
    y_pred_sparse = sparse.predict(x)
    scores_sparse = sparse.decision_function(x)

    assert np.array_equal(dense.coef_, sparse.coef_)
    assert np.array_equal(y_pred_dense, y_pred_sparse)
    assert np.array_equal(scores_dense, scores_sparse)

