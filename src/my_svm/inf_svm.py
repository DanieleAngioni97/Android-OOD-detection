from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy


class InfSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1):
        self.coef_ = None
        self.intercept_ = None
        self.C = C

    def fit(self, X, y):
        (n, d) = X.shape
        y = 2 * y - 1

        g = np.zeros((2 * n + 2 * d, 1))
        g[-n:] = -1

        c = np.zeros((d + 1 + n + 1, 1))
        c[(d+1):] = self.C
        c[-1] = 1

        A = np.zeros((2 * n + 2 * d, d + 1 + n + 1))

        # constraint for w infinity norm minimization
        A[:d, :d] = np.eye(d)
        A[:d, -1] = -1

        A[d:2 * d, :d] = -np.eye(d)
        A[d:2 * d, -1] = -1

        # positivity hinge loss slack constraint
        A[2 * d:-n, d + 1:-1] = -np.eye(n)

        Y = np.tile(y, reps=[d, 1]).T

        # hinge loss's constraint
        A[-n:, :d] = (-Y) * X
        A[-n:, d] = -y
        A[-n:, d + 1:-1] = -np.eye(n)

        res = scipy.optimize.linprog(c, A_ub=A, b_ub=g, bounds=(None, None))
        self.coef_ = res.x[0:d]
        self.intercept_ = res.x[d]
        print("t: ", res.x[-1])
        return self

    def predict(self, X):
        scores = self.decision_function(X)
        ypred = np.argmax(scores, axis=1)
        return ypred

    def decision_function(self, X):
        n_samples = X.shape[0]
        scores = np.dot(X, self.coef_) + self.intercept_
        return np.array([-scores.T, scores.T]).T


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from plots import plot_dataset, plot_decision_regions
    import matplotlib.pyplot as plt

    x, y = make_moons(n_samples=2000, noise=0.2)

    clf = InfSVM(C=0.1)
    clf.fit(x, y)

    print(clf.coef_)
    print(clf.intercept_)

    print(clf.decision_function(x).shape)
    print(clf.predict(x).shape)

    plot_decision_regions(x, y, clf)
    plot_dataset(x, y)
    plt.show()
