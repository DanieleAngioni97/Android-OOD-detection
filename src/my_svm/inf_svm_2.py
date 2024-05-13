from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy


class InfSVM2(BaseEstimator, ClassifierMixin):

    def __init__(self, t=1):
        self.coef_ = None
        self.intercept_ = None
        self.t = t

    def fit(self, X, y):
        (n, d) = X.shape
        y = 2 * y - 1

        g = -np.ones(shape=(n, 1))

        c = np.zeros(shape=(d + 1 + n, 1))
        c[-n:] = 1

        A = np.zeros(shape=(n, d + 1 + n))

        Y = np.tile(y, reps=[d, 1]).T

        # hinge loss's constraint
        A[:, :d] = (-Y) * X
        A[:, d] = -y
        A[:, d + 1:] = -np.eye(n)

        #print(c,g,np.round(A,2))
        lb = np.array([None] * (d+1+n))
        ub = np.array([None] * (d+1+n))

        lb[0:d] = -self.t
        lb[d+1:] = 0
        ub[0:d] = self.t

        res = scipy.optimize.linprog(c, A_ub=A, b_ub=g,
                                     bounds=np.vstack((lb, ub)).T)
        self.coef_ = res.x[0:d]
        self.intercept_ = res.x[d]
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

    clf = InfSVM2(t=0.7)
    clf.fit(x, y)

    print(clf.coef_)
    print(clf.intercept_)

    print(clf.decision_function(x).shape)
    print(clf.predict(x).shape)

    plot_decision_regions(x, y, clf)
    plot_dataset(x, y)
    plt.show()
