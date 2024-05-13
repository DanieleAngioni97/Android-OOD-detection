import sklearn.svm


class MaxConfidence:
    def __init__(self, clf,
                 rej_perc=0.15):
        self.clf = clf
        self.rej_perc = rej_perc
        self.threshold = 0.1

    def fit(self, *args):
        pass

    def predict_proba(self, X):
        if isinstance(self.clf, sklearn.svm.LinearSVC):
            return self.clf._predict_proba_lr(X)
        else:
            return self.clf.predict_proba(X)

    def predict(self, X_test):
        pred = self.predict_proba(X_test).argmax(axis=1)
        conf = self.predict_proba(X_test).max(axis=1)
        mask = conf < self.threshold
        pred[mask] = -1
        return pred

    def set_threshold(self, X):
        maxconf = self.predict_proba(X).max(axis=1)
        maxconf.sort()
        n_rejected_samples = int(self.rej_perc * X.shape[0])
        self.threshold = maxconf[n_rejected_samples]

