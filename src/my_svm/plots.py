from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, \
    precision_score


def plot_dataset(x, y, feat0=0, feat1=1):
    colors = ['b.', 'r.', 'g.', 'k.', 'c.', 'm.']
    class_labels = np.unique(y).astype(int)
    for k in class_labels:
        plt.plot(x[y == k, feat0], x[y == k, feat1], colors[k % 7])


def plot_decision_regions(x, y, classifier, resolution=1e-2):
    # setup marker generator and color map
    colors = ('blue', 'red', 'lightgreen', 'black', 'cyan', 'magenta')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.02, x[:, 0].max() + 0.02
    x2_min, x2_max = x[:, 1].min() - 0.02, x[:, 1].max() + 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


def plot_roc(y_true, scores, img_path="", title="Roc"):
    fpr, tpr, th = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.semilogx(fpr, tpr, color="darkorange", lw=2,
                 label=f"AUC = {roc_auc:0.2f}")
    plt.axvline(fpr[np.argmin(np.abs(th))], color="k", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if img_path > "":
        plt.savefig(img_path)
    plt.show()
    plt.clf()


def get_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"TPR (aka Recall): {tp / (tp + fn):.4f}")
    print(f"FPR: {fp / (fp + tn):.4f}")



if __name__ == "__main__":
    x, y = make_moons(n_samples=4, noise=0.2)

    C=10.0

    clf = SVC(C=C, kernel="linear")
    clf.fit(x, y)

    print(clf.coef_)
    print(clf.intercept_)

    plot_decision_regions(x,y,clf)
    plot_dataset(x,y)
    plt.show()
