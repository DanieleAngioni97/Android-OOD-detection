from my_svm.sparse_inf_svm_2 import SparseInfSVM2
from sklearn.svm import LinearSVC
import utils.data as ut_data
import utils.visualization as ut_viz
import time

def main():
    reduced = '1k'
    splits = ut_data.load_and_preprocess_dataset(updated=True,
                                                 train_size=12,
                                                 reduced=reduced)
    X_train, y_train, _, _, X_tests, y_tests, _, _, _ = splits


    clf_list = [LinearSVC(C=1),
                SparseInfSVM2(t=0.7)
                ]

    fig, axs = ut_viz.create_figure(nrows=2, ncols=2, fontsize=15, figsize=(5, 10))
    for i, clf in enumerate(clf_list):
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        training_time = t2 - t1
        print(f"Trained in {training_time} seconds")
        ax = axs[0, i]
        # ut_viz.evaluate_and_plot_metrics(clf, X_tests, y_tests, ax=ax1)

        t1 = time.time()
        y_preds = [clf.predict(X_test) for X_test in X_tests]
        t2 = time.time()
        testing_time = t2 - t1
        print(f"Tested in {testing_time} seconds")
        ut_viz.plot_metrics(y_preds, y_tests, ax=ax)
        ax.set_ylim(0, 1)

        ax = axs[1, i]
        ut_viz.plot_weight_distribution(clf, ax)

    fig.suptitle(f"Train/Test time: {training_time:.3f}/{testing_time:.3f}\nNumber of features: {X_train.shape[1]}")
    fig.tight_layout()
    fig.show()
    fig.savefig(f'../figures/inf_svm-{reduced}.pdf')


    print("")

if __name__ == '__main__':
    main()


