from utils import fm
import os
import utils.data as ut_data
import utils.visualization as ut_viz
import utils.evaluation as ut_eval
from utils.constants import MODEL_NAME_TO_CLASS_DICT, MODEL_CLASS_LIST, MODEL_NAME_LIST, HPARAMS_DICT
import numpy as np
import sklearn

import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, calibration_curve
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def main():
    models_path = '../experiments/sklearn_models/'
    figures_path = '../figures/'
    detectors_path = '../experiments/detectors/'
    reduced = None
    calibrate_clf = True
    cal_method = 'sigmoid'

    cache_det_results = True

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(detectors_path, exist_ok=True)

    splits = ut_data.load_and_preprocess_dataset(updated=True,
                                                 train_size=11,
                                                 reduced=reduced)
    X_train, y_train, _, _, X_tests, y_tests, _, _, _ = splits
    X_val, y_val = X_tests.pop(0), y_tests.pop(0)

    # i = 1
    # model_class, model_name = MODEL_CLASS_LIST[i], MODEL_NAME_LIST[i]
    for i, (model_name, model_class) in enumerate(MODEL_NAME_TO_CLASS_DICT.items()):
        hparams = HPARAMS_DICT[model_name]
        model_name += str(hparams).replace(' ', '')
        if os.path.exists(os.path.join(models_path, model_name + '.pkl')):
            clf = fm.my_load(os.path.join(models_path, model_name + '.pkl'))
        else:
            clf = model_class(**hparams)
            clf.fit(X_train, y_train)
            fm.my_save(clf, os.path.join(models_path, model_name + '.pkl'))

        if calibrate_clf:
            model_name += '-calibrated'
            if os.path.exists(os.path.join(models_path, model_name + '.pkl')):
                clf = fm.my_load(os.path.join(models_path, model_name + '.pkl'))
            else:
                clf = CalibratedClassifierCV(clf, method=cal_method).fit(X_val, y_val)
                fm.my_save(clf, os.path.join(models_path, model_name + '.pkl'))

        # fig1, axs1 = ut_viz.create_figure(nrows=2, fontsize=15, figsize=(5, 10))
        # ax1 = axs1[0]
        # # ut_viz.evaluate_and_plot_metrics(clf, X_tests, y_tests, ax=ax1)
        # y_preds = [clf.predict(X_test) for X_test in X_tests]
        # ut_viz.plot_metrics(y_preds, y_tests, ax=ax1)
        # ax1.set_title(f"{model_name}\n{hparams}")
        # ax1.set_ylim(0, 1)
        #
        # ece_list = []
        # skip_n_month = 5
        # n_bins = 10
        # fig2, axs2 = ut_viz.create_figure(nrows=2, ncols=(len(X_tests) // skip_n_month) + 1,
        #                                   fontsize=15, figsize=(5, 10), squeeze=False)
        # for month, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
        #     if isinstance(clf, sklearn.svm.LinearSVC):
        #         y_prob = clf._predict_proba_lr(X_test)[:, 1]
        #     else:
        #         y_prob = clf.predict_proba(X_test)[:, 1]
        #     prob_true, prob_pred, ece, bin_total = ut_eval.my_calibration_curve(y_test, y_prob, n_bins=n_bins)
        #
        #     if month % skip_n_month == 0:
        #         plot_i = month // skip_n_month
        #         ax2 = axs2[0, plot_i]
        #         disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
        #         disp.plot(ax=ax2)
        #         ax2.set_title(f"Month {month} (ECE={ece:.3f})")
        #
        #         ax2 = axs2[1, plot_i]
        #         bins = np.arange(0, 1, n_bins + 1)
        #         ax2.plot(prob_pred, bin_total)
        #         ax2.set_xticks(ticks=np.arange(0, 1.1, 0.2))
        #         # ax2.set_ylim(1e-1, 1e-4)
        #         ax2.set_yscale('log')
        #         ax2.set_xlim(0, 1)
        #         ax2.set_ylabel('# samples')
        #     ece_list.append(ece)
        # ece_list = np.array(ece_list)
        #
        # ax1 = axs1[1]
        # ax1.plot(ece_list, marker='o')
        # ax1.set_ylabel(f'Expected Calibration Error\n(Avg. {ece_list.mean():.4f}')
        # ax1.set_xlabel('month')
        # ax1.set_ylim(0, 1)
        # fig1.tight_layout()
        # fig1.show()
        # m = model_name.replace('{', '(').replace('}', ')')
        # fig1.savefig(os.path.join(figures_path, f'PERF-{m}.pdf'))
        #
        # fig2.suptitle(f'Calibration Curves ({model_name})')
        # fig2.tight_layout()
        # fig2.show()
        # fig2.savefig(os.path.join(figures_path, f'CAL_CURVES-{m}.pdf'))



        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.pipeline import make_pipeline
        from custom_outlier_detectors import MaxConfidence

        outliers_fraction = 0.05

        anomaly_algorithms = [
            (
                'Confidence', MaxConfidence(clf=clf, rej_perc=outliers_fraction)
            ),
            (
                "One-Class SVM",
                OneClassSVM(nu=outliers_fraction, kernel="linear")),
            (
                "Isolation Forest",
                IsolationForest(contamination=outliers_fraction, random_state=42),
            ),
            (
                "Local Outlier Factor",
                LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=True),
            )
        ]

        # fig, axs = ut_viz.create_figure(nrows=3, ncols=len(anomaly_algorithms) + 1,
        #                                 fontsize=20, figsize=(5, 10))
        fig, axs = ut_viz.create_figure(nrows=2, ncols=3,
                                        fontsize=25, figsize=(5, 10))
        # fig_rej, axs_rej = ut_viz.create_figure(nrows=1, ncols=3,
        #                                         fontsize=25, figsize=(5, 10))
        # k = 0
        # name, algorithm = anomaly_algorithms[k]
        for k, (name, algorithm) in enumerate(anomaly_algorithms):
            algorithm_dict = {}
            for class_id, class_name in enumerate(['gw', 'mw']):
                print(f">>> Detector: {name}")
                base_detector_fname = f"d-{name.replace(' ', '_')}-m-{model_name}"
                detector_fname = base_detector_fname + f"_c-{class_name}"
                detector_file_path = os.path.join(detectors_path, detector_fname + '.pkl')
                if os.path.exists(detector_file_path) and cache_det_results:
                    algorithm = fm.my_load(detector_file_path)
                    algorithm_dict[class_name] = algorithm
                else:
                    t0 = time.time()
                    algorithm.fit(X_tests[0][y_tests[0] == class_id, :])
                    # algorithm.fit(X_val[y_val == class_id, :])
                    if name == 'Confidence':
                        # algorithm.set_threshold(X_val[y_val == class_id, :])
                        algorithm.set_threshold(X_tests[0][y_tests[0] == class_id, :])
                    t1 = time.time()
                    fm.my_save(algorithm, detector_file_path)
                    algorithm_dict[class_name] = algorithm
                    print(f"Elapsed time: {t1 - t0}")

            results_file_path = os.path.join(detectors_path, 'PREDS-' + base_detector_fname + '-double_det.pkl')
            if os.path.exists(results_file_path) and cache_det_results:
                data = fm.my_load(results_file_path)
            else:
                t0 = time.time()
                rejected_gw = [algorithm_dict['gw'].predict(X_test) for X_test in X_tests]
                rejected_mw = [algorithm_dict['mw'].predict(X_test) for X_test in X_tests]
                y_preds_list = [clf.predict(X_test) for X_test in X_tests]

                rejected = []
                for rej_m, rej_g, y_pred in zip(rejected_gw, rejected_mw, y_preds_list):
                    rej = rej_g
                    rej[y_pred == 1] = rej_m[y_pred == 1]
                    rej = (rej == -1)
                    rejected.append(rej)

                y_preds_kept = [y_preds[~rej] for y_preds, rej in zip(y_preds_list, rejected)]
                y_tests_kept = [y_test[~rej] for y_test, rej in zip(y_tests, rejected)]
                y_preds_rej = [y_preds[rej] for y_preds, rej in zip(y_preds_list, rejected)]
                y_tests_rej = [y_test[rej] for y_test, rej in zip(y_tests, rejected)]
                t1 = time.time()
                print(f"Elapsed time: {t1 - t0}")
                data = (y_preds_kept, y_tests_kept, y_preds_rej, y_tests_rej)
                fm.my_save(data, results_file_path)

            y_preds_kept, y_tests_kept, y_preds_rej, y_tests_rej = data


            # fig, axs = ut_viz.create_figure(nrows=3, ncols=len(anomaly_algorithms) + 1, fontsize=15, figsize=(5, 10))

            # # fig, axs = ut_viz.create_figure(nrows=3, fontsize=15, figsize=(5, 10))
            # ax = axs[0, k]
            # ut_viz.plot_metrics(y_preds_kept, y_tests_kept, ax=ax, plot_gw=False)
            # ax.set_title(f"{name}")
            # ax.set_ylabel('Kept')
            # ax.set_ylim(0, 1)
            #
            # ax = axs[1, k]
            # ut_viz.plot_metrics(y_preds_rej, y_tests_rej, ax=ax, plot_gw=False)
            # # ax.set_title(f"{model_name}\n{hparams}")
            # ax.set_ylabel('Rejected')
            # ax.set_ylim(0, 1)
            #
            # ax = axs[2, k]
            # #
            # n_rej_gw = np.array([(y_test_rej == 0).sum() for y_test_rej in y_tests_rej])
            # n_rej_mw = np.array([(y_test_rej == 1).sum() for y_test_rej in y_tests_rej])
            # n_rej = np.array([y_test_rej.size for y_test_rej in y_tests_rej])
            #
            # n_gw_per_month = np.array([(y_test == 0).sum() for y_test in y_tests])
            # n_mw_per_month = np.array([(y_test == 1).sum() for y_test in y_tests])
            # n_sample_per_month = np.array([y_test.size for y_test in y_tests])
            #
            alpha = 0.7
            for i, (funct, metric_name) in enumerate(zip((recall_score, precision_score, f1_score),
                                                         ('Precision', 'Recall', 'F1-score'))):
                ax = axs[0, i]
                res = [funct(y_pred_kept, y_test)
                       for y_pred_kept, y_test in zip(y_preds_kept, y_tests_kept)]
                ax.plot(res, marker=ut_viz.DET_MARKERS[k], label=name, alpha=alpha)
                #
                # ax = axs[1, i]
                # res = [funct(y_pred_rej, y_test)
                #        for y_pred_rej, y_test in zip(y_preds_rej, y_tests_rej)]
                # ax.plot(res, marker=DET_MARKERS[k], label=name, alpha=alpha)
                ax.set_ylim(0, 1)
                ax.grid(axis='y')



            n_gw_per_month = np.array([(y_test == 0).sum() for y_test in y_tests])
            n_mw_per_month = np.array([(y_test == 1).sum() for y_test in y_tests])
            n_sample_per_month = np.array([y_test.size for y_test in y_tests])

            n_rej_gw = np.array([(y_test_rej == 0).sum() for y_test_rej in y_tests_rej]) / n_gw_per_month
            n_rej_mw = np.array([(y_test_rej == 1).sum() for y_test_rej in y_tests_rej]) / n_mw_per_month
            n_rej = np.array([y_test_rej.size for y_test_rej in y_tests_rej]) / n_sample_per_month

            axs_rej = axs
            titles = ['Rejected samples', 'Rejected goodware', 'Rejected malware']
            curves = [n_rej, n_rej_gw, n_rej_mw]
            for pi in range(3):
                if k == 0:
                    axs_rej[1, pi].axhline(y=outliers_fraction, xmax=len(X_tests), label='validation rej. fraction',
                               color='grey', linestyle='dashed')
                    axs_rej[1, pi].set_title(titles[pi])
                    axs_rej[1, pi].set_ylim(0, 1)
                    axs_rej[1, pi].grid('y')
                    axs_rej[1, pi].set_ylim(0, 1)

                axs_rej[1, pi].plot(curves[pi], label=name,
                                   # marker='v', color='b', linestyle='dashed',
                                   alpha=0.7)



        alpha = 0.7
        for i, (funct, metric_name) in enumerate(zip((recall_score, precision_score, f1_score),
                                                     ('Precision', 'Recall', 'F1-score'))):
            ax = axs[0, i]
            res = [funct(clf.predict(X_test), y_test)
                   for X_test, y_test in zip(X_tests, y_tests)]
            ax.plot(res, marker='o', linestyle='dashed', color='grey', label='No rejection', alpha=alpha)

            ax.set_ylim(0, 1)
            ax.grid(axis='y')

        axs[0, 0].legend()
        axs[1, 0].legend()

        axs[0, 0].set_title('Precision')
        axs[0, 1].set_title('Recall')
        axs[0, 2].set_title('F1-score')

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(figures_path, 'OUT_COMPARISON-' + m + '_prova_.pdf'))
        # fname = name.replace(' ', '_')
        # fig.savefig(f'../figures/{fname}.pdf')
        print("")

    #########################################################
    #   Sweep di C su SVM linear + plot dei pesi
    #########################################################
    # i = 0
    # model_name, model_class = MODEL_NAME_LIST[i], MODEL_CLASS_LIST[i]
    #
    # C_list = [1e-5, 1e-4, 1e-2, 1e-1, 1]
    # # C_list = [1]
    # fig, axs = ut_viz.create_figure(nrows=2, ncols=len(C_list),
    #                                 fontsize=15, figsize=(5, 10),
    #                                 squeeze=False)
    # for j, C in enumerate(C_list):
    #     hparams = {'C': C}
    #     clf = model_class(**hparams)
    #
    #     if mw_weight is not None:
    #         sample_weight = np.ones(y_train.shape)
    #         sample_weight[y_train == 1] = mw_weight
    #     else:
    #         sample_weight = None
    #
    #     clf.fit(X_train, y_train, sample_weight)
    #
    #     ax = axs[0, j]
    #     ut_viz.evaluate_and_plot_metrics(clf, X_tests, y_tests, ax=ax)
    #     ax.set_title(f"{model_name}\n{hparams}")
    #     ax.set_ylim(0, 1)
    #
    #     ax = axs[1, j]
    #     ut_viz.plot_weight_distribution(clf, ax=ax)
    #     ax.set_ylim(1e-3, 10)
    #     ax.set_yscale('log')
    #     print("")
    #
    # fig.suptitle(f'mw weight: {mw_weight}')
    # fig.tight_layout()
    # fig.show()
    # fig.savefig(f'../figures/linear_svm-mw_weight-{mw_weight}.pdf')



    print("")

if __name__ == '__main__':
    main()