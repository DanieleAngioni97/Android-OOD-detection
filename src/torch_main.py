import copy
import os

import sklearn.metrics
# import gdown
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from scipy.sparse import vstack
from models.base.base_torch_svm import LinearSVM
from pytorch_ood.loss import EnergyRegularizedLoss, EntropicOpenSetLoss, CrossEntropyLoss
from pytorch_ood.detector import MaxSoftmax, EnergyBased, Entropy, OpenMax
from utils import data as ut_data
from utils import fm
from utils.fm import my_save, my_load
from tesseract import temporal
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC
import utils.visualization as ut_viz
import utils.evaluation as ut_eval
from utils.utils import set_all_seed
import math
import utils

def main():
    ##############################################################################################################
    # Input arguments
    ##############################################################################################################
    test_only = False
    overwrite_detectors = True
    use_only_train_features = True

    use_pretrained_svm = True
    train_torch_model = False
    reduced = None
    
    b_size = 64
    max_train_samples = None
    random_seed = 0

    path = "../experiments"
    mw_weight = 4
    # torch_svm_fname = f"linear_torch-nfeat-{reduced}-mw_weight-{mw_weight}"
    torch_svm_fname = f"linear_torch-pretrained_svm"
    sklearn_svm_fname = 'sklearn_svm'

    os.makedirs(path, exist_ok=True)

    ##############################################################################################################
    # Loading training and testing features
    ##############################################################################################################
    splits = ut_data.load_and_preprocess_dataset(updated=True,
                                                 reduced=reduced,
                                                 use_only_train_features=use_only_train_features,
                                                 max_train_samples=max_train_samples)
    X_train, y_train, t_train, meta_train, X_tests, y_tests, t_tests, meta_tests, feature_names = splits

    fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    ut_viz.plot_dataset_stats(y_tests, ax=ax)
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(path, f'testset_stats.pdf'))


    ##############################################################################################################
    # Train and evaluate the vanilla Linear SVM from sklearn
    ##############################################################################################################
    print(f"Final X_train shape: {X_train.shape}")
    n_train_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_train_features = (X_train.sum(axis=0) != 0).sum()  # features that appear only in the train set
    n_all_zero_train_samples = (X_train.sum(axis=1) != 0).sum()  # samples with all zero features (???)
    max_n_samples = n_train_samples

    sklearn_svm_path = os.path.join(path, f'{sklearn_svm_fname}-nfeat-{reduced}')
    if not os.path.exists(f'{sklearn_svm_path}.pkl'):
        sklearn_svm = LinearSVC(C=1)
        sklearn_svm.fit(X_train, y_train)
        my_save(sklearn_svm, f'{sklearn_svm_path}.pkl')
    else:
        sklearn_svm = my_load(f'{sklearn_svm_path}.pkl')

    fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    ut_viz.evaluate_and_plot_metrics(sklearn_svm, X_tests, y_tests, ax=ax,
                              ylabel='sklearn Linear SVM')
    fig.tight_layout()
    ax.set_ylim(0, 1)
    fig.show()
    fig.savefig(f'{sklearn_svm_path}.pdf')

    ##############################################################################################################
    # Create dataloaders
    ##############################################################################################################
    # apply a feature selection method?

    train_loader, test_loaders = ut_data.create_dataloaders(X_train, y_train, t_train, meta_train,
                                                            X_tests, y_tests, t_tests, meta_tests,
                                                            random_seed=random_seed)
    # set_all_seed(random_seed)
    # sparse_dataset = ut_data.SparseDataset(X_train, y_train, t_train, meta_train)
    # train_loader = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
    #                           collate_fn=ut_data.dense_batch_collate)
    #
    # test_loader_list = []
    # for X_test, y_test, t_test, meta_test in zip(X_tests, y_tests, t_tests, meta_tests):
    #     sparse_dataset = ut_data.SparseDataset(X_test, y_test, t_test, meta_test)
    #     test_loader_i = DataLoader(sparse_dataset, batch_size=b_size, shuffle=False,
    #                                collate_fn=ut_data.dense_batch_collate)
    #     test_loader_list.append(test_loader_i)

    print("")

    ##############################################################################################################
    # Training Loop
    ##############################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    torch_svm_path = os.path.join(path, torch_svm_fname)
    if not os.path.exists(f"{torch_svm_path}.pt"):
        torch_linear = LinearSVM(n_features=n_features, C=1, lr=1e-3, class_weight=(1, mw_weight))
        if use_pretrained_svm:
            torch_linear.load_sklearn_pretrained(sklearn_svm)
            torch_linear.to(device)
        else:
            torch_linear.to(device)
            set_all_seed(random_seed)
            torch_linear.fit(train_loader, epochs=1, device='cpu')

            fig, ax = torch_linear.plot_loss_path()
            fig.tight_layout()
            fig.show()
            fig.savefig(f"{torch_svm_path}.pdf")

            # fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
            # # ut_viz.evaluate_and_plot_metrics(classifier, [X_train]*3, [y_train]*3, ax=ax,
            # #                           ylabel='Torch Linear')
            # ut_viz.evaluate_and_plot_metrics(classifier, X_tests, y_tests, ax=ax,
            #                           ylabel='Torch Linear')
            # fig.tight_layout()
            # fig.show()

        torch.save(torch_linear.to('cpu'), f"{torch_svm_path}.pt")

    else:
        torch_linear = torch.load(f"{torch_svm_path}.pt")
    fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    ut_viz.evaluate_and_plot_metrics(torch_linear, X_tests, y_tests, ax=ax,
                              ylabel='Torch Linear')
    title = f"feat_reduct: {reduced}, mw_weight: {mw_weight}"
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.show()

    fig.savefig(os.path.join(path, f"{torch_svm_path}.pdf"))

    print("")



    detector = EnergyBased(torch_linear)
    rej_outputs = ut_eval.get_outputs(torch_linear, X_tests)
    rej_y_tests = []
    rej_perc = 0.1
    rej_pile = []

    X_val, y_val, t_val, meta_val = X_tests[0], y_tests[0], t_tests[0], meta_tests[0]
    out = ut_eval.get_outputs(torch_linear, X_val)[0]
    ood_score = detector.score(torch.tensor(out)).numpy()
    n_rejected_samples = int(rej_perc * out.shape[0])
    rejection_threshold = np.sort(ood_score)[-n_rejected_samples]

    for i, (out, y_test) in enumerate(zip(rej_outputs, y_tests)):
        # ood_score = -np.log(1 / (1 + np.exp(out)))
        ood_score = detector.score(torch.tensor(out)).numpy()
        rejected_samples = int(rej_perc * out.shape[0])

        # idx = ood_score.argsort(axis=0)
        # to_be_rej_idx = idx[-rejected_samples:]
        # to_keep_idx = idx[:-rejected_samples]

        idx = np.arange(y_test.size)
        rej_idx = (ood_score > rejection_threshold)
        to_be_rej_idx = idx[rej_idx]
        to_keep_idx = idx[~rej_idx]

        rej_outputs[i] = out[to_keep_idx.flatten(), :]
        rej_y_tests.append(y_test[to_keep_idx.flatten()])

        rej_pile.append(y_test[to_be_rej_idx.flatten()])
        print("")

    fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    rej_y_preds = ut_eval._get_predictions(rej_outputs)
    ut_viz.plot_metrics(rej_y_preds, rej_y_tests, ax=ax,
                              ylabel='Torch Linear')
    title = f"torch Linear + Energy (rej={rej_perc})"
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(path, f"{torch_svm_path}-rej-{rej_perc}.pdf"))





    alpha = .7
    fig, axs = ut_viz.create_figure(ncols=4, fontsize=15, figsize=(5, 10))

    for i, (funct, metric_name) in enumerate(zip((recall_score, precision_score, f1_score),
                                                 ('Precision', 'Recall', 'F1-score'))):
        ax = axs[i]
        res = [funct(ut_eval.get_predictions(sklearn_svm, X_test)[0], y_test)
               for X_test, y_test in zip(X_tests, y_tests)]
        ax.plot(res, label='sklearn SVM',
                marker='^', alpha=alpha)
        res = [funct(ut_eval.get_predictions(torch_linear, X_test)[0], y_test)
               for X_test, y_test in zip(X_tests, y_tests)]
        ax.plot(res, label='torch Linear',
                marker='v', alpha=alpha)
        res = [funct(y_pred, y_test)
               for y_pred, y_test in zip(rej_y_preds, rej_y_tests)]
        ax.plot(res, label=f"torch Linear + Energy (rej={rej_perc})",
                marker='o', alpha=alpha)
        ax.legend()
        ax.set_title(metric_name)
        ax.set_ylim(0, 1)
        ax.grid(axis='y')


    # fig.show()

    max_fpr = 0.05
    ax = axs[-1]
    res = [roc_auc_score(ut_eval.get_predictions(sklearn_svm, X_test)[0], y_test, max_fpr=max_fpr)
           for X_test, y_test in zip(X_tests, y_tests)]
    ax.plot(res, label='sklearn SVM',
            marker='^', alpha=alpha)
    res = [roc_auc_score(ut_eval.get_predictions(torch_linear, X_test)[0], y_test, max_fpr=max_fpr)
           for X_test, y_test in zip(X_tests, y_tests)]
    ax.plot(res, label='torch Linear',
            marker='v', alpha=alpha)
    res = [roc_auc_score(y_pred, y_test, max_fpr=max_fpr)
           for y_pred, y_test in zip(rej_y_preds, rej_y_tests)]
    ax.plot(res, label=f"torch Linear + Energy (rej={rej_perc})",
            marker='o', alpha=alpha)
    ax.legend()
    ax.set_title(f'AUROC (max FPR = {max_fpr*100}%)')
    # ax.set_ylim(0, 1)
    ax.grid(axis='y')
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(path, "comparisons.pdf"))




    # y_preds = ut_eval._get_predictions(outputs)
    # fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    # ut_viz.plot_metrics(y_preds, y_tests_temp, ax=ax,
    #                           ylabel='Energy', plot_prec=False, plot_f1=False, plot_gw=False)
    # ax.set_ylim(0, 1)
    # title = f"feat_reduct: {reduced}, mw_weight: {mw_weight}, rej_perc: {rej_perc}"
    # ax.set_title(title)
    # fig.tight_layout()
    # fig.show()

    # fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    # # y_scores = ut_eval.get_outputs(sklearn_svm, X_tests)
    # # aurocs = [roc_auc_score(y_test, y_score, max_fpr=0.01) for (y_test, y_score) in zip(y_tests, y_scores)]
    # ax.plot(aurocs, label='sklearn SVM')
    # fig.tight_layout()
    # ax.set_ylim(0, 1)
    # fig.show()

    #
    # fig, ax = ut_viz.create_figure(fontsize=15, figsize=(5, 10))
    # n_mw_rej = [(y == 1).sum() for y in rej_pile]
    # n_gw_rej = [(y == 0).sum() for y in rej_pile]
    #
    # n_gw = [(y_test == 0).sum() for y_test in y_tests]
    # n_mw = [(y_test == 1).sum() for y_test in y_tests]
    #
    #
    # for x, y in test_loader_list[0]:
    #     out = classifier(x)
    #     score = -t * torch.log(torch.sigmoid(out / t))
    #
    #
    # print("")
    # return
    # detectors_path = os.path.join(path, f"detectors.pkl")
    # if overwrite_detectors or (not os.path.isfile(detectors_path)):
    #     # The detector that we want to test from the paper [1] - [6] - [26] (in addition to the MaxSoftMax baseline detector)
    #     print("Creating OOD Detectors")
    #
    #     detectors = {}
    #     detectors["MaxSoftmax"] = MaxSoftmax(torch.load(os.path.join(path, 'linear_retrained_CE_loss.pt')))
    #     detectors["Entropy"] = Entropy(torch.load(os.path.join(path, 'linear_retrained_CE-Entropy_loss.pt')))
    #     detectors["EnergyBased"] = EnergyBased(torch.load(os.path.join(path, 'linear_retrained_CE-Energy_loss.pt')))
    #     detectors["OpenMax"] = OpenMax(torch.load(os.path.join(path, 'linear_retrained_CE_loss.pt')))
    #
    #     print(f"> Fitting {len(detectors)} detectors")
    #
    #     for name, detector in detectors.items():
    #         print(f"--> Fitting {name}")
    #         detector.fit(train_loader, device=device)
    #         # fm.my_save(detector, os.path.join(path, f"detector-{name}.pkl"))
    #     fm.my_save(detectors, os.path.join(path, f"detectors.pkl"))
    # else:
    #     detectors = fm.my_load(detectors_path)
    #
    # import json
    # from utils.evaluation import detector_scores
    #
    # datasets = {}
    #
    # results_list = []
    # for i, loader in enumerate(test_loader_list):
    #     datasets["OOD test set "] = loader  # todo: this may be deleted
    #     results = detector_scores(detectors, datasets)
    #     results_list.append(results)
    #
    # # Saving the results in a json file
    # fm.my_save(results_list, os.path.join(path, f'test_results.pkl'))
    # # with open(f'test_results_{i}.json', 'w') as f:
    # #     json.dump(results, f)
    #
    # fm.my_load(results_list, os.path.join(path, f'test_results.pkl'))
    # print("")


if __name__ == '__main__':
    main()