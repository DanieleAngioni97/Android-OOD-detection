import copy
import os
# import gdown
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from scipy.sparse import vstack
from models.base.base_torch_svm import LinearSVM
from pytorch_ood.loss import EnergyRegularizedLoss, EntropicOpenSetLoss, CrossEntropyLoss
from pytorch_ood.detector import MaxSoftmax, EnergyBased, Entropy, OpenMax
from utils import visualization as viz
from utils import data as ut_data
from utils import fm
from utils.fm import my_save, my_load
from tesseract import temporal
from sklearn.svm import LinearSVC
from utils.visualization import create_figure, evaluate_and_plot_metrics


if __name__ == '__main__':
    ##############################################################################################################
    # Input arguments
    ##############################################################################################################
    test_only = False
    overwrite_detectors = True
    use_only_train_features = True
    method_train_list = ['CE-Entropy']
    b_size = 1000

    path = "../experiments"
    sklearn_svm_path = os.path.join(path, 'sklearn_svm.pkl')
    torch_svm_path = os.path.join(path, f"linear_torch_from_skratch.pt")

    ##############################################################################################################
    # Loading training and testing features
    ##############################################################################################################

    data_dict = ut_data.load_dataset(updated=False, reduced=None)
    X, y, time_index, feature_names, T, meta = tuple(data_dict.values())
    splits = temporal.time_aware_train_test_split(
        X, y, T, meta, train_size=12, test_size=1, granularity='month')
    X_train, X_tests, y_train, y_tests, t_train, t_tests, meta_train, meta_tests = splits

    if use_only_train_features:
        # remove features that do not appear in the train set
        idx_tr_features = np.array(X_train.sum(axis=0)).flatten() > 0
        X_train = X_train[:, idx_tr_features]
        X_tests = [X_test[:, idx_tr_features] for X_test in X_tests]

        # remove samples with all-zero features from the train set
        idx_non_zero_tr_samples = np.array(X_train.sum(axis=1)).flatten() > 0
        X_train = X_train[idx_non_zero_tr_samples, :]
        y_train = y_train[idx_non_zero_tr_samples]
        t_train = t_train[idx_non_zero_tr_samples]
        meta_train = meta_train[idx_non_zero_tr_samples]


    ##############################################################################################################
    # Train and evaluate the vanilla Linear SVM from sklearn
    ##############################################################################################################
    print(f"X_train shape: {X_train.shape}")
    n_train_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_train_features = (X_train.sum(axis=0) != 0).sum()     # features that appear only in the train set
    n_all_zero_train_samples = (X_train.sum(axis=1) != 0).sum()     # samples with all zero features (???)
    max_n_samples = n_train_samples

    if not os.path.exists(sklearn_svm_path):
        classifier = LinearSVC(C=1)
        classifier.fit(X_train, y_train)
        my_save(classifier, os.path.join(path, 'sklearn_svm.pkl'))
    else:
        classifier = my_load(sklearn_svm_path)

    fig, ax = create_figure(fontsize=15, figsize=(5, 10))
    evaluate_and_plot_metrics(classifier, X_tests, y_tests, ax=ax,
                              ylabel='SVM baseline')
    fig.tight_layout()
    fig.show()



    ##############################################################################################################
    # Create dataloaders
    ##############################################################################################################
    # apply a feature selection method?

    sparse_dataset = ut_data.SparseDataset(X_train, y_train, t_train, meta_train)
    train_loader = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                             collate_fn=ut_data.dense_batch_collate)

    test_loader_list = []
    for X_test, y_test, t_test, meta_test in zip(X_tests, y_tests, t_tests, meta_tests):
        sparse_dataset = ut_data.SparseDataset(X_test, y_test, t_test, meta_test)
        test_loader_i = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                                 collate_fn=ut_data.dense_batch_collate)
        test_loader_list.append(test_loader_i)


    print("")



    ##############################################################################################################
    # Training Loop
    ##############################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(torch_svm_path):
        classifier = LinearSVM(n_features=n_features, C=1, lr=1e-3)

        classifier.to(device)
        classifier.fit(train_loader, epochs=10, device='cpu')

        fig, ax = classifier.plot_loss_path()
        fig.tight_layout()
        fig.show()

        torch.save(classifier.to('cpu'), torch_svm_path)

    else:
        classifier = torch.load(torch_svm_path)
        fig, ax = create_figure(fontsize=15, figsize=(5, 10))
        evaluate_and_plot_metrics(classifier, X_tests, y_tests, ax=ax,
                                  ylabel='SVM baseline')
        fig.tight_layout()
        fig.show()
    print("")
    # fig, ax = viz.create_figure(figsize=(7, 7), fontsize=15)
    # alpha = .7
    # ax.plot(loss_ce_path, label='ce', alpha=alpha)
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('Loss')
    # fig.legend()
    # fig.tight_layout()
    # fig.show()

    # if not test_only:
    #     for idx, method in enumerate(method_train_list):
    #         print(f"### Training with {method} loss ###")
    #         loss_ce_fn = CrossEntropyLoss()
    #         loss_energy_fn = EnergyRegularizedLoss()
    #         loss_entropy_fn = EntropicOpenSetLoss()
    #
    #         loss_tot_path = []
    #         loss_ce_path = []
    #         loss_energy_path = []
    #         loss_entropy_path = []
    #
    #         # Baseline Linear Layer
    #         linear = copy.deepcopy(linear_pretrained)
    #         # linear.linear.weight = torch.nn.Parameter(torch.tensor(classifier.coef_).to(device))
    #         # if classifier.fit_intercept:
    #         #     linear.linear.bias = torch.nn.Parameter(
    #         #         torch.tensor(classifier.intercept_).to(device))
    #         linear.to(device)
    #
    #         optimizer = torch.optim.SGD(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    #
    #         for e in range(epochs):
    #             for b, (x, y) in enumerate(train_loader):
    #                 x = x.to(device)
    #                 y = y.to(device).long()
    #                 out = linear(x).float()
    #
    #                 # Loss function requires float values for the logits
    #                 loss_ce = loss_ce_fn(out, y)
    #                 loss_ce_path.append(loss_ce.item())
    #
    #                 loss_energy, loss_entropy = None, None  #this is only to not crash in the final print
    #                 if 'Energy' in method:
    #                     loss_energy = gamma_energy * loss_energy_fn(out, y)
    #                     loss_energy_path.append(loss_energy.item())
    #                     loss = loss_ce + loss_energy
    #                 elif 'Entropy' in method:
    #                     loss_entropy = gamma_entropy * loss_entropy_fn(out, y)
    #                     loss_entropy_path.append(loss_entropy.item())
    #                     loss = loss_ce + loss_entropy
    #                 else:
    #                     loss = loss_ce
    #
    #                 loss_tot_path.append(loss.item())
    #
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #
    #                 print(f"e [{e}/{epochs}], b: [{b}/{len(train_loader)}] -> "\
    #                       f"L_tot: {loss.item()} / "\
    #                       f"L_ce: {loss_ce.item()} / "\
    #                       f"L_energy: {loss_energy.item() if method == 'CE-Energy' else '-'} / "\
    #                       f"L_entropy: {loss_entropy.item() if method == 'CE-Entropy' else '-'}"
    #                       )
    #
    #         fig, ax = viz.create_figure(figsize=(7, 7), fontsize=15)
    #         alpha = .7
    #         ax.plot(loss_tot_path, label='tot', alpha=alpha)
    #         if 'Energy' in method:
    #             ax.plot(loss_ce_path, label='ce', alpha=alpha)
    #             ax.plot(loss_energy_path, label='energy', alpha=alpha)
    #         if 'Entropy' in method:
    #             ax.plot(loss_ce_path, label='ce', alpha=alpha)
    #             ax.plot(loss_entropy_path, label='entropy', alpha=alpha)
    #         ax.set_title(f'loss_path-{method}')
    #         ax.set_xlabel('iteration')
    #         ax.set_ylabel('Loss')
    #         fig.legend()
    #         fig.tight_layout()
    #         fig.show()
    #         fig.savefig(os.path.join(path, f'loss_path-{method}.pdf'))
    #
    #         torch.save(linear.to('cpu'), os.path.join(path, f"linear_retrained_{method}_loss.pt"))



    detectors_path = os.path.join(path, f"detectors.pkl")
    if overwrite_detectors or (not os.path.isfile(detectors_path)):
        # The detector that we want to test from the paper [1] - [6] - [26] (in addition to the MaxSoftMax baseline detector)
        print("Creating OOD Detectors")

        detectors = {}
        detectors["MaxSoftmax"] = MaxSoftmax(torch.load(os.path.join(path, 'linear_retrained_CE_loss.pt')))
        detectors["Entropy"] = Entropy(torch.load(os.path.join(path, 'linear_retrained_CE-Entropy_loss.pt')))
        detectors["EnergyBased"] = EnergyBased(torch.load(os.path.join(path, 'linear_retrained_CE-Energy_loss.pt')))
        detectors["OpenMax"] = OpenMax(torch.load(os.path.join(path, 'linear_retrained_CE_loss.pt')))

        print(f"> Fitting {len(detectors)} detectors")

        for name, detector in detectors.items():
            print(f"--> Fitting {name}")
            detector.fit(train_loader, device=device)
            # fm.my_save(detector, os.path.join(path, f"detector-{name}.pkl"))
        fm.my_save(detectors, os.path.join(path, f"detectors.pkl"))
    else:
        detectors = fm.my_load(detectors_path)

    import json
    from utils.evaluation import detector_scores

    datasets = {}

    results_list = []
    for i, loader in enumerate(test_loader_list):
      datasets["OOD test set "] = loader    # todo: this may be deleted
      results = detector_scores(detectors, datasets)
      results_list.append(results)

    # Saving the results in a json file
    fm.my_save(results_list, os.path.join(path, f'test_results.pkl'))
    # with open(f'test_results_{i}.json', 'w') as f:
    #     json.dump(results, f)

    fm.my_load(results_list, os.path.join(path, f'test_results.pkl'))
    print("")