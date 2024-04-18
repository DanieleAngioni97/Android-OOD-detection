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


if __name__ == '__main__':
    # data_id = "1uSyoNCCfUNVvvOxPP4ga25uIMt8unX2T"
    # data_folder = "data/"
    #
    # gdown.download(id=data_id, output="data.zip")
    # zip_file_path = "data.zip"
    #
    # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #     # Extract all the contents to the specified directory
    #     zip_ref.extractall(data_folder)
    #

    ##############################################################################################################
    # Loading training and testing features
    ##############################################################################################################

    path = "../data/datasets_and_clf-10k/"
    clf_path = os.path.join(path, "clf.pkl")
    method_train_list = ['CE', 'CE-Energy', 'Ce-Entropy']
    # method_train_list = ['CE-Entropy']


    test_only = True
    overwrite_detectors = False
    train_features = fm.my_load(os.path.join(path, "train.pkl"))
    feature_names = fm.my_load(os.path.join(path, "feature_names.pkl"))

    """
    test_features_1 = pickle.load(open(path+"test-1.pkl", 'rb'))
    test_features_2 = pickle.load(open(path+"test-2.pkl", 'rb'))
    test_features_3 = pickle.load(open(path+"test-3.pkl", 'rb'))
    test_features_4 = pickle.load(open(path+"test-4.pkl", 'rb'))
    """
    test_features_w = fm.my_load(os.path.join(path, "tests_monthly.pkl"))

    n_test_months = len(test_features_w[0])
    n_months_per_test = 12
    n_slots = n_test_months // n_months_per_test

    test_set_list = []
    X_tests, y_tests, t_tests = test_features_w
    for i in range(n_slots):
        start = i * n_months_per_test
        end = (i + 1) * n_months_per_test
        test_set_i = {}
        test_set_i['X'] = vstack(X_tests[start:end])
        test_set_i['y'] = np.concatenate(y_tests[start:end])
        test_set_i['t'] = np.concatenate(t_tests[start:end])
        test_set_list.append(test_set_i)

    # # todo: questo non ha senso, non capisco cosa hanno fatto
    # test_features_1 = [test_feature[0:2] for test_feature in test_features_w]
    # test_features_1[0] = vstack(test_features_1[0])
    # test_features_1[1] = np.concatenate(test_features_1[1])
    # test_features_1[2] = np.concatenate(test_features_1[2])
    #
    # test_features_2 = [test_feature[18:20] for test_feature in test_features_w]
    # test_features_2[0] = vstack(test_features_2[0])
    # test_features_2[1] = np.concatenate(test_features_2[1])
    # test_features_2[2] = np.concatenate(test_features_2[2])
    #
    # test_features_3 = [test_feature[20:22] for test_feature in test_features_w]
    # test_features_3[0] = vstack(test_features_3[0])
    # test_features_3[1] = np.concatenate(test_features_3[1])
    # test_features_3[2] = np.concatenate(test_features_3[2])
    #
    # test_features_4 = [test_feature[22:24] for test_feature in test_features_w]
    # test_features_4[0] = vstack(test_features_4[0])
    # test_features_4[1] = np.concatenate(test_features_4[1])
    # test_features_4[2] = np.concatenate(test_features_4[2])
    #
    # print("")


    ##############################################################################################################
    # Creating our dataloaders
    ##############################################################################################################

    """#train data loader
    print(type(train_features[0]))
    print(type(train_features[1]))
    if type(train_features[0]) == coo_matrix:
      print("bhe")
    """


    b_size = 200
    #train data loader

    # apply a feature selection method (what method?)
    X_train, y_train, t_train = train_features
    print(f"X_train shape: {X_train.shape}")
    n_train_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_train_features = (X_train.sum(axis=0) != 0).sum() # features that appear only in the train set
    n_all_zero_train_samples = (X_train.sum(axis=1) != 0).sum()     # samples with all zero features (???)
    max_n_samples = n_train_samples
    sparse_dataset = ut_data.SparseDataset(X_train[:max_n_samples], y_train[:max_n_samples])
    train_loader = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                             collate_fn=ut_data.dense_batch_collate)

    test_loader_list = []
    for test_set_i in test_set_list:
        sparse_dataset = ut_data.SparseDataset(test_set_i['X'][:max_n_samples],
                                       test_set_i['y'][:max_n_samples])
        test_loader_i = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
                                 collate_fn=ut_data.dense_batch_collate)
        test_loader_list.append(test_loader_i)


    # sparse_dataset = ut_data.SparseDataset(test_features_2[0][:max_n_samples],
    #                                test_features_2[1][:max_n_samples])
    # test_loader_2 = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
    #                          collate_fn=ut_data.dense_batch_collate)
    #
    # sparse_dataset = ut_data.SparseDataset(test_features_3[0][:s],
    #                                test_features_3[1][:s])
    # test_loader_3 = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
    #                          collate_fn=ut_data.dense_batch_collate)
    #
    # sparse_dataset = ut_data.SparseDataset(test_features_4[0][:s],
    #                                test_features_4[1][:s])
    # test_loader_4 = DataLoader(sparse_dataset, batch_size=b_size, shuffle=True,
    #                          collate_fn=ut_data.dense_batch_collate)


    ##############################################################################################################
    # Loading the pretrained SVM model
    ##############################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = pickle.load(open(clf_path, 'rb'))
    classifier.n_features_in_ = X_train.shape[1]


    ##############################################################################################################
    # Training Loop
    ##############################################################################################################
    epochs = 1
    gamma_energy = 1
    gamma_entropy = 1

    linear_pretrained = LinearSVM(n_features=n_features,
                                  bias=classifier.fit_intercept)
    linear_pretrained.load_sklearn_pretrained(classifier)
    torch.save(linear_pretrained.to('cpu'), os.path.join(path, f"linear_svm_pretrained.pt"))

    if not test_only:
        for idx, method in enumerate(method_train_list):
            print(f"### Training with {method} loss ###")
            loss_ce_fn = CrossEntropyLoss()
            loss_energy_fn = EnergyRegularizedLoss()
            loss_entropy_fn = EntropicOpenSetLoss()

            loss_tot_path = []
            loss_ce_path = []
            loss_energy_path = []
            loss_entropy_path = []

            # Baseline Linear Layer
            linear = copy.deepcopy(linear_pretrained)
            # linear.linear.weight = torch.nn.Parameter(torch.tensor(classifier.coef_).to(device))
            # if classifier.fit_intercept:
            #     linear.linear.bias = torch.nn.Parameter(
            #         torch.tensor(classifier.intercept_).to(device))
            linear.to(device)

            optimizer = torch.optim.SGD(linear.parameters(), lr=1e-2, weight_decay=1e-4)

            for e in range(epochs):
                for b, (x, y) in enumerate(train_loader):
                    x = x.to(device)
                    y = y.to(device).long()
                    out = linear(x).float()

                    # Loss function requires float values for the logits
                    loss_ce = loss_ce_fn(out, y)
                    loss_ce_path.append(loss_ce.item())

                    loss_energy, loss_entropy = None, None  #this is only to not crash in the final print
                    if 'Energy' in method:
                        loss_energy = gamma_energy * loss_energy_fn(out, y)
                        loss_energy_path.append(loss_energy.item())
                        loss = loss_ce + loss_energy
                    elif 'Entropy' in method:
                        loss_entropy = gamma_entropy * loss_entropy_fn(out, y)
                        loss_entropy_path.append(loss_entropy.item())
                        loss = loss_ce + loss_entropy
                    else:
                        loss = loss_ce

                    loss_tot_path.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f"e [{e}/{epochs}], b: [{b}/{len(train_loader)}] -> "\
                          f"L_tot: {loss.item()} / "\
                          f"L_ce: {loss_ce.item()} / "\
                          f"L_energy: {loss_energy.item() if method == 'CE-Energy' else '-'} / "\
                          f"L_entropy: {loss_entropy.item() if method == 'CE-Entropy' else '-'}"
                          )

            fig, ax = viz.create_figure(figsize=(7, 7), fontsize=15)
            alpha = .7
            ax.plot(loss_tot_path, label='tot', alpha=alpha)
            if 'Energy' in method:
                ax.plot(loss_ce_path, label='ce', alpha=alpha)
                ax.plot(loss_energy_path, label='energy', alpha=alpha)
            if 'Entropy' in method:
                ax.plot(loss_ce_path, label='ce', alpha=alpha)
                ax.plot(loss_entropy_path, label='entropy', alpha=alpha)
            ax.set_title(f'loss_path-{method}')
            ax.set_xlabel('iteration')
            ax.set_ylabel('Loss')
            fig.legend()
            fig.tight_layout()
            fig.show()
            fig.savefig(os.path.join(path, f'loss_path-{method}.pdf'))

            torch.save(linear.to('cpu'), os.path.join(path, f"linear_retrained_{method}_loss.pt"))



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


    print("")