import torch
from torch import nn
import utils.visualization as viz
from utils import data as ut_data
import numpy as np


class LinearSVM(nn.Module):

    def __init__(self, n_features,
                 C=1,
                 lr=1e-2,
                 bias=True,
                 class_weight=(1, 5),
                 optim='adam'):
        super().__init__()
        self.linear = nn.Linear(n_features, 2, bias=bias).double()
        self.threshold = 0.0
        self.C = C
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=C)
        # self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr)
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr, weight_decay=C)
        else:
            self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr, weight_decay=C)

        # self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.DoubleTensor(list(class_weight)))
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.DoubleTensor(list(class_weight)))
        self.loss_path = []
        self.loss_hinge_path = []
        self.loss_reg_path = []
        self.epoch = 0

    def forward(self, x):
        out = self.linear(x) - self.threshold
        # out = out.flatten()
        # return F.softmax(torch.cat((1-out, out), dim=1),dim=1)
        return out
        # return torch.cat((-out, out), dim=1)

    def fit(self, train_loader, epochs, device='cpu', lr=None):
        self.linear.train()
        # if lr is not None:
        #     self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr)

        for e in range(epochs):
            self.epoch += 1
            for b, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                # y = y.type(x.dtype)
                y = y.long()
                # y[y == 0] = -1

                out = self(x)
                # loss_hinge = torch.mean(torch.clamp(1 - y * out, min=0))
                loss_hinge = self.loss_fn(out, y)
                loss_reg = torch.norm(self.linear.weight, p=2) / 2.0
                loss = loss_hinge #+ self.C * loss_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                self.loss_path.append(loss.item())
                self.loss_hinge_path.append(loss_hinge.item())
                self.loss_reg_path.append(loss_reg.item())

                # print(f"e [{e}/{epochs}], b: [{b}/{len(train_loader)}] -> " \
                #       f"L_ce: {loss.item()} / "
                #       )
                print(f"e [{e}/{epochs}], b: [{b}/{len(train_loader)}] ->"
                      f"L: {loss.item()} / "
                      f"L_err: {loss_hinge.item()} / "
                      f"L_reg: {loss_reg.item()} / "
                      )

        self.linear.eval()

    def plot_loss_path(self, figsize=(7, 7), fontsize=15, ax=None, fig=None,
                       filename=None):

        if ax is None:
            fig, ax = viz.create_figure(figsize=figsize, fontsize=fontsize)
        alpha = .7
        # ax.plot(self.loss_path, alpha=alpha, label='tot')
        ax.plot(self.loss_hinge_path, alpha=alpha, label='hinge')
        ax.plot(self.loss_reg_path, alpha=alpha, label='reg')
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('Loss')
        return fig, ax

    def predict(self, test_loader):
        return

    # @torch.jit.export
    # def forward_explain(self, x, k):
    #     assert x.shape[0] == 1
    #     out = self.linear(x)
    #     w = self.linear.weight.data.squeeze()
    #     idx = torch.argsort(w, descending=True)
    #     x_sorted = x[:, idx].squeeze()
    #     nnz_idx = idx[x_sorted != 0]
    #     k = int(min(k.item(), nnz_idx.shape[0]))
    #     return torch.cat((-out, out), dim=1), nnz_idx[:k]

    def load_sklearn_pretrained(self, classifier, device='cpu'):
        self.linear.weight = torch.nn.Parameter(
            torch.tensor(np.vstack([-classifier.coef_, classifier.coef_]))).to(device)
        if classifier.fit_intercept:
            self.linear.bias = torch.nn.Parameter(
                torch.tensor([-classifier.intercept_[0], classifier.intercept_[0]]).to(device))

    def export_params_to_sklearn_clf(self):
        pass
