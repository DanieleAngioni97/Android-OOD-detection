import torch
from torch import nn
import utils.visualization as viz
from utils import data as ut_data


class LinearSVM(nn.Module):

    def __init__(self, n_features,
                 C=1,
                 lr=1e-2,
                 bias=True):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias).double()
        self.threshold = 0.0
        self.C = C
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=C)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_path = []

    def forward(self, x):
        out = self.linear(x) - self.threshold
        # return F.softmax(torch.cat((1-out, out), dim=1),dim=1)
        return out.flatten()    # TODO: valid only if BCEloss is used
        # return torch.cat((-out, out), dim=1)

    def fit(self, train_loader, epochs, device='cpu'):
        for e in range(epochs):
            for b, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.type(x.dtype).to(device)
                out = self(x)
                # Loss function requires float values for the logits
                loss = self.loss_fn(out, y)
                self.loss_path.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"e [{e}/{epochs}], b: [{b}/{len(train_loader)}] -> " \
                      f"L_ce: {loss.item()} / "
                      )

    def plot_loss_path(self, figsize=(7, 7), fontsize=15, ax=None, fig=None,
                       filename=None):

        if ax is None:
            fig, ax = viz.create_figure(figsize=figsize, fontsize=fontsize)
        alpha = .7
        ax.plot(self.loss_path, alpha=alpha)
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
        self.linear.weight = torch.nn.Parameter(torch.tensor(classifier.coef_).to(device))
        if classifier.fit_intercept:
            self.linear.bias = torch.nn.Parameter(
                torch.tensor(classifier.intercept_).to(device))

    def export_params_to_sklearn_clf(self):
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=1)

        print("")

        pass
