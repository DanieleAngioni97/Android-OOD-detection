import torch
from torch import nn


class LinearSVM(nn.Module):

    def __init__(self, n_features,bias=False):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        self.threshold = 0.0

    def forward(self, x):
        out = torch.sub(self.linear(x), self.threshold)
        # return F.softmax(torch.cat((1-out, out), dim=1),dim=1)
        # return out
        return torch.cat((-out, out), dim=1)

    @torch.jit.export
    def forward_explain(self, x, k):
        assert x.shape[0] == 1
        out = self.linear(x)
        w = self.linear.weight.data.squeeze()
        idx = torch.argsort(w, descending=True)
        x_sorted = x[:, idx].squeeze()
        nnz_idx = idx[x_sorted != 0]
        k = int(min(k.item(), nnz_idx.shape[0]))
        return torch.cat((-out, out), dim=1), nnz_idx[:k]

    def load_sklearn_pretrained(self, classifier, device='cpu'):
        self.linear.weight = torch.nn.Parameter(torch.tensor(classifier.coef_).to(device))
        if classifier.fit_intercept:
            self.linear.bias = torch.nn.Parameter(
                torch.tensor(classifier.intercept_).to(device))
