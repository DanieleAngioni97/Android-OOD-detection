import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn

import utils


def format_mpl(font_size: int = 30):
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['mathtext.fontset'] = 'stix'


def create_figure(nrows=1, ncols=1, figsize=(5, 5), squeeze=True, fontsize=30):
    format_mpl(fontsize)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * figsize[1], nrows * figsize[0]), squeeze=squeeze)
    return fig, axs


def evaluate_and_plot_metrics(clf, X_tests, y_tests, ax=None,
                              title='', ylabel=''):
    from sklearn.metrics import precision_recall_fscore_support

    LINESTYLE_DICT = {'gw': 'dashed', 'mw': 'solid'}
    COLOR_DICT = {'prec': 'orange', 'rec': 'red', 'f1': 'blue'}
    MARKER_DICT = {'prec': '^', 'rec': 'v', 'f1': 'D'}

    if ax is None:
        fig, ax = create_figure(fontsize=15)

    for j, class_name in enumerate(('gw', 'mw')):
        f1_list, prec_list, rec_list = [], [], []
        for i, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
            if not isinstance(clf, torch.nn.Module):
                y_pred = clf.predict(X_test)
            else:
                X_test = utils.data.sparse_coo_to_tensor(X_test.tocoo())
                y_pred = (clf(X_test).detach().numpy() > 0).astype(float)
            # pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            prec_list.append(pr[j])
            rec_list.append(rec[j])
            f1_list.append(f1[j])

        common_args = {'alpha': .7}
        ax.plot(prec_list, label=f'precision ({class_name})',
                color=COLOR_DICT['prec'],
                marker=MARKER_DICT['prec'],
                linestyle=LINESTYLE_DICT[class_name],
                **common_args)
        ax.plot(rec_list, label=f'recall ({class_name})',
                color=COLOR_DICT['rec'],
                marker=MARKER_DICT['prec'],
                linestyle=LINESTYLE_DICT[class_name],
                **common_args)
        ax.plot(f1_list, label=f'F1-score ({class_name})',
                color=COLOR_DICT['f1'],
                marker=MARKER_DICT['prec'],
                linestyle=LINESTYLE_DICT[class_name],
                **common_args)
    ax.set_xlabel('month')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis='y')

    # fig.tight_layout()
    # fig.show()
