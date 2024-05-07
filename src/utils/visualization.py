import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn

import utils.evaluation as ut_eval
import math
import numpy as np


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


def plot_metrics(y_preds, y_tests, ax=None,
                              title='', ylabel='',
                 plot_rec=True, plot_prec=True, plot_f1=True, plot_gw=True):
    from sklearn.metrics import precision_recall_fscore_support

    LINESTYLE_DICT = {'gw': 'dashed', 'mw': 'solid'}
    COLOR_DICT = {'prec': 'orange', 'rec': 'red', 'f1': 'blue'}
    MARKER_DICT = {'prec': '^', 'rec': 'v', 'f1': 'D'}

    if ax is None:
        fig, ax = create_figure(fontsize=15)

    for j, class_name in enumerate(('gw', 'mw')):
        if (class_name == 'gw') and not plot_gw:
            continue
        f1_list, prec_list, rec_list = [], [], []
        for i, (y_pred, y_test) in enumerate(zip(y_preds, y_tests)):
            # pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            try:
                pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
                prec_list.append(pr[j])
                rec_list.append(rec[j])
                f1_list.append(f1[j])
            except:
                prec_list.append(math.nan)
                rec_list.append(math.nan)
                f1_list.append(math.nan)


        common_args = {'alpha': .7}
        if plot_prec:
            ax.plot(prec_list, label=f'precision ({class_name})',
                    color=COLOR_DICT['prec'],
                    marker=MARKER_DICT['prec'],
                    linestyle=LINESTYLE_DICT[class_name],
                    **common_args)
        if plot_rec:
            ax.plot(rec_list, label=f'recall ({class_name})',
                    color=COLOR_DICT['rec'],
                    marker=MARKER_DICT['prec'],
                    linestyle=LINESTYLE_DICT[class_name],
                    **common_args)
        if plot_f1:
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

def evaluate_and_plot_metrics(clf, X_tests, y_tests, ax=None,
                              title='', ylabel=''):
    y_preds = ut_eval.get_predictions(clf, X_tests)
    plot_metrics(y_preds, y_tests, ax=ax, title=title, ylabel=ylabel)

def plot_dataset_stats(y_tests, ax=None):
    if ax is None:
        fig, ax = create_figure(fontsize=15, figsize=(5, 10))
    n_samples = [y_test.size for y_test in y_tests]
    n_mw = [(y_test == 1).sum() for y_test in y_tests]
    n_gw = [(y_test == 0).sum() for y_test in y_tests]
    ax.plot(n_mw, label='# malware', marker='^', color='blue')
    ax.plot(n_gw, label='# goodware', marker='v', color='red')
    ax.legend()
    # dates = [f"{t_test[0].year}-{t_test[0].month}" for t_test in t_tests]
    # ax.set_xticklabels(dates, rotation=50)
    #
    # fig.tight_layout()
    # fig.show()
    # fig.savefig(file_path)


def plot_weight_distribution(clf, ax=None):
    if ax is None:
        fig, ax = create_figure(fontsize=15, figsize=(5, 10))
    w_sorted = np.sort(np.abs(clf.coef_.flatten()))[::-1]
    ax.plot(w_sorted)
    ax.set_ylabel('|w|')
    ax.set_xlabel('# w')
    ax.set_xscale('log')



