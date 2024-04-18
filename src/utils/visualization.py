import matplotlib as mpl
import matplotlib.pyplot as plt

def format_mpl(font_size: int = 30):
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['mathtext.fontset'] = 'stix'

def create_figure(nrows=1, ncols=1, figsize=(5, 5), squeeze=True, fontsize=30):
    format_mpl(fontsize)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figsize[1], nrows*figsize[0]), squeeze=squeeze)
    return fig, axs