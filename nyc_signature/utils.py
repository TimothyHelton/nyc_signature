#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Utilities Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import os.path as osp

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


ax_formatter = {
    'percent_convert': FuncFormatter(lambda x, position: f'{x * 100:.1f}%'),
    'percent': FuncFormatter(lambda x, position: f'{x:.0f}%'),
    'thousands': FuncFormatter(lambda x, position: f'{x * 1e-3:.0f}'),
}

colors = sns.color_palette()

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data'))

size = {
    'label': 14,
    'legend': 12,
    'title': 20,
    'super_title': 24,
}


def save_fig(name=None, save=False):
    """
    Helper function to save or display figure.

    :param str name: file name
    :param bool save: if True the figure will be saved
    """
    if save:
        plt.savefig(f'{name}.png', bbox_inches='tight',
                    bbox_extra_artists=[size['super_title']])
    else:
        plt.show()
