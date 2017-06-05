#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Demographics Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import os
import os.path as osp

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


thousands_formatter = FuncFormatter(lambda x, position: f'{x * 1e-3}')
colors = sns.color_palette()
label_size = 14
title_size = 20
super_title_size = 24


def save_fig(name=None, save=False):
    """
    Helper function to save or display figure.

    :param str name: file name
    :param bool save: if True the figure will be saved
    """
    if save:
        plt.savefig(f'{name}.png', bbox_inches='tight',
                    bbox_extra_artists=[super_title_size])
    else:
        plt.show()


class Age:
    """
    Class to analyze age characteristics of voters.

    """
    def __init__(self):
        self.columns = [
            'sex',
            'age_years',
            'us_population',
            'us_citizen_population',
            'registered_yes',
            'registered_no',
            'registered_no_response',
            'voted_yes',
            'voted_no',
            'voted_no_response',
        ]
        self.data = None
        self.data_link = ('https://www2.census.gov/programs-surveys/cps'
                          '/tables/p20/580/table01.xls')
        self.data_file = osp.join('us_age_2016_voter.xls')

        self.data_types = ([str] * 2
                           + [np.int32] * 2
                           + [np.int32, np.float64] * 6
                           + [np.float64] * 2)

    def __repr__(self):
        return f'Age()'

    def load_data(self):
        """
        Load data from file.
        """
        current_dir = osp.dirname(osp.realpath(__file__))
        age_data = osp.realpath(
            osp.join(current_dir, '..', 'data', self.data_file))

        self.data = pd.read_excel(
            age_data,
            dtype={x: y for x, y in enumerate(self.data_types)},
            header=None,
            skip_footer=5,
            skiprows=6)

        self.data = (self.data
                     .drop(self.data.loc[:, list(range(5, 16, 2)) + [16, 17]],
                           axis=1))
        self.data.columns = self.columns
        self.data.loc[self.data.sex == 'nan', 'sex'] = np.nan
        self.data.loc[:, 'sex'] = (self.data.loc[:, 'sex']
                                   .fillna(method='ffill')
                                   .str.lower())
        self.data.loc[:, 'sex'] = (self.data.loc[:, 'sex']
                                   .astype('category'))
        self.data.loc[:, 'age_years'] = (self.data.loc[:, 'age_years']
                                         .str.replace(' years', ''))
        self.data = self.data[self.data.loc[:, 'age_years'].str.isnumeric()]
        self.data.iloc[:, 2:] = self.data.iloc[:, 2:] * 1000
        self.data = self.data.set_index('age_years')

    def age_vote_plot(self, save=False):
        """
        Age Kernel Density Estimate plot.

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Age Vote',
                         figsize=(10, 15), facecolor='white',
                         edgecolor='black')
        rows, cols = (3, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (2, 0))

        for n, vote in enumerate(('voted_yes', 'voted_no')):
            (self.data.query('sex == "both sexes"')[vote]
             .plot(kind='area', alpha=0.5, ax=ax0))
        ax0.legend(['Voted', 'Did Not Vote'])
        ax0.set_title('Voters vs Age', fontsize=title_size)
        ax0.set_xlabel('Age ($years$)', fontsize=label_size)
        ax0.set_ylabel('People ($thousands$)', fontsize=label_size)
        ax0.yaxis.set_major_formatter(thousands_formatter)

        plt.suptitle('Voter Age Distributions',
                     fontsize=super_title_size, y=1.03)
        plt.tight_layout()

        save_fig('age_voted', save)
