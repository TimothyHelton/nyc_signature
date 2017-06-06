#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Demographics Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import os.path as osp

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


percent_formatter = FuncFormatter(lambda x, position: f'{x * 100:.1f}%')
thousands_formatter = FuncFormatter(lambda x, position: f'{x * 1e-3:.0f}')
sns.set_palette('rainbow')
colors = sns.color_palette()
label_size = 14
legend_size = 12
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

    :Attributes:

    - **columns**: *list* list of column names for age data
    - **data**: *DataFrame* age data
    - **data_link**: *str* link to US Census web page containing the source \
        age data
    - **data_file**: *str* path to data file on disk
    - **data_types**: *tuple* data types for each column

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
                         figsize=(10, 10), facecolor='white',
                         edgecolor='black')
        rows, cols = (2, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1), sharex=ax0, sharey=ax0)
        ax2 = plt.subplot2grid((rows, cols), (1, 0), sharex=ax0)
        ax3 = plt.subplot2grid((rows, cols), (1, 1), sharex=ax0, sharey=ax2)

        both_sexes = self.data.query('sex == "both sexes"')
        female = self.data.query('sex == "female"')
        male = self.data.query('sex == "male"')

        # Voters All US Plot
        (both_sexes
         .voted_yes
         .plot(kind='area', alpha=0.5, ax=ax0))
        ax0.set_title('Voters vs Age', fontsize=title_size)
        ax0.set_ylabel('People ($thousands$)', fontsize=label_size)
        ax0.yaxis.set_major_formatter(thousands_formatter)

        # Voters Gender Plot
        for n, gender in enumerate((female, male)):
            (gender
             .voted_yes
             .plot(kind='area', alpha=0.5, color=colors[-n], ax=ax1))
        ax1.set_title('Voters vs Age by Gender', fontsize=title_size)

        # Voters All US Percentage Plot
        (both_sexes
         .voted_yes
         .div(both_sexes.us_population)
         .plot(kind='area', alpha=0.5, ax=ax2))
        ax2.set_title('Percent Voters vs Age', fontsize=title_size)
        ax2.set_ylabel('Voting Percentage of Age', fontsize=label_size)
        ax2.yaxis.set_major_formatter(percent_formatter)

        # Voters Gender Percentage Plot
        for n, gender in enumerate((female, male)):
            (gender
             .voted_yes
             .div(gender.us_population)
             .plot(kind='area', alpha=0.5, color=colors[-n], ax=ax3))
        ax3.set_title('Percent Voters vs Age by Gender',
                      fontsize=title_size)

        for ax in (ax0, ax1):
            ax.set_xlabel('')

        for ax in (ax2, ax3):
            ax.set_xlabel('Age ($years$)', fontsize=label_size)

        for ax in (ax1, ax3):
            ax.legend(['Female', 'Male'], fontsize=legend_size)

        plt.suptitle('US Voter Age Distributions',
                     fontsize=super_title_size, y=1.03)
        plt.tight_layout()

        save_fig('age_voted', save)
