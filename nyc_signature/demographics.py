#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Demographics Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nyc_signature.utils import ax_formatter, colors, size, save_fig


class Age:
    """
    Class to analyze age characteristics of 2016 US voters.

    :Attributes:

    - **columns**: *list* list of column names for age data
    - **data**: *DataFrame* US voter age data
    - **data_link**: *str* link to US Census web page containing the source \
        age data
    - **data_file**: *str* path to data file on disk
    - **data_types**: *tuple* data types for each column
    """
    def __init__(self):
        self.data_types = {
            'sex': str,
            'age_years': str,
            'us_population': np.int32,
            'us_citizen_population': np.int32,
            'registered_yes': np.int32,
            'registered_yes_pct': np.float64,
            'registered_no': np.int32,
            'registered_no_pct': np.float64,
            'registered_no_response': np.int32,
            'registered_no_response_pct': np.float64,
            'voted_yes': np.int32,
            'voted_yes_pct': np.float64,
            'voted_no': np.int32,
            'voted_no_pct': np.float64,
            'voted_no_response': np.int32,
            'voted_no_response_pct': np.float64,
            'total_registered_pct': np.float64,
            'total_voted_pct': np.float64,
        }
        self.data = None
        self.data_url = ('https://timothyhelton.github.io/assets/data/'
                         'us_age_2016_voter.xls')
        self.load_data()

    def __repr__(self):
        return f'Age()'

    def load_data(self):
        """
        Load data from file.
        """
        self.data = pd.read_excel(
            self.data_url,
            dtype=self.data_types,
            header=0,
            names=self.data_types.keys(),
            skip_footer=5,
            skiprows=5)

        self.data = (self.data
                     .drop(self.data.iloc[:, list(range(5, 16, 2)) + [16, 17]],
                           axis=1))
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
        self.data['voted_yes_pct'] = (self.data
                                      .voted_yes
                                      .div(self.data.us_population))

    def age_vote_plot(self, save=False):
        """
        US Voter Age plots

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Voter Age',
                         figsize=(12, 10), facecolor='white',
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
         .plot(kind='area', alpha=0.5, label='Voters', ax=ax0))

        target_group = both_sexes.voted_yes.copy()
        target_group.loc[target_group < target_group.quantile(0.7)] = 0
        (target_group
         .plot(kind='area', alpha=0.2, color=colors[2], label='Target Group',
               ax=ax0))

        ax0.set_title('Voters vs Age', fontsize=size['title'])
        ax0.legend(loc='upper left', fontsize=size['legend'])
        ax0.set_ylabel('People ($thousands$)', fontsize=size['label'])
        ax0.yaxis.set_major_formatter(ax_formatter['thousands'])

        # Voters Gender Plot
        for n, gender in enumerate((female, male)):
            (gender
             .voted_yes
             .plot(kind='area', alpha=0.5, color=colors[n], ax=ax1))
        ax1.set_title('Voters vs Age by Gender', fontsize=size['title'])

        # Voters All US Percentage Plot
        (both_sexes
         .voted_yes
         .div(both_sexes.us_population)
         .plot(kind='area', alpha=0.5, label='Voter Percentage', ax=ax2))

        target_group = both_sexes.voted_yes_pct.copy()
        target_group.loc[target_group < 0.5] = 0
        (target_group
         .plot(kind='area', alpha=0.2, color=colors[2], label='Target Group',
               ax=ax2))

        ax2.set_title('Percent Voters vs Age', fontsize=size['title'])
        ax2.legend(loc='upper left', fontsize=size['legend'])
        ax2.set_ylabel('Voting Percentage of Age', fontsize=size['label'])
        ax2.yaxis.set_major_formatter(ax_formatter['percent_convert'])

        # Voters Gender Percentage Plot
        for n, gender in enumerate((female, male)):
            (gender
             .voted_yes
             .div(gender.us_population)
             .plot(kind='area', alpha=0.5, color=colors[n], ax=ax3))
        ax3.set_title('Percent Voters vs Age by Gender',
                      fontsize=size['title'])

        for ax in (ax0, ax1):
            ax.set_xlabel('')

        for ax in (ax2, ax3):
            ax.set_xlabel('Age ($years$)', fontsize=size['label'])

        for ax in (ax1, ax3):
            ax.legend(['Female', 'Male'], fontsize=size['legend'])

        plt.tight_layout()
        plt.suptitle('2016 US Voter Age Distributions',
                     fontsize=size['super_title'], y=1.03)

        save_fig('age_voted', save)


class NewYork:
    """
    Class to analyze age characteristics of 2016 New York state voters.

    :Attributes:

    - **columns**: *list* list of column names for age data
    - **data**: *DataFrame* New York state voter data
    - **data_link**: *str* link to US Census web page containing the source \
        age data
    - **data_file**: *str* path to data file on disk
    - **data_types**: *tuple* data types for each column
    """
    def __init__(self):
        self.data_types = {
            'state': str,
            'race': str,
            'state_population': np.int32,
            'state_citizen_population': np.int32,
            'registered_yes': np.int32,
            'registered_total_pct': np.float64,
            'registered_total_error': np.float64,
            'registered_citizen_pct': np.float64,
            'registered_citizen_error': np.float64,
            'voted_yes': np.int32,
            'voted_total_pct': np.float64,
            'voted_total_error': np.float64,
            'voted_citizen_pct': np.float64,
            'voted_citizen_error': np.float64,
        }
        self.data = None
        self.data_url = ('https://timothyhelton.github.io/assets/data/'
                         'states_2016_voter.xls')
        self.load_data()

    def load_data(self):
        """
        Load data from file.
        """
        self.data = pd.read_excel(
            self.data_url,
            dtype=self.data_types,
            header=0,
            names=self.data_types.keys(),
            skip_footer=5,
            skiprows=5)

        self.data.loc[:, 'state'] = (self.data.loc[:, 'state']
                                     .fillna(method='ffill')
                                     .str.lower())
        for col in ('state', 'race'):
            self.data.loc[:, col] = (self.data.loc[:, col]
                                     .astype('category'))

        thousands_cols = (
            'state_population',
            'state_citizen_population',
            'registered_yes',
            'voted_yes',
        )
        self.data.loc[:, thousands_cols] = (self.data.loc[:, thousands_cols]
                                            * 1000)
        self.data = self.data.set_index('state')

    def ethnicity_plot(self, save=False):
        """
        New York state ethnicity voter plots.

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Age Vote',
                         figsize=(10, 15), facecolor='white',
                         edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))

        # Voter Turnout
        (self.data
         .loc['new york', ['race',
                           'state_population',
                           'registered_yes',
                           'voted_yes']]
         .set_index('race')
         .sort_values(by='voted_yes')
         .plot(kind='bar', alpha=0.5, edgecolor='black', ax=ax0))

        ax0.set_title('2016 New York Voters', fontsize=size['title'])
        ax0.legend(['Total Population', 'Registered Voters', 'Voted in 2016'],
                   fontsize=size['legend'])
        ax0.set_ylabel('People ($thousands$)', fontsize=size['label'])
        ax0.yaxis.set_major_formatter(ax_formatter['thousands'])

        # Voter Turnout Percentage
        voter_pct = (self.data
                     .loc['new york', ['race',
                                       'registered_total_pct',
                                       'registered_total_error',
                                       'voted_total_pct',
                                       'voted_total_error']]
                     .set_index('race')
                     .applymap(lambda x: float(x))
                     .sort_values(by='voted_total_pct'))

        data_cols = ['registered_total_pct', 'voted_total_pct']
        voter_err = voter_pct.drop(data_cols, axis=1)
        voter_err.columns = data_cols
        error_kw = {'capsize': 3, 'capthick': 1, 'elinewidth': 1}

        voter_pct.plot(kind='bar', y=data_cols, alpha=0.5, edgecolor='black',
                       error_kw=error_kw, yerr=voter_err, ax=ax1)

        ax1.set_title('2016 New York Voter Percentages',
                      fontsize=size['title'])
        ax1.legend(['Registered Voters', 'Voted in 2016'],
                   fontsize=size['legend'])
        ax1.set_ylabel('Percentage', fontsize=size['label'])
        ax1.yaxis.set_major_formatter(ax_formatter['percent'])

        for ax in (ax0, ax1):
            ax.set_xlabel('Group', fontsize=size['label'])
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)

        plt.tight_layout()
        plt.suptitle('2016 New York State Voter Distributions',
                     fontsize=size['super_title'], y=1.03)

        save_fig('age_voted', save)
