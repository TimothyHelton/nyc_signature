#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Subway Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from nyc_signature.utils import size, save_fig


class Stations:
    """
    Class to describe New York city subway stations.
    """
    def __init__(self):
        self.data = None
        self.data_url = 'https://data.ny.gov/api/views/i9wp-a4ja/rows.csv'
        self.data_types = OrderedDict({
            'division': str,
            'line': str,
            'station_name': str,
            'latitude': np.float64,
            'longitude': np.float64,
            'route_1': str,
            'route_2': str,
            'route_3': str,
            'route_4': str,
            'route_5': str,
            'route_6': str,
            'route_7': str,
            'route_8': str,
            'route_9': str,
            'route_10': str,
            'route_11': str,
            'entrance_type': str,
            'entry': str,
            'exit_only': str,
            'vending': str,
            'staffing': str,
            'staff_hours': str,
            'ada': str,
            'ada_notes': str,
            'free_crossover': str,
            'north_south_street': str,
            'east_west_street': str,
            'corner': str,
            'entrance_latitude': np.float64,
            'entrance_longitude': np.float64,
            'station_location': str,
            'entrance_location': str,
        })
        self.trains = None

    def __repr__(self):
        return f'Stations()'

    def load_data(self):
        """
        Load data from file.
        """
        self.data = pd.read_csv(self.data_url,
                                dtype=self.data_types,
                                header=0,
                                names=self.data_types.keys())

        not_categories_cols = (
            'station_name',
            'latitude',
            'longitude',
            'north_south_street',
            'east_west_street',
            'entrance_latitude',
            'entrance_longitude',
            'station_location',
            'entrance_location',
        )
        categories_cols = [x for x in self.data_types.keys()
                           if x not in not_categories_cols]
        for col in categories_cols:
            self.data.loc[:, col] = (self.data.loc[:, col]
                                     .astype('category'))

        self.trains = pd.melt(self.data,
                              id_vars=['latitude', 'longitude'],
                              value_vars=[f'route_{x}' for x in range(1, 12)],
                              var_name='route',
                              value_name='train')
        for col in ('route', 'train'):
            self.trains.loc[:, col] = (self.trains.loc[:, col]
                                       .astype('category'))

    def train_plot(self, save=False):
        """
        Plot stations by train.
        """
        sns.lmplot(x='latitude', y='longitude',
                   data=(self.trains
                         .sort_values(by='train')),
                   hue='train', fit_reg=False, legend=False, markers='d',
                   size=10)
        legend = plt.legend(bbox_to_anchor=(1.10, 0.5), fancybox=True,
                            fontsize=size['legend'], loc='center right',
                            shadow=True, title='Train')
        plt.setp(legend.get_title(), fontsize=size['label'])
        plt.xlabel('Latitude', fontsize=size['label'])
        plt.ylabel('Longitude', fontsize=size['label'])

        plt.tight_layout()
        plt.suptitle('New York Subway Train Stations',
                     fontsize=size['super_title'], y=1.03)

        save_fig('stations_trains', save)
