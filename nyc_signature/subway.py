#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Subway Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from collections import OrderedDict
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from nyc_signature.utils import ax_formatter, colors, size, save_fig


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
            'route_8': np.int32,
            'route_9': np.int32,
            'route_10': np.int32,
            'route_11': np.int32,
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
                                dtype=self.data_types)
        self.data.columns = list(self.data_types.keys())
        self.trains = pd.melt(self.data,
                              id_vars=['latitude', 'longitude'],
                              value_vars=[f'route_{x}' for x in range(1, 12)],
                              var_name='train',
                              value_name='line')

    # TODO format plot
    def train_plot(self, save=False):
        """
        Plot stations by train.
        """
        sns.lmplot(x='latitude', y='longitude', data=self.trains, hue='line',
                   fit_reg=False, size=10)
        save_fig('stations_train', save)
