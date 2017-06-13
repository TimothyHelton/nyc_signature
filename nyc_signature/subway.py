#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Subway Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from collections import OrderedDict

from bokeh import io as bkio
from bokeh import models as bkm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from . import keys
from nyc_signature import locations
from nyc_signature.utils import size, save_fig


class Stations:
    """
    Class to describe New York city subway stations.
    """
    def __init__(self):
        self.data = None
        self.data_url = ('https://timothyhelton.github.io/assets/data/'
                         'nyc_subway_locations.csv')
        self.data_types = OrderedDict({
            'division': str,
            'line': str,
            'name': str,
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
        self.load_data()

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
            'name',
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
        self.trains.loc[:, 'train'] = (self.trains.loc[:, 'train']
                                       .str.strip())
        for col in ('route', 'train'):
            self.trains.loc[:, col] = (self.trains.loc[:, col]
                                       .astype('category'))

    def train_plot(self, save=False):
        """
        Plot subway stations by train.

        :param bool save: if True the figure will be saved
        """
        sns.lmplot(x='longitude', y='latitude',
                   data=(self.trains
                         .sort_values(by='train')),
                   hue='train', fit_reg=False, legend=False, markers='d',
                   scatter_kws={'alpha': 0.3}, size=10)

        legend = plt.legend(bbox_to_anchor=(1.10, 0.5), fancybox=True,
                            fontsize=size['legend'], loc='center right',
                            shadow=True, title='Train')
        plt.setp(legend.get_title(), fontsize=size['label'])
        plt.xlabel('Longitude', fontsize=size['label'])
        plt.ylabel('Latitude', fontsize=size['label'])

        plt.tight_layout()
        plt.suptitle('New York Subway Train Stations',
                     fontsize=size['super_title'], y=1.03)

        save_fig('stations_trains', save)

    def train_locations_plot(self):
        """
        Plot subway stations and interest locations.

        .. warning:: This method requires a Google API Key
        """
        map_options = {
            'lat': 40.70,
            'lng': -73.92,
            'map_type': 'roadmap',
            'zoom': 10,
        }
        plot = bkm.GMapPlot(
            api_key=keys.GOOGLE_API_KEY,
            x_range=bkm.Range1d(),
            y_range=bkm.Range1d(),
            map_options=bkm.GMapOptions(**map_options),
            plot_width=400,
            plot_height=600,
        )
        plot.title.text = 'New York City Hospitals and Subway Stations'

        hospital = bkm.Circle(
            x='longitude',
            y='latitude',
            fill_alpha=0.8,
            fill_color='#cd5b1b',
            line_color=None,
            size=14,
        )

        subway = bkm.Diamond(
            x='longitude',
            y='latitude',
            fill_color='#3062C8',
            line_color=None,
            size=10,
        )

        hosp = locations.Hospitals()
        hospitals = bkm.sources.ColumnDataSource(hosp.hospitals)
        plot.add_glyph(hospitals, hospital)

        subway_stations = bkm.sources.ColumnDataSource(
            data=(self.data
                  .loc[:, ['name', 'latitude', 'longitude']]
                  .drop_duplicates())
        )
        plot.add_glyph(subway_stations, subway)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Location', '@name'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.show(plot)
