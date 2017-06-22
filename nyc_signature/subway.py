#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Subway Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import io
import re

from bokeh import io as bkio
from bokeh import models as bkm
import geopy.distance as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import seaborn as sns

try:
    from nyc_signature import keys
except ModuleNotFoundError:
    print('A Google API Key is required to generate the geographic images.')
    print('Upon instancing the Stations class please assign your key to the '
          'api_key attribute.')
from nyc_signature import locations
from nyc_signature.utils import ax_formatter, size, save_fig


class Stations:
    """
    Class to describe New York city subway stations.

    .. _`Google API Key`: https://developers.google.com/maps/documentation/
        javascript/get-api-key

    .. note:: A `Google API Key`_ is required to create the geographic plots.

    :Attributes:

    - **api_key**: *str* Google API Key
    - **data**: *DataFrame* New York City subway station data
    - **data_types**: *dict* data types for each column
    - **data_url**: *str* link to web page containing the source data
    - **hosp**: *Hospitals* instance of locations.Hospitals class
    - **hosp_dist**: *DataFrame* distance from hospitals to subway stations
    - **hosp_prox**: *DataFrame** hospital and subway stations in close \
        proximity
    - **hosp_stations**: *DataFrame* subway stations closest to hospitals
    - **map_glyph_hospital**: *Circle* map glyph for hospital points
    - **map_glyph_subway**: *Diamond* map glyph for subway points
    - **map_options**: *dict* Bokeh plot map options
    - **map_plot**: *GMapPlot* Bokeh Google Map Plot object
    - **trains**: *DataFrame* New York City subway train data
    """
    def __init__(self):
        try:
            self.api_key = keys.GOOGLE_API_KEY
        except NameError:
            self.api_key = None
        self.data = None
        self.data_url = ('https://timothyhelton.github.io/assets/data/'
                         'nyc_subway_locations.csv')
        self.data_types = {
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
        }
        self.hosp = locations.Hospitals()
        self.hosp_dist = None
        self.hosp_prox = None
        self.hosp_stations = None

        self.map_glyph_hospital = bkm.Circle(
            x='longitude',
            y='latitude',
            fill_alpha=0.8,
            fill_color='#cd5b1b',
            line_color=None,
            size=14,
        )
        self.map_glyph_subway = bkm.Diamond(
            x='longitude',
            y='latitude',
            fill_color='#3062C8',
            line_color=None,
            size=10,
        )
        self.map_options = {
            'lat': 40.70,
            'lng': -73.92,
            'map_type': 'roadmap',
            'zoom': 10,
        }
        self.map_plot = bkm.GMapPlot(
            api_key=self.api_key,
            x_range=bkm.Range1d(),
            y_range=bkm.Range1d(),
            map_options=bkm.GMapOptions(**self.map_options),
            plot_width=400,
            plot_height=600,
        )
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

    def hospital_distances(self):
        """
        Distances from subway stations to hospitals in NYC.
        """
        hospital_locs = np.array(self.hosp.hospitals
                                 .loc[:, ['latitude', 'longitude']])

        stations = (self.data
                    .loc[:, ['name', 'latitude', 'longitude']]
                    .drop_duplicates(['latitude', 'longitude'])
                    .reset_index(drop=True))
        subway_locs = np.array(stations.loc[:, ['latitude', 'longitude']])

        distances = np.empty((hospital_locs.shape[0], subway_locs.shape[0]))
        for hosp_n, h in enumerate(hospital_locs):
            for sub_n, s in enumerate(subway_locs):
                distances[hosp_n, sub_n] = gpd.vincenty(h, s).miles

        self.hosp_dist = pd.DataFrame(distances,
                                      index=self.hosp.hospitals.name)
        self.hosp_dist = pd.concat([(stations
                                     .loc[:, ['latitude', 'longitude']]
                                     .T),
                                    self.hosp_dist])
        self.hosp_dist.columns = stations.name

        self.hosp_dist['min_dist'] = (
            self.hosp_dist
            .drop(['latitude', 'longitude'], axis=0)
            .apply(lambda x: x.min(), axis=1))

        self.hosp_prox = (self.hosp_dist
                          .sort_values('min_dist')
                          .idxmin(axis=1))

    def hospital_proximity_plot(self, number=10):
        """
        Plot hospital and subway stations of interest

        .. warning:: This method requires a Google API Key

        :param int number: number of hospitals to query
        """
        if self.hosp_prox is None:
            self.hospital_distances()

        hosp_interest = self.hosp_prox[:number]
        hospital_locs = (self.hosp.hospitals
                         .loc[(self.hosp.hospitals
                               .name.isin(hosp_interest.index))])

        station_idx = (self.hosp_dist
                       .loc[hosp_interest.index, :]
                       .sort_values('min_dist')
                       .T
                       .reset_index(drop=True)
                       .T
                       .idxmin(axis=1))

        self.hosp_stations = (self.hosp_dist
                              .iloc[:, station_idx]
                              .loc[['latitude', 'longitude'], :]
                              .T
                              .reset_index())

        plot = self.map_plot
        plot.title.text = ('New York City Hospitals and Subway Stations of '
                           'Interest')

        hospitals = bkm.sources.ColumnDataSource(hospital_locs)
        plot.add_glyph(hospitals, self.map_glyph_hospital)

        subway_stations = bkm.sources.ColumnDataSource(self.hosp_stations)
        plot.add_glyph(subway_stations, self.map_glyph_subway)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Location', '@name'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.output_file('stations_interest.html')
        bkio.show(plot)

    def stations_plot(self):
        """
        Plot subway stations.

        .. warning:: This method requires a Google API Key
        """
        plot = self.map_plot
        plot.title.text = 'New York City Subway Stations'

        subway_stations = bkm.sources.ColumnDataSource(
            data=(self.data
                  .loc[:, ['name', 'latitude', 'longitude']]
                  .join(self.data.loc[:, 'entrance_type']
                        .astype(str))
                  .join(self.data.loc[:, 'exit_only']
                        .astype(str)
                        .str.replace('nan', 'No'))
                  .drop_duplicates())
        )
        plot.add_glyph(subway_stations, self.map_glyph_subway)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Location', '@name'),
            ('Entrance Type', '@entrance_type'),
            ('Exit Only', '@exit_only'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.output_file('stations.html')
        bkio.show(plot)

    def stations_locations_plot(self):
        """
        Plot subway stations and interest locations.

        .. warning:: This method requires a Google API Key
        """
        plot = self.map_plot
        plot.title.text = 'New York City Hospitals and Subway Stations'

        hospitals = bkm.sources.ColumnDataSource(self.hosp.hospitals)
        plot.add_glyph(hospitals, self.map_glyph_hospital)

        subway_stations = bkm.sources.ColumnDataSource(
            data=(self.data
                  .loc[:, ['name', 'latitude', 'longitude']]
                  .drop_duplicates())
        )
        plot.add_glyph(subway_stations, self.map_glyph_subway)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Location', '@name'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.output_file('stations_locations.html')
        bkio.show(plot)

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


class Turnstile:
    """
    Class to investigate the New York City subway turnstile data

    :Attributes:

    **data**: *pandas.DataFrame* NYC turnstile data
    **data_files**: *list* names of all available data files to download from \
        the url attribute
    **data_text**: *str* scraped text of NYC turnstile data
    **days**: *tuple* weekday names
    **request**: *requests.models.Response* response object from scraping \
        the url attribute
    **target_data** =
    **url**: *str* web address for turnstile data
    """
    def __init__(self):
        self.url = 'http://web.mta.info/developers/turnstile.html'
        self.request = requests.get(self.url)
        self.data = None
        self.data_files = None
        self.data_text = None
        self.data_types = {
            'ctrl_area': str,
            'unit': str,
            'scp': str,
            'station': str,
            'line_name': str,
            'division': str,
            'date': str,
            'time': str,
            'description': str,
            'entry': np.int32,
            'exit': np.int32,
        }
        self.date_start = '2017-05'
        self.date_end = '2017-05'
        self.days = (
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday',
            'Sunday',
        )
        self.target_data = None
        self.targets = None
        self.top_stations = None

        self.available_data_files()

    def __repr__(self):
        return f'Turnstile()'

    def available_data_files(self):
        """
        Find all available turnstile data files to retrieve.
        """
        files = re.findall(pattern=r'href="(data.*?.txt)"',
                           string=self.request.text)
        dates = ['20' + re.search(r'.*_(\d+).txt', x).groups(1)[0]
                 for x in files]
        index_names = [pd.to_datetime(f'{x[:4]}-{x[4:6]}-{x[6:]}')
                       for x in dates]
        self.data_files = pd.Series([f'http://web.mta.info/developers/{x}'
                                     for x in files],
                                    index=index_names,
                                    name='url')

    def format_date_time(self, data_file):
        """
        Format date and time into ISO 8601 Format

        :param str data_file: url of data file to retrieve
        """
        request = requests.get(data_file).text
        pattern = r'(\d{2})/(\d{2})/(\d{4})'
        self.data_text = re.sub(pattern, r'\3-\1-\2', request)

    def get_data(self, start=None, end=None):
        """
        Load turnstile data.

        .. note:: The attribute "data" will be reset each time this method \
            is called.

        :param str start: most current date given as YYYY, YYYY-MM or \
            YYYY-MM-DD
        :param str end: oldest date given as YYYY, YYYY-MM or YYYY-MM-DD
        """
        if end is None:
            end = self.date_end
        else:
            self.date_end = end
        if start is None:
            start = self.date_start
        else:
            self.date_start = start

        self.data = None
        data_urls = self.data_files[end:start]

        frames = []
        for url in data_urls:
            self.format_date_time(url)
            frames.append(pd.read_csv(io.StringIO(self.data_text),
                                      dtype=self.data_types,
                                      header=None,
                                      names=self.data_types.keys(),
                                      parse_dates=[[6, 7]],
                                      skiprows=1,
                                      )
                          )
        self.data = pd.concat(frames)
        self.data.set_index('date_time', inplace=True)
        for col in ('division', 'description'):
            self.data.loc[:, col] = (self.data.loc[:, col]
                                     .astype('category'))

    def get_targets(self, qty=10):
        """
        Get subset of all value containing only target location stations.

        .. warning:: The station dataset station names are not all exactly \
            the same as the turnstile dataset station names.

        :param int qty: number of top target stations to retrieve
        """
        stations = Stations()
        stations.hospital_distances()
        self.targets = stations.hosp_prox.head(qty)
        self.targets = (self.targets
                        .str
                        .upper())
        self.targets = (self.targets
                        .str
                        .replace('TH', ''))
        self.targets = (self.targets
                        .str
                        .replace('RD', ''))
        self.targets = (self.targets
                        .str
                        .replace('-', ' '))
        self.targets = (self.targets
                        .str
                        .replace('PLAZA', 'PLZ'))
        self.targets = (self.targets
                        .str
                        .replace('WYCK', 'WK'))

        self.target_data = (self.data
                            .loc[(self.data.station.isin(self.targets
                                                         .values
                                                         .tolist()))])

    def get_top_stations(self):
        """
        Get top stations based on entrance data.

        The ranking criteria is to find the fewest number of stations that \
            has a combined total less than or equal to 90% of all entry values.
        """
        if self.target_data is None:
            self.get_targets()

        entry_data = (self.target_data
                      .groupby('station')
                      .entry
                      .sum())

        entry_data = entry_data.to_frame()
        total = entry_data.entry.sum()
        entry_data['normalized'] = entry_data.apply(lambda x: x / total)
        entry_data.sort_values(by='normalized', ascending=False, inplace=True)

        self.top_stations = (entry_data[entry_data.cumsum().normalized <= 0.9]
                             .copy())

        station_names = list(self.top_stations.index)

        for day in self.days:
            day_mask = self.target_data.index.weekday_name == day
            station_mask = self.target_data.station.isin(station_names)
            self.top_stations[day] = (self.target_data
                                      .loc[day_mask & station_mask]
                                      .groupby('station')
                                      .sum()
                                      .entry)

    def targets_entry_plot(self, save=False):
        """
        Bar chart of turnstile entrance data.

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Station Box Plot',
                         figsize=(8, 10), facecolor='white',
                         edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))

        # Pie Plot
        pie = (self.target_data
               .groupby('station')['entry']
               .sum())
        (pie
         .plot(kind='pie', colormap='CMRmap', explode=[0.05] * pie.shape[0],
               labels=None, legend=None, shadow=True, ax=ax0))
        ax0.set_aspect('equal')
        ax0.set_ylabel('')
        ax0.legend(bbox_to_anchor=(1.3, 0.5),
                   labels=pie.index,
                   loc='center')

        # Bar Plot
        (self.target_data
         .groupby('station')[['entry', 'exit']]
         .sum()
         .sort_values(by='entry')
         .plot(kind='bar', alpha=0.5, edgecolor='black', ax=ax1))

        ax1.set_title(f'{self.date_start}  -  {self.date_end}',
                      fontsize=size['title'])
        ax1.legend(['Entry', 'Exit'])
        ax1.set_xlabel('Subway Station', fontsize=size['label'])
        ax1.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=80)
        ax1.set_ylabel('Turnstile Use ($billions$)',
                       fontsize=size['label'])
        ax1.yaxis.set_major_formatter(ax_formatter['billions'])

        plt.tight_layout()
        plt.suptitle('Turnstile Data',
                     fontsize=size['super_title'], x=0.53, y=1.05)

        save_fig('age_voted', save)

    def top_entry_bar_plot(self, save=False):
        """
        Bar chart of turnstile entrance data for top stations.

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Top Station Box Plot',
                         figsize=(8, 5), facecolor='white',
                         edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        (self.top_stations
         .sort_values(by='entry')
         .loc[:, 'Monday':'Sunday']
         .plot(kind='bar', alpha=0.5, edgecolor='black', ax=ax0))

        ax0.set_title(f'{self.date_start}  -  {self.date_end}',
                      fontsize=size['title'])
        ax0.set_xlabel('Subway Station', fontsize=size['label'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=80)
        ax0.set_ylabel('Turnstile Entries ($billions$)',
                       fontsize=size['label'])
        ax0.yaxis.set_major_formatter(ax_formatter['billions'])

        plt.tight_layout()
        plt.suptitle('Turnstile Data',
                     fontsize=size['super_title'], x=0.54, y=1.05)

        save_fig('age_voted', save)
