#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Locations Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import re
import urllib

from bokeh import io as bkio
from bokeh import models as bkm
import geocoder
import pandas as pd
import requests

from . import keys


class Hospitals:
    """
    Class to identify and describe hospitals in New York City.
    """
    def __init__(self):
        try:
            self.api_key = keys.GOOGLE_API_KEY
        except NameError:
            self.api_key = None
        self.url = ('https://en.wikipedia.org/wiki/'
                    'List_of_hospitals_in_New_York_City')
        self.request = requests.get(self.url)
        try:
            self.hospitals = pd.read_csv('https://timothyhelton.github.io/'
                                         'assets/data/nyc_hospitals.csv',
                                         index_col=0)
        except urllib.error.HTTPError:
            self.scrape_hospitals()

    def __repr__(self):
        return f'Hospitals()'

    def scrape_hospitals(self):
        """
        Scrap Wikipedia for New York City hospital names and addresses and \
        request geolocation from Google.
        """
        match = re.search(r'(.*)?Includes former names of hospitals',
                          self.request.text,
                          flags=re.DOTALL)
        open_hospitals = match.group()
        match_links = re.findall(r'<li><a.*?>(.*?)</a>.*', open_hospitals)
        match_no_links = re.findall(r'<li>(.*?),.*', open_hospitals)
        match_no_links = [x for x in match_no_links if '<a' not in x]
        match_no_links = [x for x in match_no_links if 'Division' not in x]
        hospitals = []
        for name in (match_links + match_no_links):
            g = geocoder.google(f'{name}, NY')
            lat, long = g.latlng
            hospitals.append([name, lat, long])
        self.hospitals = pd.DataFrame(hospitals,
                                      columns=['name', 'latitude',
                                               'longitude'])

    def hospital_plot(self):
        """
        Plot hospital locations.

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
        plot.title.text = 'New York City Hospitals'

        hospital = bkm.Circle(
            x='longitude',
            y='latitude',
            fill_alpha=0.8,
            fill_color='#cd5b1b',
            line_color=None,
            size=14,
        )

        hospitals = bkm.sources.ColumnDataSource(self.hospitals)
        plot.add_glyph(hospitals, hospital)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Location', '@name'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.output_file('hospitals.html')
        bkio.show(plot)
