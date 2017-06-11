#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound
import os.path as osp

from . import demographics
from . import locations
from . import subway
from . import utils


__version__ = '1.0.0'

try:
    _dist = get_distribution('nyc_signature')
    dist_loc = osp.normcase(_dist.location)
    here = osp.normcase(__file__)
    if not here.startswith(osp.join(dist_loc, 'nyc_signature')):
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version
