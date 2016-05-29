#!/usr/bin/env python

"""
	Configuration script for the analyzer of B0s -> K*0 Ds+ Ds- background events
	                                                 |   |   |-> pi- pi- pi+ pi0
						                             |   |-> pi+ pi+ pi- pi0
						                             |-> K+ pi-

	Note: it is supposed to be used within heppy_fcc framework
"""

import os
import heppy.framework.config as cfg
import logging

from ROOT import gSystem
from EventStore import EventStore as Events

from heppy_fcc.analyzers.BackgroundBs2DsDsKWithDs2PiPiPiPiAnalyzer import BackgroundBs2DsDsKWithDs2PiPiPiPiAnalyzer

logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('ILD-like', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/background_Bs2DsDsK_with_Ds2PiPiPiPi_100k.root'])

selected_components  = [input_component]

# analyzers

# analyzer for Bs -> Ds Ds K* events
bgana = cfg.Analyzer(BackgroundBs2DsDsKWithDs2PiPiPiPiAnalyzer,
					 smear_momentum = True,
					 momentum_x_resolution = 0.01,
					 momentum_y_resolution = 0.01,
					 momentum_z_resolution = 0.01,
					 smear_pv = True,
					#  IDL-like res
					 pv_x_resolution = 0.0025,
					 pv_y_resolution = 0.0025,
					 pv_z_resolution = 0.0025,
					#  progressive res
					#  pv_x_resolution = 0.001,
					#  pv_y_resolution = 0.001,
					#  pv_z_resolution = 0.001,
					#  outstanding res
					#  pv_x_resolution = 0.0005,
					#  pv_y_resolution = 0.0005,
					#  pv_z_resolution = 0.0005,
					 smear_sv = True,
					#  IDL-like res
					 sv_x_resolution = 0.007,
					 sv_y_resolution = 0.007,
					 sv_z_resolution = 0.007,
					#  progressive res
					#  sv_x_resolution = 0.003,
					#  sv_y_resolution = 0.003,
					#  sv_z_resolution = 0.003,
					#  outstanding res
					#  sv_x_resolution = 0.0015,
					#  sv_y_resolution = 0.0015,
					#  sv_z_resolution = 0.0015,
					 smear_tv = True,
					#  IDL-like res
					 tv_x_resolution = 0.005,
					 tv_y_resolution = 0.005,
					 tv_z_resolution = 0.005,
					#  progressive res
					#  tv_x_resolution = 0.002,
					#  tv_y_resolution = 0.002,
					#  tv_z_resolution = 0.002,
					#  outstanding res
					#  tv_x_resolution = 0.001,
					#  tv_y_resolution = 0.001,
					#  tv_z_resolution = 0.001,
					 stylepath = os.environ.get('FCC') + 'lhcbstyle.C',
					 tree_name = 'Events',
					 tree_title = 'Events',
					 mc_truth_tree_name = 'MCTruth',
					 mc_truth_tree_title = 'MC Truth',
					 verbose = False)

# definition of a sequence of analyzers, the analyzers will process each event in this order
sequence = cfg.Sequence([bgana])

# finalization of the configuration object.
gSystem.Load('libdatamodel')
config = cfg.Config(components = selected_components, sequence = sequence, services = [],events_class = Events)
