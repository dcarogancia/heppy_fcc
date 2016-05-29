#!/usr/bin/env python

"""
	Configuration script for the analyzer of B0d -> K*0 Ds+ tau- nu background events
	                                                 |   |   |-> pi- pi- pi+ nu
						                             |   |-> tau+ nu
						                             |	      |-> pi+ pi+ pi- nu
						                             |-> K+ pi-

	Note: it is supposed to be used within heppy_fcc framework
"""

import os
import heppy.framework.config as cfg
import logging
logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('ILD-like', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/background_Bd2DsKTauNu_with_Ds2TauNu_100k.root'])

selected_components  = [input_component]

# analyzers

# analyzer for Bd -> Ds K* tau nu_tau events
from heppy_fcc.analyzers.BackgroundBd2DsKTauNuWithDs2TauNuAnalyzer import BackgroundBd2DsKTauNuWithDs2TauNuAnalyzer
bgana = cfg.Analyzer(BackgroundBd2DsKTauNuWithDs2TauNuAnalyzer,
					 smear_momentum = True,
					 momentum_x_resolution = 0.01,
					 momentum_y_resolution = 0.01,
					 momentum_z_resolution = 0.01,
					 smear_pv = True,
					#  ILD-like res
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
					#  ILD-like res
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
					#  ILD-like res
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
from ROOT import gSystem
gSystem.Load('libdatamodel')
from EventStore import EventStore as Events
config = cfg.Config(components = selected_components, sequence = sequence, services = [], events_class = Events)
