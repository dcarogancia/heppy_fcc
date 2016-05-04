## Configuration script for background (B0s -> Ds+ Ds- K*0) analyzer

import os
import heppy.framework.config as cfg
import logging
logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('b2stt', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/background_Bs2DsDsK_with_Ds2TauNu_250.root'])

selected_components  = [input_component]

# analyzers

# analyzer for Bs -> Ds Ds K* events
from heppy_fcc.analyzers.BackgroundBs2DsDsKWithDs2TauNuAnalyzer import BackgroundBs2DsDsKWithDs2TauNuAnalyzer
bgana = cfg.Analyzer(BackgroundBs2DsDsKWithDs2TauNuAnalyzer,
					 smear_momentum = True,
					 momentum_x_resolution = 0.01,
					 momentum_y_resolution = 0.01,
					 momentum_z_resolution = 0.01,
					 smear_pv = True,
					#  IDL-like res
					 pv_x_resolution = 0.0025,
					 pv_y_resolution = 0.0025,
					 pv_z_resolution = 0.0025,
					#  improved res
					#  pv_x_resolution = 0.001,
					#  pv_y_resolution = 0.001,
					#  pv_z_resolution = 0.001,
					#  ALEPH-like res
					#  pv_x_resolution = 0.01,
					#  pv_y_resolution = 0.01,
					#  pv_z_resolution = 0.01,
					 smear_sv = True,
					#  IDL-like res
					 sv_x_resolution = 0.007,
					 sv_y_resolution = 0.007,
					 sv_z_resolution = 0.007,
					#  improved res
					#  sv_x_resolution = 0.003,
					#  sv_y_resolution = 0.003,
					#  sv_z_resolution = 0.003,
					#  ALEPH-like res
 					# sv_x_resolution = 0.04,
 					# sv_y_resolution = 0.04,
 					# sv_z_resolution = 0.04,
					 smear_tv = True,
					#  IDL-like res
					 tv_x_resolution = 0.005,
					 tv_y_resolution = 0.005,
					 tv_z_resolution = 0.005,
					#  improved res
					#  tv_x_resolution = 0.002,
					#  tv_y_resolution = 0.002,
					#  tv_z_resolution = 0.002,
					#  ALEPH-like res
					#  tv_x_resolution = 0.02,
					#  tv_y_resolution = 0.02,
					#  tv_z_resolution = 0.02,
					 stylepath = os.environ.get('FCC') + 'lhcbstyle.C',
					 tree_name = 'Events',
					 tree_title = 'Events',
					 mc_truth_tree_name = 'MCTruth',
					 mc_truth_tree_title = 'MC Truth',
					 verbose = False
					 )

# definition of a sequence of analyzers, the analyzers will process each event in this order
sequence = cfg.Sequence([bgana])

# finalization of the configuration object.
from ROOT import gSystem
gSystem.Load('libdatamodel')
from EventStore import EventStore as Events
config = cfg.Config(components = selected_components, sequence = sequence, services = [],events_class = Events)
