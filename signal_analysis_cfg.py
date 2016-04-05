## Configuration script for signal (B0d -> K*0 tau+ tau-) analyzer

import os
import heppy.framework.config as cfg
import logging
logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('b2stt', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/output.root'])

selected_components  = [input_component]

# analyzers

# analyzer for signal events
from heppy_fcc.analyzers.SignalAnalyzer import SignalAnalyzer
sigana = cfg.Analyzer(SignalAnalyzer,
					 smear_pv = True,
					 # IDL-like res
					#  pv_x_resolution = 0.0025,
					#  pv_y_resolution = 0.0025,
					#  pv_z_resolution = 0.0025,
					 # improved res
					 pv_x_resolution = 0.001,
					 pv_y_resolution = 0.001,
					 pv_z_resolution = 0.001,
					 smear_sv = True,
					 # IDL-like res
					#  sv_x_resolution = 0.007,
					#  sv_y_resolution = 0.007,
					#  sv_z_resolution = 0.007,
					 # improved res
					 sv_x_resolution = 0.003,
					 sv_y_resolution = 0.003,
					 sv_z_resolution = 0.003,
					 smear_tv = True,
					 # IDL-like res
					#  tv_x_resolution = 0.005,
					#  tv_y_resolution = 0.005,
					#  tv_z_resolution = 0.005,
					 # improved res
					 tv_x_resolution = 0.002,
					 tv_y_resolution = 0.002,
					 tv_z_resolution = 0.002,
					 smear_momentum = True,
					 momentum_x_resolution = 0.01,
					 momentum_y_resolution = 0.01,
					 momentum_z_resolution = 0.01,
					 stylepath = os.environ.get('FCC') + 'lhcbstyle.C',
					 tree_name = 'Events',
					 tree_title = 'Events',
					 mc_truth_tree_name = 'MCTruth',
					 mc_truth_tree_title = 'MC Truth',
					 verbose = False
					 )

# definition of a sequence of analyzers, the analyzers will process each event in this order
sequence = cfg.Sequence([sigana])

# finalization of the configuration object.
from ROOT import gSystem
gSystem.Load('libdatamodel')
from EventStore import EventStore as Events
config = cfg.Config(components = selected_components, sequence = sequence, services = [], events_class = Events)
