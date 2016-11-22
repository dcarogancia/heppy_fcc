#!/usr/bin/env python

"""
    Configuration script for the analyzer of B0s -> K+ Ds- events (special for Stephane)
                                                        |-> K- K+ pi-

    Note: it is supposed to be used within heppy_fcc framework
"""

import os
import heppy.framework.config as cfg
import logging

from ROOT import gSystem
from EventStore import EventStore as Events

from heppy_fcc.analyzers.SpecialBs2DsKWithDs2KKPiAnalyzer import SpecialBs2DsKWithDs2KKPiAnalyzer

logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('ILD', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/special_Bs2DsK_with_Ds2KKPi_10k.root'])

selected_components  = [input_component]

# analyzers

# analyzer for Bs -> Ds K events
bgana = cfg.Analyzer(SpecialBs2DsKWithDs2KKPiAnalyzer,
                     smear_momentum = True,
                    #  ILD
                     momentum_resolution_a = 2e-5,
                     momentum_resolution_b = 1e-3,
                     # FCC
                    #  momentum_resolution_a = 1e-5,
                    #  momentum_resolution_b = 5e-4,
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
config = cfg.Config(components = selected_components, sequence = sequence, services = [], events_class = Events)
