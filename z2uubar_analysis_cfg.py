import os
import heppy.framework.config as cfg
import logging
logging.basicConfig(level=logging.WARNING)

# input component
# several input components can be declared and added to the list of selected components
input_component = cfg.Component('b2stt', files = ['/afs/cern.ch/work/a/ansemkiv/private/FCC/analysis/Z2uubar_10.root'])

selected_components  = [input_component]

# analyzers

# analyzer for signal events
from heppy_fcc.analyzers.Z2uubar_analyzer import Z2uubar_analyzer
ana = cfg.Analyzer(Z2uubar_analyzer)

# definition of a sequence of analyzers, the analyzers will process each event in this order
sequence = cfg.Sequence([ana])

# finalization of the configuration object.
from ROOT import gSystem
gSystem.Load("libdatamodel")
from EventStore import EventStore as Events
config = cfg.Config(components = selected_components, sequence = sequence, services = [], events_class = Events)
