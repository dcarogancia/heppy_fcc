import os
import copy
import heppy.framework.config as cfg

import logging
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

comp = cfg.Component(
    'example',
    files = ['hh_ttbar.root']
)
comp.files.append("example2.root")

comp.splitFactor = 2
selectedComponents = [comp]

from heppy_fcc.analyzers.FCCReader import FCCReader
source = cfg.Analyzer(
    FCCReader
)

from ROOT import gSystem
gSystem.Load("libdatamodel")
from EventStore import EventStore as Events

from heppy_fcc.analyzers.JetClusterizer import JetClusterizer
gen_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'gen',
    particles = 'gen_particles_stable'
)

from heppy_fcc.analyzers.SimpleTreeProducer import SimpleTreeProducer
gen_tree = cfg.Analyzer(
    SimpleTreeProducer,
)

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    gen_jets,
    ] )

# comp.files.append('example_2.root')
# comp.splitFactor = 2  # splitting the component in 2 chunks

config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)
