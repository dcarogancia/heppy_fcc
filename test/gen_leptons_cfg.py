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
    files = ['top_mu.root']
)
selectedComponents = [comp]

from heppy_fcc.analyzers.FCCReader import FCCReader
source = cfg.Analyzer(
    FCCReader
)  

from ROOT import gSystem
gSystem.Load("libdatamodel")
from eventstore import EventStore as Events


from heppy_fcc.analyzers.LeptonAnalyzer import LeptonAnalyzer
gen_muons = cfg.Analyzer(
    LeptonAnalyzer,
    instance_label = 'gen_muons', 
    particles = 'gen_particles_stable',
    pdgid = 13
)

from heppy_fcc.analyzers.LeptonTreeProducer import LeptonTreeProducer
gen_muons_tree = cfg.Analyzer(
    LeptonTreeProducer,
    instance_label = 'gen',
    tree_name = 'events',
    tree_title = 'leptons',
    leptons = 'gen_muons'
)

from heppy_fcc.analyzers.IsoParticleTreeProducer import IsoParticleTreeProducer
gen_muons_tree = cfg.Analyzer(
    IsoParticleTreeProducer,
    instance_label = 'gen',
    tree_name = 'ptcs',
    tree_title = 'isolation particles',
    leptons = 'gen_muons'
)


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    gen_muons,
    gen_muons_tree,
    ] )
 
config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)
