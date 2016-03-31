import os
import copy
import heppy.framework.config as cfg

debug = False

if debug:
    print 'DEBUG MODE IS ON!'

source = None
selectedComponents = []
Events = None

# from heppy_fcc.samples.higgs_350 import *  
# from heppy_fcc.samples.gun import *
# from heppy_fcc.samples.gun_MatEff_10_50 import *
# selectedComponents = [gun_22_0_50]
# selectedComponents = [gun_22_0_50_eta3]
from heppy_fcc.samples.qcd import qcd_em as qcd
selectedComponents = [qcd]
# qcd.files = ['qcd_mu_enriched.root']
# from heppy_fcc.samples.gun_fullsim import *
# selectedComponents = [gun_piplus]
for comp in selectedComponents:
    comp.splitFactor = 10
    comp.isMC = True

from heppy_fcc.analyzers.CMSJetReader import CMSJetReader
source = cfg.Analyzer(
    CMSJetReader,
    gen_jets = 'ak4GenJetsNoNu',
    gen_jet_pt = 20, 
    jets = 'ak4PFJets', 
    jet_pt = 20,
    nlead = 2 
 
)
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
  
if debug:
    comp = selectedComponents[0]
    comp.splitFactor =1 
    selectedComponents = [comp]



from heppy_fcc.analyzers.JetClusterizer import JetClusterizer
gen_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'gen',
    particles = 'gen_particles_stable'
)

from heppy_fcc.analyzers.FastsimCleaner import FastsimCleaner 
cms_fastsim_cleaner = cfg.Analyzer(
    FastsimCleaner, 
    particles = 'pf_particles'
)

from heppy_fcc.analyzers.JetClusterizer import JetClusterizer
cms_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'cms', 
    particles = 'pf_particles'
)

from heppy_fcc.analyzers.Matcher import Matcher
cms_jet_match = cfg.Analyzer(
    Matcher,
    instance_label = 'cms', 
    match_particles = 'gen_jets',
    particles = 'cms_jets'
)

from heppy_fcc.analyzers.JetTreeProducer import JetTreeProducer
cms_jet_tree = cfg.Analyzer(
    JetTreeProducer,
    instance_label = 'cms',
    tree_name = 'events',
    tree_title = 'jets',
    jets = 'cms_jets'
)

from heppy_cms.analyzers.core.JSONAnalyzer import JSONAnalyzer
cms_json = cfg.Analyzer(
    JSONAnalyzer, 
    json = None
    )

cms_sequence = [
    cms_json,
    # cms_fastsim_cleaner,
    # cms_jets, 
    cms_jet_match,
    cms_jet_tree,
    ]

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    # gen_jets,
    ] )
sequence.extend(cms_sequence)

    
config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)

if __name__ == '__main__':
    print config
