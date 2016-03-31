import os
import copy
import heppy.framework.config as cfg

debug = True
do_display = False

do_cms = False
do_papas = True
do_fcc = True
particle_matching = False
nevents_per_job = 1000
gen_jobs = 1

GEN = gen_jobs>0

if debug:
    print 'DEBUG MODE IS ON!'

source = None
selectedComponents = []
Events = None
if GEN:
    do_cms = False
    do_papas = True
    do_fcc = False
    for i in range(gen_jobs):
        component = cfg.Component(''.join(['sample_Chunk',str(i)]), files=['dummy.root'])
        selectedComponents.append(component)
    from heppy_fcc.analyzers.Gun import Gun
    source = cfg.Analyzer(
        Gun,
        pdgid = 211,
        ptmin = 0,
        ptmax = 20.,
        thetamin = -1.5,
        thetamax = 1.5,
        flat_pt = True
    )
    from heppy.framework.eventsgen import Events
elif do_cms:
    # from heppy_fcc.samples.higgs_350 import *  
    # from heppy_fcc.samples.gun import *
    # from heppy_fcc.samples.gun_MatEff_10_50 import *
    # selectedComponents = [gun_22_0_50]
    # selectedComponents = [gun_22_0_50_eta3]
    from heppy_fcc.samples.ee import ee_qq
    selectedComponents = [ee_qq]
    # from heppy_fcc.samples.gun_fullsim import *
    # selectedComponents = [gun_piplus]
    for comp in selectedComponents:
        comp.splitFactor = 10
        comp.isMC = True

    from heppy_fcc.analyzers.CMSReader import CMSReader
    source = cfg.Analyzer(
        CMSReader,
        gen_particles = 'genParticles',
        pf_particles = 'particleFlow' if do_cms else None
    )
    from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
elif do_fcc:
    comp = cfg.Component(
        'pythiafcc',
        files = ['ee_qq.root']
    )
    selectedComponents = [comp]
    from heppy_fcc.analyzers.FCCReader import FCCReader
    source = cfg.Analyzer(
        FCCReader
    )  
    from ROOT import gSystem
    gSystem.Load("libdatamodel")
    from eventstore import EventStore as Events

    
if debug:
    comp = selectedComponents[0]
    comp.splitFactor =1 
    # comp.files = selectedComponents[0].files[:1]
    # comp.files = ['gun_211_0to10_ME0_RECOSIM.root'] 
    selectedComponents = [comp]


from heppy_fcc.analyzers.Recoil import Recoil
gen_recoil = cfg.Analyzer(
    Recoil,
    instance_label = 'gen',
    sqrts = 350.,
    particles = 'gen_particles_stable'
)

from heppy_fcc.analyzers.JetClusterizer import JetClusterizer
gen_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'gen',
    particles = 'gen_particles_stable'
)

from heppy_fcc.analyzers.PFSim import PFSim
from heppy_fcc.fastsim.detectors.CMS import CMS
pfsim = cfg.Analyzer(
    PFSim,
    instance_label = 'papas',
    detector = CMS(),
    gen_particles = 'gen_particles_stable',
    sim_particles = 'sim_particles',
    rec_particles = 'rec_particles',
    display = do_display,
    verbose = False
)


papas_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'papas', 
    particles = 'papas_rec_particles'
)

from heppy_fcc.analyzers.Matcher import Matcher
papas_jet_match = cfg.Analyzer(
    Matcher,
    instance_label = 'papas', 
    match_particles = 'gen_jets',
    particles = 'papas_jets'
)

from heppy_fcc.analyzers.JetTreeProducer import JetTreeProducer
papas_jet_tree = cfg.Analyzer(
    JetTreeProducer,
    instance_label = 'papas',
    tree_name = 'events',
    tree_title = 'jets',
    jets = 'papas_jets'
)

papas_recoil = cfg.Analyzer(
    Recoil,
    instance_label = 'papas',
    sqrts = 350.,
    particles = 'papas_rec_particles'
)

from heppy_fcc.analyzers.Matcher import Matcher
papas_particle_match_r2g = cfg.Analyzer(
    Matcher,
    instance_label = 'papas_r2g', 
    particles = 'papas_rec_particles',
    match_particles = 'gen_particles_stable'
)

papas_particle_match_g2r = cfg.Analyzer(
    Matcher,
    instance_label = 'papas_g2r', 
    particles = 'gen_particles_stable',
    match_particles = [
        ('papas_rec_particles', None),
        ('papas_rec_particles', 211),
        ('papas_rec_particles', 130),
        ('papas_rec_particles', 22)
    ] 
)

from heppy_fcc.analyzers.ParticleTreeProducer import ParticleTreeProducer
papas_particle_tree_r2g = cfg.Analyzer(
    ParticleTreeProducer, 
    instance_label = 'papas_r2g',
    particles = 'papas_rec_particles'
    )

papas_particle_tree_g2r = cfg.Analyzer(
    ParticleTreeProducer, 
    instance_label = 'papas_g2r',
    particles = 'gen_particles_stable'
    )

from heppy_fcc.analyzers.Higgs350TreeProducer import Higgs350TreeProducer
papas_tree = cfg.Analyzer(
    Higgs350TreeProducer,
    instance_label = 'papas'
    )

papas_sequence = [
    pfsim, 
    papas_jets, 
    papas_jet_match,
    papas_jet_tree,
    papas_recoil,
    papas_tree, 
    ]

if particle_matching: 
    papas_sequence.extend([
            papas_particle_match_g2r,
            papas_particle_tree_g2r,      
            ])


from heppy_fcc.analyzers.FastsimCleaner import FastsimCleaner 
cms_fastsim_cleaner = cfg.Analyzer(
    FastsimCleaner, 
    particles = 'pf_particles'
)

cms_jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'cms', 
    particles = 'pf_particles'
)

cms_jet_match = cfg.Analyzer(
    Matcher,
    instance_label = 'cms', 
    match_particles = 'gen_jets',
    particles = 'cms_jets'
)

cms_jet_tree = cfg.Analyzer(
    JetTreeProducer,
    instance_label = 'cms',
    tree_name = 'events',
    tree_title = 'jets',
    jets = 'cms_jets'
)


cms_recoil = cfg.Analyzer(
    Recoil,
    instance_label = 'cms',
    sqrts = 350.,
    particles = 'pf_particles'
)

from heppy_fcc.analyzers.Matcher import Matcher
cms_particle_match_r2g = cfg.Analyzer(
    Matcher,
    instance_label = 'cms_r2g', 
    particles = 'pf_particles',
    match_particles = [
        ('gen_particles_stable', None),
        ('gen_particles_stable', 211),
        ('gen_particles_stable', 130),
        ('gen_particles_stable', 22)
    ] 
)

cms_particle_match_g2r = cfg.Analyzer(
    Matcher,
    instance_label = 'cms_g2r', 
    particles = 'gen_particles_stable',
    match_particles = [
        ('pf_particles', None),
        ('pf_particles', 211),
        ('pf_particles', 130)
    ] 
)

from heppy_fcc.analyzers.PFAnalyzer import PFAnalyzer
cms_pfanalyzer = cfg.Analyzer(
    PFAnalyzer,
    particles = 'pf_particles'
)

from heppy_cms.analyzers.core.JSONAnalyzer import JSONAnalyzer
cms_json = cfg.Analyzer(
    JSONAnalyzer, 
    json = None
    )

from heppy_fcc.analyzers.ParticleTreeProducer import ParticleTreeProducer
cms_particle_tree_r2g = cfg.Analyzer(
    ParticleTreeProducer, 
    instance_label = 'cms_r2g',
    particles = 'pf_particles'
    )
cms_particle_tree_g2r = cfg.Analyzer(
    ParticleTreeProducer, 
    instance_label = 'cms_g2r',
    particles = 'gen_particles_stable'
    )

from heppy_fcc.analyzers.Higgs350TreeProducer import Higgs350TreeProducer
cms_tree = cfg.Analyzer(
    Higgs350TreeProducer,
    instance_label = 'cms'
    )

cms_sequence = [
    cms_json,
    cms_fastsim_cleaner,
    cms_jets, 
    cms_jet_match,
    cms_jet_tree,
    cms_recoil,
    cms_tree, 
    ]

if particle_matching: 
    cms_sequence.extend([
            cms_particle_match_g2r,
            cms_particle_tree_g2r,      
            ])


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    gen_recoil,
    gen_jets,
    ] )

if do_papas:
    sequence.extend(papas_sequence)
if do_cms:
    sequence.extend(cms_sequence)

# sequence = cfg.Sequence([source])

# sequence = cfg.Sequence(
#     [source, 
#      cms_fastsim_cleaner,
#      cms_particle_match_r2g,
#      cms_pfanalyzer
#      ]
#     )
    
config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)

    
if __name__ == '__main__':
    import sys
    from heppy.framework.looper import Looper
    import logging

    # next 2 lines necessary to deal with reimports from ipython
    logging.shutdown()
    reload(logging)
    logging.basicConfig(level=logging.INFO)

    import random
    # for reproducible results
    random.seed(0xdeadbeef)

    def process(iev=None):
        if iev is None:
            iev = loop.iEvent
        loop.process(iev)
        if display:
            display.draw()

    def next():
        loop.process(loop.iEvent+1)
        if display:
            display.draw()            

    iev = None
    if len(sys.argv)==2:
        iev = int(sys.argv[1])
    loop = Looper( 'looper', config,
                   nEvents=nevents_per_job,
                   nPrint=0,
                   timeReport=True)
    # pfsim = loop.analyzers[1]
    # display = getattr(pfsim, 'display', None)
    display = None
    # simulator = pfsim.simulator
    # detector = simulator.detector
    if iev is not None:
        process(iev)
    else:
        loop.loop()
        loop.write()
