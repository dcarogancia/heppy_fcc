import os
import copy
import heppy.framework.config as cfg

# input component 
# several input components can be declared,
# and added to the list of selected components

gen_jobs = 0
do_display = True
do_pf = False

nevents_per_job = 5000

GEN = gen_jobs 
FCC = os.environ.get('FCCEDM', False) and not GEN
CMS = os.environ.get('CMSSW_BASE', False) and not GEN

if gen_jobs>1:
    do_display = False

selectedComponents = None
if CMS:
    # from heppy_fcc.samples.gun_0_50 import *  
    from heppy_fcc.samples.higgs_350 import hz_cms  
    selectedComponents  = [hz_cms]
    # selectedComponents = [gun_211_0_50]
    for comp in selectedComponents:
        comp.splitFactor = 1
else: 
    inputSample = cfg.Component(
        'albers_example',
        files = ['example.root']
    )
    selectedComponents  = [inputSample]

source = None
if GEN:
    selectedComponents = []
    for i in range(gen_jobs):
        component = cfg.Component(''.join(['sample_Chunk',str(i)]), files=['dummy.root'])
        selectedComponents.append(component)
    from heppy_fcc.analyzers.Gun import Gun
    source = cfg.Analyzer(
        Gun,
        pdgid = 130,
        ptmin = 0.,
        ptmax = 10.
        )
elif FCC:
    from heppy_fcc.analyzers.FCCReader import FCCReader
    source = cfg.Analyzer(
        FCCReader
        )    
elif CMS:
    from heppy_fcc.analyzers.CMSReader import CMSReader
    source = cfg.Analyzer(
        CMSReader,
        gen_particles = 'genParticles',
        pf_particles = 'particleFlow' if do_pf else None
        )
else:
    raise ValueError('not a generator job, and experience unrecognized. Set the CMS or FCC environment')



from heppy_fcc.analyzers.PFSim import PFSim
pfsim = cfg.Analyzer(
    PFSim,
    display = do_display,
    verbose = False
)

from heppy_fcc.analyzers.JetClusterizer import JetClusterizer


genjets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'gen',
    particles = 'gen_particles_stable'
)


# jets from pfsim 

jets = cfg.Analyzer(
    JetClusterizer,
    instance_label = 'rec', 
    particles = 'particles'
)

from heppy_fcc.analyzers.JetAnalyzer import JetAnalyzer
jetana = cfg.Analyzer(
    JetAnalyzer,
    instance_label = 'rec', 
    jets = 'rec_jets',
    genjets = 'gen_jets'
)

from heppy_fcc.analyzers.JetTreeProducer import JetTreeProducer
tree = cfg.Analyzer(
    JetTreeProducer,
    instance_label = 'rec',
    tree_name = 'events',
    tree_title = 'jets',
    jets = 'rec_jets'
)

jetsequence = [
    jets,
    jetana, 
    tree
]

# pf jet sequence

if CMS and do_pf:
    pfjetsequence = copy.deepcopy(jetsequence)
    for ana in pfjetsequence: 
        ana.instance_label = 'pf'
        if hasattr(ana, 'jets'):
            ana.jets = 'pf_jets'
        if hasattr(ana, 'particles'):
            ana.particles = 'pf_particles'
    

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    pfsim,
    genjets,
    ] )

sequence.extend(jetsequence)
if CMS and do_pf:
    sequence.extend(pfjetsequence)

if FCC:
    from heppy_fcc.analyzers.GenAnalyzer import GenAnalyzer
    genana = cfg.Analyzer(
        GenAnalyzer
    )
    # sequence.append(genana)
    
# inputSample.files.append('albers_2.root')
# inputSample.splitFactor = 2  # splitting the component in 2 chunks

# finalization of the configuration object.
Events = None
if gen_jobs:
    from heppy.framework.eventsgen import Events 
elif os.environ.get('FCCEDM'):
    from ROOT import gSystem
    gSystem.Load("libdatamodel")
    from eventstore import EventStore as Events
elif os.environ.get('CMSSW_BASE'):
    from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
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
    logging.basicConfig(level=logging.ERROR)

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
                   nPrint=5,
                   timeReport=True)
    pfsim = loop.analyzers[1]
    display = getattr(pfsim, 'display', None)
    simulator = pfsim.simulator
    detector = simulator.detector
    if iev is not None:
        process(iev)
    else:
        loop.loop()
        loop.write()
