from cpyroot import * 

def init(treetype):
    import sys
    import os

    officialStyle(gStyle)
    
    sample = sys.argv[1]

    if treetype == 'particles':
        papas_rootfile = 'heppy_fcc.analyzers.ParticleTreeProducer.ParticleTreeProducer_papas_g2r/tree.root'
        cms_rootfile = 'heppy_fcc.analyzers.ParticleTreeProducer.ParticleTreeProducer_cms_g2r/tree.root'
    elif treetype == 'jets':
        papas_rootfile = 'heppy_fcc.analyzers.JetTreeProducer.JetTreeProducer_papas/jet_tree.root'
        cms_rootfile = 'heppy_fcc.analyzers.JetTreeProducer.JetTreeProducer_cms/jet_tree.root'
        
    papas_rootfile = '/'.join([sample, papas_rootfile])
    cms_rootfile = '/'.join([sample, cms_rootfile])

    do_papas = os.path.isfile(papas_rootfile)
    do_cms = os.path.isfile(cms_rootfile)
    
    # import pdb; pdb.set_trace()
    papas = Chain(papas_rootfile) if do_papas else None
    cms = Chain(cms_rootfile) if do_cms else None
    return papas, cms

        
