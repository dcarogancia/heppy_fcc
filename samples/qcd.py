import heppy.framework.config as cfg

from getFiles import getFiles 

qcd = cfg.Component(
    'qcd',
    files = getFiles(
        "/QCD_Pt-120to170_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-AsymptNoPUReco_MCRUN2_74_V9A-v1/GEN-SIM-RECO", 
        user='CMS', cache=True),
) 

qcd_em = cfg.Component(
    'qcd_em',
    files = getFiles(
        "/QCD_Pt-15to20_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-AsymptNoPUReco_MCRUN2_74_V9A-v1/GEN-SIM-RECO", 
        user='CMS', cache=True),
) 


samples = [
    qcd
]

if __name__ == '__main__':

    import pprint 
    for g in samples:
        print g
