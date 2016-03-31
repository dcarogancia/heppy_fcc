import heppy.framework.config as cfg

from getFiles import getFiles 

gun_211_MatEff_10_50 = cfg.Component(
    'gun_211_Mateff_10_50',
    files = getFiles("/Gun_211_MatEff_10_50/743_v1/RECOSIM"),
) 



samples = [
    gun_211_MatEff_10_50,
]

if __name__ == '__main__':

    import pprint 
    for g in samples:
        print g
