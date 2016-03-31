import heppy.framework.config as cfg

from getFiles import getFiles 


wwh = cfg.Component(
    'wwh',
    files = getFiles("/WWH/HZHA_100k_v1/GEN"),
) 

hz = cfg.Component(
    'hz',
    files = getFiles("/HZ/HZHA_100k_v1/GEN"),
) 

hz_cms =  cfg.Component(
    'hz_cms',
    files = getFiles("/HZ/HZHA_100k_fastsim_v1/AODSIM"),
) 

samples = [
    wwh,
    hz
]

samples_cms = [
    hz_cms
]

if __name__ == '__main__':

    import pprint 
    for g in samples:
        print g
