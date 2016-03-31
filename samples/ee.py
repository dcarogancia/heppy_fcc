import heppy.framework.config as cfg

from getFiles import getFiles 

ee_qq = cfg.Component(
    'ee_qq',
    files = getFiles("/ee_qq/745_v1/RECOSIM", cache=True),
) 


samples = [
    ee_qq
]

if __name__ == '__main__':

    import pprint 
    for g in samples:
        print g
