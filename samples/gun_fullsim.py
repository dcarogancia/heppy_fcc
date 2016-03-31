import heppy.framework.config as cfg

from getFiles import getFiles 

gun_piplus = cfg.Component(
    'gun_piplus',
    files = getFiles("/SinglePiPlus_P-1to2000_Expo_13TeV_ExpoRandomPGun/RunIISpring15DR74-AsymptNoPUReco_MCRUN2_74_V9A-v1/GEN-SIM-RECO", user='CMS'),
) 
