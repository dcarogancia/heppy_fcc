from cpyroot import *
from cpyroot import *
from heppy_fcc.macros.resolution import *
from heppy_fcc.macros.init import init


papas, cms = init()

papas_res_e = Resolution('papas_res_e', papas, style=sBlack)
papas_res_e.project('ptc_match_e/ptc_e:ptc_e', 'ptc_match_pdgid == ptc_pdgid',
                    20, 0.3, 20, 20, 0., 2)
# papas_res_e.fit_slice(10)
# papas_res_e.draw_2d()

if cms:
    cms_res_e = Resolution('cms_res_e', cms, style=sBlue)
    cms_res_e.project('ptc_match_e/ptc_e:ptc_e', 'ptc_match_pdgid == ptc_pdgid',
                      20, 0.3, 20, 20, 0, 2)
    cms_res_e.hsigma.Draw()

    
papas_res_e.hsigma.Draw('same')
