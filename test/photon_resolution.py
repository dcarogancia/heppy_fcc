from cpyroot import *
from cpyroot import *
from heppy_fcc.macros.resolution import *
from heppy_fcc.macros.init import init


papas, cms = init()

xnbins, xmin, xmax = 50, 3, 20
ynbins, ymin, ymax = 20, 0.9, 1.1

papas_res_e = Resolution('papas_res_e', papas, style=sBlack)
papas_res_e.project('ptc_match_e/ptc_e:ptc_e', 'ptc_match_pdgid == ptc_pdgid',
                    xnbins, xmin, xmax, ynbins, ymin, ymax)
# papas_res_e.fit_slice(10)
# papas_res_e.draw_2d()

if cms:
    cms_res_e = Resolution('cms_res_e', cms, style=sBlue)
    cms_res_e.project('ptc_match_e/ptc_e:ptc_e', 'ptc_match_22_pdgid == ptc_pdgid',
                      xnbins, xmin, xmax, ynbins, ymin, ymax)
    cms_res_e.hsigma.Draw()

    
papas_res_e.hsigma.Draw('same')
