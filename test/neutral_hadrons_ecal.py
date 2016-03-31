from cpyroot import *
from heppy_fcc.macros.efficiency import *
from heppy_fcc.macros.init import init

papas, cms = init()
cms.SetMarkerColor(4)

def prob_ecal(tree, cut, same=''):
    tree.Draw('ptc_match_22_pdgid==22:ptc_e', '(1)&&({cut})'.format(cut=cut),
              'prof'+same)

def e_ecal(tree, cut, same=''):
    tree.Draw('ptc_match_22_e/ptc_e:ptc_e',
              '(ptc_match_22_e>0)&&({cut})'.format(cut=cut), 'prof'+same)

cut = 'abs(ptc_eta)>1.4'
 
c1 = TCanvas()
prob_ecal(papas, cut)
prob_ecal(cms, cut, 'same')

c2 = TCanvas()
e_ecal(papas, cut)
e_ecal(cms, cut, 'same')
