from cpyroot import *
from heppy_fcc.macros.efficiency import *
from heppy_fcc.macros.init import init



def efficiencies(name, tree):
    
    # e_eff = Efficiency('_'.join([name, 'e']), tree)
    # e_eff.project('ptc_e', '', 'ptc_match_e>0. && ptc_match_pdgid==ptc_pdgid', 20, 0, 5)

    found = 'ptc_match_pt>0. && ptc_match_pdgid==ptc_pdgid'

    pt_eff = Efficiency('_'.join([name, 'pt']), tree)
    pt_eff.project('ptc_pt', 'abs(ptc_eta)<1.4', found, 100, 0, 20)

    e_eff = Efficiency('_'.join([name, 'e']), tree)
    e_eff.project('ptc_e', 'abs(ptc_eta)<1.4', found, 100, 0, 20)

    theta_eff = Efficiency('_'.join([name, 'theta']), tree)
    theta_eff.project('ptc_theta', 'ptc_e>2.', found, 100, -3, 3 )

    eta_eff = Efficiency('_'.join([name, 'eta']), tree)
    eta_eff.project('ptc_eta', 'ptc_e>2.', found, 100, -4, 4 )

    return [pt_eff, e_eff, theta_eff, eta_eff]


papas, cms = init('particles')

papas_eff = efficiencies('papas', papas)
cms_eff = [None]*len(papas_eff)
if cms:
    cms_eff = efficiencies('cms', cms)

canvases = [] 
for pp, cm in zip(papas_eff, cms_eff):
        canvases.append(TCanvas())
        pp.eff.Draw()
        if cm:
            sBlue.formatHisto(cm.eff)
            cm.eff.Draw('same')

