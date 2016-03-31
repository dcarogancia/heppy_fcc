from cpyroot import *
from heppy_fcc.macros.fractionstack import *
from heppy_fcc.macros.init import init

papas, cms = init('jets')

pdgids = [211,22,130, 11, 13]
        
def prepare_tree(tree):
    for pdgid in pdgids:
        tree.SetAlias("jet1_{pdgid}_frac".format(pdgid=pdgid),
                      "jet1_{pdgid}_e/jet1_e".format(pdgid=pdgid))
        tree.SetAlias("jet1_gen_{pdgid}_frac".format(pdgid=pdgid),
                      "jet1_gen_{pdgid}_e/jet1_gen_e".format(pdgid=pdgid))
    # elist = TEventList("elist")
    # tree.Draw(">>elist", "jet1_gen_211_e>0 && abs(jet1_gen_eta)<2. && abs(jet1_gen_pt>2.)")
    # tree.SetEventList(elist)

if papas:
    prepare_tree(papas)
if cms: 
    prepare_tree(cms)


c1 = TCanvas()

hists = []

def process_tree(tree, name):
    res_stack = FractionStack(pdgids, TH1F(name, ';E/E_{gen} (GeV)', 40, 0., 3))
    cut = 'jet1_pt>0. &&  abs(jet1_eta<1.2) && jet1_211_frac>0.'
    # cut = 'jet1_pt>0. &&  abs(jet1_eta<1.2) && jet1_211_frac>0. && jet1_22_frac==0.'
    res_stack.Project(tree, 'jet1_e / jet1_gen_e', cut)
    # res_stack.SetYRange(50, 50000)

    double_count = res_stack.histsum.Clone("double_count")
    double_count.SetLineStyle(2)
    double_count.SetLineWidth(3)
    double_count.SetLineColor(0)
    tree.Draw("jet1_e/jet1_gen_e>>double_count",
              "jet1_211_frac>0. && jet1_211_frac<1. && ({cut})".format(cut=cut),
              "same")
    
    missed = double_count.Clone("missed")
    missed.SetLineColor(5)
    tree.Draw("jet1_e/jet1_gen_e>>missed",
              "jet1_211_frac==0. && ({cut})".format(cut=cut),
              "same")

    pure = double_count.Clone("pure")
    pure.SetLineColor(4)
    tree.Draw("jet1_e/jet1_gen_e>>pure",
              "jet1_211_frac==1. && jet1_211_num==2 && ({cut})".format(cut=cut),
              "same")

    gPad.RedrawAxis()
    gPad.SetLogy()

    hists.extend([double_count, missed, pure])
    
    ntot = res_stack.histsum.GetEntries()
    nmissed = missed.GetEntries()
    ndouble = double_count.GetEntries() 
    npure = pure.GetEntries()
    nseen = ntot - nmissed
    eff = nseen / ntot
    ineff = nmissed / ntot
    dcount_prob = ndouble / ntot
    pure_prob = npure / ntot
    print 'eff = ', eff
    print 'missed = ', ineff
    print 'dcount_prob = ', dcount_prob
    print 'pure = ', pure_prob
    return res_stack

if papas:
    papas_can = TCanvas('papas', 'papas')
    papas_stack = process_tree(papas, 'papas') 
if cms:
    cms_can = TCanvas('cms', 'cms')
    cms_stack = process_tree(cms, 'cms')
