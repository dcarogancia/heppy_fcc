from cpyroot import *

class Efficiency(object):

    def __init__(self, name, tree):
        self.name = name
        self.tree = tree
        self.num = None
        self.denom = None
        self.eff = None
        
    def hname(self, name):
        return '_'.join([self.name, name])
        
    def project(self, var, cut, numcut, nbins, xmin, xmax):
        if cut == '':
            cut = '1'
        if self.num is None: 
            self.num = TH1F( self.hname('num'), self.name, nbins, xmin, xmax)
        else:
            self.num.Reset()
        self.num.Sumw2()
        sRed.formatHisto(self.num)
        if self.denom is None: 
            self.denom = TH1F( self.hname('denom'), self.name, nbins, xmin, xmax)
        else:
            self.denom.Reset()
        self.denom.Sumw2()
        sBlue.formatHisto(self.denom)
        if self.eff is None: 
            self.eff = TH1F( self.hname('eff'), self.name, nbins, xmin, xmax)
        else:
            self.eff.Reset()
        self.eff.Sumw2()
        sBlack.formatHisto(self.eff)
        hists = [self.num, self.denom, self.eff]
        for h in hists:
            h.SetTitle(self.name)
            h.GetXaxis().SetTitle(var)
            h.GetYaxis().SetTitle('efficiency')
        
        print 'denom: var = ', var, ' cut = ', cut
        self.tree.Project(self.denom.GetName(), var, cut)
        numcut = '({cut}) && ({numcut})'.format(cut=cut,
                                                numcut=numcut)
        print 'num: var = ', var, ' cut = ', numcut
        self.tree.Project(self.num.GetName(), var, numcut)
        self.eff.Divide(self.num, self.denom, 1, 1, 'B')
        self.eff.GetYaxis().SetRangeUser(0,1)

