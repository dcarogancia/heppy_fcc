from cpyroot import *
from heppy.statistics.value import Value

import copy

class Resolution(object):

    def __init__(self, name, tree, style=sBlack):
        self.name = name
        self.tree = tree
        self.h2d = None
        self.style = copy.copy(style)
        self.style.fillStyle = 0
        
    def hname(self, name):
        return '_'.join([self.name, name])
        
    def project(self, var, cut, nbins, xmin, xmax, ynbins, ymin, ymax):
        if self.h2d is None: 
            self.h2d = TH2F( self.hname('h2d'), self.name,
                             nbins, xmin, xmax,
                             ynbins, ymin, ymax)
        else:
            self.h2d.Reset()
        self.h2d.SetTitle(self.name)
        vary, varx = var.split(':')
        self.h2d.GetXaxis().SetTitle(varx)
        self.h2d.GetYaxis().SetTitle(vary)

        print vary
        print varx
        print cut
        
        self.tree.Project(self.h2d.GetName(), var, cut)
        # self.h2d.Draw('colz')
        self.fit()

    def fit_slice(self, ibin, draw=True, nentries_min=20):
        self.proj = self.h2d.ProjectionY('',ibin,ibin,'')
        if self.proj.Integral()<nentries_min:
            return None, None
        opt = '' if draw else 'Q0'
        self.style.formatHisto(self.proj)
        self.proj.Fit('gaus', opt)
        func = self.proj.GetFunction('gaus')
        mean = Value( func.GetParameter(1),
                      func.GetParError(1))
        sigma = Value( func.GetParameter(2), 
                       func.GetParError(2))
        return mean, sigma

    def fit(self):
        self.hmean = self.h2d.ProjectionX().Clone('hmean')
        self.hmean.Reset()
        self.hmean.SetYTitle('relative scale')
        self.style.formatHisto(self.hmean)
        self.hsigma = self.h2d.ProjectionX().Clone('hsigma')
        self.hsigma.Reset()
        self.hsigma.SetYTitle('relative resolution')
        self.style.formatHisto(self.hsigma)
        for ibin in range(self.h2d.GetNbinsX()):
            ii = ibin+1
            mean, sigma = self.fit_slice(ii, False)
            if mean and sigma: 
                self.hmean.SetBinContent(ii, mean.val)
                self.hmean.SetBinError(ii, mean.err)
                self.hsigma.SetBinContent(ii, sigma.val)
                self.hsigma.SetBinError(ii, sigma.err)
        self.fit_res()
                
    def fit_res(self, draw=True):
        func = TF1(self.hname('res_calo'),
                   'sqrt( ([0]/sqrt(x))**2 + ([1]/x)**2 ) + [2]**2 ')
        # func.SetParLimits(0, 0., 9999999.)
        # func.SetParLimits(1, 0., 9999999.)
        # func.SetParLimits(2, 0., 9999999.)
        opt = '' if draw else 'Q0'
        self.hsigma.Fit(self.hname('res_calo'), opt)
        stoch = Value( func.GetParameter(0), func.GetParError(0))
        noise = Value( func.GetParameter(1), func.GetParError(1))
        const = Value( func.GetParameter(2), func.GetParError(2))
        print stoch, noise, const
        return stoch, noise, const
        
    def draw_2d(self):
        self.h2d.Draw('colz')
        self.hmean.Draw('same')
