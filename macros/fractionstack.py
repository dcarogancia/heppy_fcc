from cpyroot import * 

class FractionStack(object):

    def __init__(self, pdgids, hist):
        self.name = hist.GetName()
        self.hists = dict()
        self.pdgids= pdgids
        self.histsum = hist
        self.stack = THStack('_'.join([self.name, 'stack']),
                             hist.GetTitle());
        colors = [2, 4, 8, 1, 1]
        for iid, pdgid in enumerate(self.pdgids):
            hname = '_'.join([self.name, str(pdgid)])
            hist = hist.Clone(hname)
            self.hists[pdgid] = hist
            hist.SetFillColor(colors[iid])
            hist.SetFillStyle(1001)
            self.stack.Add(self.hists[pdgid])
        self.histsum.SetLineWidth(1)
        self.histsum.SetLineColor(1)
            
    def Project(self, tree, var, basecut='1'):
        tree.Project(self.histsum.GetName(), var, basecut)
        same = 'same'
        for pdgid in self.pdgids:
            jet = var.split('_')[0]
            compe = '_'.join([jet, str(pdgid), 'e'])
            jete = '_'.join([jet, 'e'])
            frac = '/'.join([compe, jete])
            cutfrac = '{frac}>0'.format(frac=frac)
            cut = '{frac}*({basecut} && {cutfrac})'.format(
                basecut = basecut,
                cutfrac = cutfrac,
                frac = frac
            )
            print 'var', var
            print 'cut', cut
            print self.hists[pdgid].GetName()
            tree.Project(self.hists[pdgid].GetName(), var, cut)
            # self.hists[pdgid].Draw(same)
            # if same == '':
            #     same = 'same'
        self.Draw()

    def Draw(self):
        self.histsum.Draw('hist')
        self.stack.Draw("histsame")
        self.histsum.Draw('histsame')
        gPad.RedrawAxis()

    def SetYRange(self, min, max):
        self.histsum.GetYaxis().SetRangeUser(min, max)
        self.Draw()

