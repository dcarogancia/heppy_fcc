from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree
from heppy_fcc.analyzers.ntuple import *

from ROOT import TFile

class SimpleTreeProducer(Analyzer):

    def beginLoop(self, setup):
        super(SimpleTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'tree.root']),
                              'recreate')
        self.tree = Tree( 'events', '')
        bookParticle(self.tree, 'recoil_gen')
        bookParticle(self.tree, 'recoil_visible_gen')
        bookParticle(self.tree, 'recoil_papas')
        bookParticle(self.tree, 'recoil_visible_papas')

        
    def process(self, event):
        self.tree.reset()
        fillParticle(self.tree, 'recoil_gen', event.recoil_gen)
        fillParticle(self.tree, 'recoil_visible_gen', event.recoil_visible_gen)
        if hasattr(event, 'recoil_papas'):
            fillParticle(self.tree, 'recoil_papas', event.recoil_papas)
            fillParticle(self.tree, 'recoil_visible_papas', event.recoil_visible_papas)
        self.tree.tree.Fill()

        
    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
