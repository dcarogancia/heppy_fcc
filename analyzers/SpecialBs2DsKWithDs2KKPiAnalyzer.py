#!/usr/bin/env python

"""
    Analyzer of B0s -> K+ Ds- events (special for Stephane)
                           |-> K- K+ pi-

    Note: it is supposed to be used within heppy_fcc framework
"""

import math
import time

import numpy

from ROOT import gROOT, TFile, TH1F, TCanvas

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree
from heppy_fcc.utility.Particle import Particle

class SpecialBs2DsKWithDs2KKPiAnalyzer(Analyzer):
    """
        Analyzer of B0s -> K+ Ds- background events (special for Stephane)
                               |-> K- K+ pi-

        Attributes:
           rootfile (ROOT.TFile): output ROOT file
           tree (heppy.statistics.Tree): the tree with visible (smeared) values
           mc_truth_tree (heppy.statistics.Tree): the tree with MC truth values
           counter (int): total number of processed decays
           start_time (float): processing start time
           last_timestamp (float): last time check
    """

    def __init__(self, cfg_ana, cfg_comp, looper_name):
        """
            Constructor

            Arguments:
            cfg_ana: passed to the base class
            cfg_comp: passed to the base class
            looper_name: passed to the base class
        """

        super(SpecialBs2DsKWithDs2KKPiAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

        self.rootfile = TFile('/'.join([self.dirName, 'output.root']), 'recreate')

        # tree to store smeared values
        self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)
        self.tree.var('n_particles')
        self.tree.var('event_number')
        self.tree.var('pv_x')
        self.tree.var('pv_y')
        self.tree.var('pv_z')
        self.tree.var('tv_x')
        self.tree.var('tv_y')
        self.tree.var('tv_z')
        self.tree.var('k_px')
        self.tree.var('k_py')
        self.tree.var('k_pz')
        self.tree.var('k_q')
        self.tree.var('kplus_d_px')
        self.tree.var('kplus_d_py')
        self.tree.var('kplus_d_pz')
        self.tree.var('kminus_d_px')
        self.tree.var('kminus_d_py')
        self.tree.var('kminus_d_pz')
        self.tree.var('pi_d_px')
        self.tree.var('pi_d_py')
        self.tree.var('pi_d_pz')
        self.tree.var('pi_d_q')

        # MC truth tree
        self.mc_truth_tree = Tree(self.cfg_ana.mc_truth_tree_name, self.cfg_ana.mc_truth_tree_title)

        # MC truth values
        self.mc_truth_tree.var('n_particles')
        self.mc_truth_tree.var('event_number')
        self.mc_truth_tree.var('pv_x')
        self.mc_truth_tree.var('pv_y')
        self.mc_truth_tree.var('pv_z')
        self.mc_truth_tree.var('sv_x')
        self.mc_truth_tree.var('sv_y')
        self.mc_truth_tree.var('sv_z')
        self.mc_truth_tree.var('tv_x')
        self.mc_truth_tree.var('tv_y')
        self.mc_truth_tree.var('tv_z')
        self.mc_truth_tree.var('b_px')
        self.mc_truth_tree.var('b_py')
        self.mc_truth_tree.var('b_pz')
        self.mc_truth_tree.var('k_px')
        self.mc_truth_tree.var('k_py')
        self.mc_truth_tree.var('k_pz')
        self.mc_truth_tree.var('k_q')
        self.mc_truth_tree.var('d_px')
        self.mc_truth_tree.var('d_py')
        self.mc_truth_tree.var('d_pz')
        self.mc_truth_tree.var('d_q')
        self.mc_truth_tree.var('kplus_d_px')
        self.mc_truth_tree.var('kplus_d_py')
        self.mc_truth_tree.var('kplus_d_pz')
        self.mc_truth_tree.var('kminus_d_px')
        self.mc_truth_tree.var('kminus_d_py')
        self.mc_truth_tree.var('kminus_d_pz')
        self.mc_truth_tree.var('pi_d_px')
        self.mc_truth_tree.var('pi_d_py')
        self.mc_truth_tree.var('pi_d_pz')
        self.mc_truth_tree.var('pi_d_q')

        # statistics
        self.counter = 0 # Total number of processed decays

        gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

        # time
        self.start_time = None
        self.last_timestamp = None

    def beginLoop(self, setup):
        """
            Overriden base class function

            Initializes processing start time and the first timestamp

            Arguments:
            setup: passed to the base class function
        """

        self.start_time = time.time()
        self.last_timestamp = time.time()

        super(SpecialBs2DsKWithDs2KKPiAnalyzer, self).beginLoop(setup)

    def process(self, event):
        """
            Overriden base class function

            Processes the event

            Arguments:
            event: unused
        """

        b = None # Bs particle
        k = None # K from Bs decay
        d = None # Ds from Bs decay
        kplus_d = None # K+ from Ds decay
        kminus_d = None # K- from Ds decay
        pi_d = None # pi from Ds decay

        pv = None # primary vertex
        sv = None # secondary vertex
        tv = None # tertiary vertex

        event_info = event.input.get("EventInfo")
        particles_info = event.input.get("GenParticle")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        # looking for B
        for ptc_gen1 in ptcs:
            if abs(ptc_gen1.pdgid) == 531 and ptc_gen1.start_vertex != ptc_gen1.end_vertex: # if B0s found and it's not an oscillation
                self.counter += 1
                if self.counter % 100 == 0:
                    print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
                    self.last_timestamp = time.time()

                b = ptc_gen1
                pv = b.start_vertex
                sv = b.end_vertex
                for ptc_gen2 in ptcs:
                    if ptc_gen2.start_vertex == b.end_vertex:
                        # looking for Ds
                        if abs(ptc_gen2.pdgid) == 431:
                            d = ptc_gen2
                            tv = d.end_vertex

                        # looking for K
                        if abs(ptc_gen2.pdgid) == 321:
                            k = ptc_gen2

                for ptc_gen3 in ptcs:
                    if ptc_gen3.start_vertex == d.end_vertex:
                        # looking for K+
                        if ptc_gen3.pdgid == 321:
                            kplus_d = ptc_gen3

                        # looking for K-
                        if ptc_gen3.pdgid == -321:
                            kminus_d = ptc_gen3

                        # looking for pi
                        if abs(ptc_gen3.pdgid) == 211:
                            pi_d = ptc_gen3

                # filling MC truth information
                self.mc_truth_tree.fill('event_number', event_number)
                self.mc_truth_tree.fill('n_particles', n_particles)

                self.mc_truth_tree.fill('pv_x', pv.x)
                self.mc_truth_tree.fill('pv_y', pv.y)
                self.mc_truth_tree.fill('pv_z', pv.z)
                self.mc_truth_tree.fill('sv_x', sv.x)
                self.mc_truth_tree.fill('sv_y', sv.y)
                self.mc_truth_tree.fill('sv_z', sv.z)
                self.mc_truth_tree.fill('tv_x', tv.x)
                self.mc_truth_tree.fill('tv_y', tv.y)
                self.mc_truth_tree.fill('tv_z', tv.z)

                self.mc_truth_tree.fill('b_px', b.p.px)
                self.mc_truth_tree.fill('b_py', b.p.py)
                self.mc_truth_tree.fill('b_pz', b.p.pz)

                self.mc_truth_tree.fill('k_q', k.charge)
                self.mc_truth_tree.fill('k_px', k.p.px)
                self.mc_truth_tree.fill('k_py', k.p.py)
                self.mc_truth_tree.fill('k_pz', k.p.pz)

                self.mc_truth_tree.fill('d_px', d.p.px)
                self.mc_truth_tree.fill('d_py', d.p.py)
                self.mc_truth_tree.fill('d_pz', d.p.pz)

                self.mc_truth_tree.fill('kplus_d_px', kplus_d.p.px)
                self.mc_truth_tree.fill('kplus_d_py', kplus_d.p.py)
                self.mc_truth_tree.fill('kplus_d_pz', kplus_d.p.pz)

                self.mc_truth_tree.fill('kminus_d_px', kminus_d.p.px)
                self.mc_truth_tree.fill('kminus_d_py', kminus_d.p.py)
                self.mc_truth_tree.fill('kminus_d_pz', kminus_d.p.pz)

                self.mc_truth_tree.fill('pi_d_q', pi_d.charge)
                self.mc_truth_tree.fill('pi_d_px', pi_d.p.px)
                self.mc_truth_tree.fill('pi_d_py', pi_d.p.py)
                self.mc_truth_tree.fill('pi_d_pz', pi_d.p.pz)

                self.mc_truth_tree.tree.Fill()

                # filling event information
                self.tree.fill('event_number', event_number)
                self.tree.fill('n_particles', n_particles)

                self.tree.fill('pv_x', numpy.random.normal(pv.x, self.cfg_ana.pv_x_resolution) if self.cfg_ana.smear_pv else pv.x)
                self.tree.fill('pv_y', numpy.random.normal(pv.y, self.cfg_ana.pv_y_resolution) if self.cfg_ana.smear_pv else pv.y)
                self.tree.fill('pv_z', numpy.random.normal(pv.z, self.cfg_ana.pv_z_resolution) if self.cfg_ana.smear_pv else pv.z)
                self.tree.fill('tv_x', numpy.random.normal(tv.x, self.cfg_ana.tv_x_resolution) if self.cfg_ana.smear_tv else tv.x)
                self.tree.fill('tv_y', numpy.random.normal(tv.y, self.cfg_ana.tv_y_resolution) if self.cfg_ana.smear_tv else tv.y)
                self.tree.fill('tv_z', numpy.random.normal(tv.z, self.cfg_ana.tv_z_resolution) if self.cfg_ana.smear_tv else tv.z)

                self.tree.fill('k_q', k.charge)
                self.tree.fill('k_px', numpy.random.normal(k.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else k.p.px)
                self.tree.fill('k_py', numpy.random.normal(k.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else k.p.py)
                self.tree.fill('k_pz', numpy.random.normal(k.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else k.p.pz)

                self.tree.fill('kplus_d_px', numpy.random.normal(kplus_d.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else kplus_d.p.px)
                self.tree.fill('kplus_d_py', numpy.random.normal(kplus_d.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else kplus_d.p.py)
                self.tree.fill('kplus_d_pz', numpy.random.normal(kplus_d.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else kplus_d.p.pz)

                self.tree.fill('kminus_d_px', numpy.random.normal(kminus_d.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else kminus_d.p.px)
                self.tree.fill('kminus_d_py', numpy.random.normal(kminus_d.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else kminus_d.p.py)
                self.tree.fill('kminus_d_pz', numpy.random.normal(kminus_d.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else kminus_d.p.pz)

                self.tree.fill('pi_d_q', k.charge)
                self.tree.fill('pi_d_px', numpy.random.normal(pi_d.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi_d.p.px)
                self.tree.fill('pi_d_py', numpy.random.normal(pi_d.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi_d.p.py)
                self.tree.fill('pi_d_pz', numpy.random.normal(pi_d.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi_d.p.pz)

                self.tree.tree.Fill()

    def write(self, setup):
        """
            Overriden base class function

            Finalizes writing to file. Shows histograms. Prints some statistics

            Arguments:
            setup: unused
        """

        # finalizing writing to the file
        self.rootfile.Write()
        self.rootfile.Close()

        # some useful statistics
        print('Total decays processed: {}'.format(self.counter))
        print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
