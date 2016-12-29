#!/usr/bin/env/python

"""
    Analyzer of B0d -> K*0 Ds+ tau- nu events
                        |   |   |-> pi- pi- pi+ nu
                        |   |-> pi+ pi+ pi- pi0
                        |-> K+ pi-

    Note: it is supposed to be used within heppy_fcc framework
"""

import math
import time

import numpy

from heppy_fcc.utility.CommonAnalyzer import CommonAnalyzer
from heppy_fcc.utility.Particle import Particle

class BackgroundBd2DsKTauNuWithDs2PiPiPiPiAnalyzer(CommonAnalyzer):
    """
        Analyzer of B0d -> K*0 Ds+ tau- nu background events
                            |   |   |-> pi- pi- pi+ nu
                            |   |-> pi+ pi+ pi- pi0
                            |-> K+ pi-

        Inherits from heppy.framework.analyzer.Analyzer. Overrides base class methods to cover analysis-specific needs
    """

    def __init__(self, cfg_ana, cfg_comp, looper_name):
        """
            Constructor

            Arguments:
            cfg_ana: passed to the base class
            cfg_comp: passed to the base class
            looper_name: passed to the base class
        """

        super(BackgroundBd2DsKTauNuWithDs2PiPiPiPiAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

        self.mc_truth_tree.var('n_particles')
        self.mc_truth_tree.var('event_number')
        self.mc_truth_tree.var('pv_x')
        self.mc_truth_tree.var('pv_y')
        self.mc_truth_tree.var('pv_z')
        self.mc_truth_tree.var('sv_x')
        self.mc_truth_tree.var('sv_y')
        self.mc_truth_tree.var('sv_z')
        self.mc_truth_tree.var('tv_tau_x')
        self.mc_truth_tree.var('tv_tau_y')
        self.mc_truth_tree.var('tv_tau_z')
        self.mc_truth_tree.var('tv_d_x')
        self.mc_truth_tree.var('tv_d_y')
        self.mc_truth_tree.var('tv_d_z')
        self.mc_truth_tree.var('b_px')
        self.mc_truth_tree.var('b_py')
        self.mc_truth_tree.var('b_pz')
        self.mc_truth_tree.var('kstar_px')
        self.mc_truth_tree.var('kstar_py')
        self.mc_truth_tree.var('kstar_pz')
        self.mc_truth_tree.var('k_px')
        self.mc_truth_tree.var('k_py')
        self.mc_truth_tree.var('k_pz')
        self.mc_truth_tree.var('k_q')
        self.mc_truth_tree.var('pi_kstar_px')
        self.mc_truth_tree.var('pi_kstar_py')
        self.mc_truth_tree.var('pi_kstar_pz')
        self.mc_truth_tree.var('pi_kstar_q')
        self.mc_truth_tree.var('d_px')
        self.mc_truth_tree.var('d_py')
        self.mc_truth_tree.var('d_pz')
        self.mc_truth_tree.var('d_q')
        self.mc_truth_tree.var('pi1_d_px')
        self.mc_truth_tree.var('pi1_d_py')
        self.mc_truth_tree.var('pi1_d_pz')
        self.mc_truth_tree.var('pi1_d_q')
        self.mc_truth_tree.var('pi2_d_px')
        self.mc_truth_tree.var('pi2_d_py')
        self.mc_truth_tree.var('pi2_d_pz')
        self.mc_truth_tree.var('pi2_d_q')
        self.mc_truth_tree.var('pi3_d_px')
        self.mc_truth_tree.var('pi3_d_py')
        self.mc_truth_tree.var('pi3_d_pz')
        self.mc_truth_tree.var('pi3_d_q')
        self.mc_truth_tree.var('pi0_d_px')
        self.mc_truth_tree.var('pi0_d_py')
        self.mc_truth_tree.var('pi0_d_pz')
        self.mc_truth_tree.var('tau_px')
        self.mc_truth_tree.var('tau_py')
        self.mc_truth_tree.var('tau_pz')
        self.mc_truth_tree.var('tau_q')
        self.mc_truth_tree.var('pi1_tau_px')
        self.mc_truth_tree.var('pi1_tau_py')
        self.mc_truth_tree.var('pi1_tau_pz')
        self.mc_truth_tree.var('pi1_tau_q')
        self.mc_truth_tree.var('pi2_tau_px')
        self.mc_truth_tree.var('pi2_tau_py')
        self.mc_truth_tree.var('pi2_tau_pz')
        self.mc_truth_tree.var('pi2_tau_q')
        self.mc_truth_tree.var('pi3_tau_px')
        self.mc_truth_tree.var('pi3_tau_py')
        self.mc_truth_tree.var('pi3_tau_pz')
        self.mc_truth_tree.var('pi3_tau_q')
        self.mc_truth_tree.var('nu_tau_px')
        self.mc_truth_tree.var('nu_tau_py')
        self.mc_truth_tree.var('nu_tau_pz')
        self.mc_truth_tree.var('nu_px')
        self.mc_truth_tree.var('nu_py')
        self.mc_truth_tree.var('nu_pz')

    def process(self, event):
        """
            Overriden base class function

            Processes the event

            Arguments:
            event: unused
        """

        event_info = event.input.get("EventInfo")
        particles_info = event.input.get("GenParticle")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        # looking for B
        for ptc_gen1 in ptcs:
            if abs(ptc_gen1.pdgid) == 511 and ptc_gen1.start_vertex != ptc_gen1.end_vertex: # if B0d found and it's not an oscillation
                self.counter += 1
                if self.counter % 100 == 0:
                    print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
                    self.last_timestamp = time.time()

                b = None # B0d particle
                kstar = None # K*0 from B0d decay
                k = None # K from K*0 decay
                pi_kstar = None # pi from K* decay
                tau = None # tau from B0d decay
                pi1_tau = None # pi from tau+ decay
                pi2_tau = None # pi from tau+ decay
                pi3_tau = None # pi from tau+ decay
                nu_tau = None # nu from B0d tau decay
                nu = None # nu from Ds decay
                d = None # Ds from B0d decay
                pi1_d = None  # pi from Ds decay
                pi2_d = None  # pi from Ds decay
                pi3_d = None  # pi from Ds decay
                pi0_d = None # pi0 from Ds decay

                pv = None # primary vertex
                sv = None # secondary vertex
                tv_tau = None # tau decay vertex
                tv_d = None # Ds vertex

                pvsv_distance = 0. # distance between PV and SV
                pb = 0. # B momentum
                max_svtv_distance = 0. # maximal distance between SV and TV

                if abs(ptc_gen1.p.pz) / ptc_gen1.p.absvalue() <= math.cos(math.pi / 12):
                    b = ptc_gen1

                    pb = b.p.absvalue()

                    if pb > 25.: # select only events with large momentum of the B
                        self.pb_counter += 1

                        pv = b.start_vertex
                        sv = b.end_vertex
                        pvsv_distance = math.sqrt((sv.x - pv.x) ** 2 + (sv.y - pv.y) ** 2 + (sv.z - pv.z) ** 2)

                        if pvsv_distance > 1.: # select only events with long flight distance of the B
                            self.pvsv_distance_counter += 1

                            for ptc_gen2 in ptcs:
                                if ptc_gen2.start_vertex == b.end_vertex and abs(ptc_gen2.p.pz) / ptc_gen2.p.absvalue() <= math.cos(math.pi / 12):
                                    # looking for K*
                                    if abs(ptc_gen2.pdgid) == 313:
                                        kstar = ptc_gen2

                                        for ptc_gen3 in ptcs:
                                            if ptc_gen3.start_vertex == kstar.end_vertex and abs(ptc_gen3.p.pz) / ptc_gen3.p.absvalue() <= math.cos(math.pi / 12):
                                                # looking for K
                                                if abs(ptc_gen3.pdgid) == 321:
                                                    k = ptc_gen3

                                                # looking for pi
                                                if abs(ptc_gen3.pdgid) == 211:
                                                    pi_kstar = ptc_gen3

                                    # looking for Ds
                                    if abs(ptc_gen2.pdgid) == 431:
                                        d = ptc_gen2
                                        tv_d = d.end_vertex

                                        pis_d = []
                                        for ptc_gen3 in ptcs:
                                            if ptc_gen3.start_vertex == d.end_vertex and abs(ptc_gen3.p.pz) / ptc_gen3.p.absvalue() <= math.cos(math.pi / 12):
                                                # looking for pi+/-
                                                if abs(ptc_gen3.pdgid) == 211:
                                                    pis_d.append(ptc_gen3)

                                                # looking for pi0
                                                if abs(ptc_gen3.pdgid) == 111:
                                                    pi0_d = ptc_gen3

                                        if len(pis_d) == 3:
                                            pi1_d, pi2_d, pi3_d = pis_d[0], pis_d[1], pis_d[2]

                                    # looking for tau
                                    if abs(ptc_gen2.pdgid) == 15:
                                        tau = ptc_gen2
                                        tv_tau = tau.end_vertex

                                        pis_tau = []
                                        for ptc_gen3 in ptcs:
                                            if ptc_gen3.start_vertex == tau.end_vertex and abs(ptc_gen3.p.pz) / ptc_gen3.p.absvalue() <= math.cos(math.pi / 12):
                                                # looking for pi+/-
                                                if abs(ptc_gen3.pdgid) == 211:
                                                    pis_tau.append(ptc_gen3)

                                                # looking for nu
                                                if abs(ptc_gen3.pdgid) == 16:
                                                    nu_tau = ptc_gen3
                                        if len(pis_tau) == 3:
                                            pi1_tau, pi2_tau, pi3_tau = pis_tau[0], pis_tau[1], pis_tau[2]

                                    # looking for nu
                                    if abs(ptc_gen2.pdgid) == 16:
                                        nu = ptc_gen2

                            if b is not None and kstar is not None and k is not None and pi_kstar is not None and tau is not None and pi1_tau is not None and pi2_tau is not None and pi3_tau is not None and nu_tau is not None and nu is not None and d is not None and pi1_d is not None and pi2_d is not None and pi3_d is not None and pi0_d is not None:
                                max_svtv_distance = max(math.sqrt((tv_tau.x - sv.x) ** 2 + (tv_tau.y - sv.y) ** 2 + (tv_tau.z - sv.z) ** 2), math.sqrt((tv_d.x - sv.x) ** 2 + (tv_d.y - sv.y) ** 2 + (tv_d.z - sv.z) ** 2))

                                if max_svtv_distance > 0.5: # select only events with long flight distance of tau
                                    self.max_svtv_distance_counter += 1

                                    # filling histograms
                                    self.pvsv_distance_hist.Fill(pvsv_distance)
                                    self.pb_hist.Fill(pb)
                                    self.max_svtv_distance_hist.Fill(max_svtv_distance)

                                    # filling MC truth information
                                    self.mc_truth_tree.fill('event_number', event_number)
                                    self.mc_truth_tree.fill('n_particles', n_particles)

                                    self.mc_truth_tree.fill('pv_x', pv.x)
                                    self.mc_truth_tree.fill('pv_y', pv.y)
                                    self.mc_truth_tree.fill('pv_z', pv.z)
                                    self.mc_truth_tree.fill('sv_x', sv.x)
                                    self.mc_truth_tree.fill('sv_y', sv.y)
                                    self.mc_truth_tree.fill('sv_z', sv.z)
                                    self.mc_truth_tree.fill('tv_tau_x', tv_tau.x)
                                    self.mc_truth_tree.fill('tv_tau_y', tv_tau.y)
                                    self.mc_truth_tree.fill('tv_tau_z', tv_tau.z)
                                    self.mc_truth_tree.fill('tv_d_x', tv_d.x)
                                    self.mc_truth_tree.fill('tv_d_y', tv_d.y)
                                    self.mc_truth_tree.fill('tv_d_z', tv_d.z)

                                    self.mc_truth_tree.fill('b_px', b.p.px)
                                    self.mc_truth_tree.fill('b_py', b.p.py)
                                    self.mc_truth_tree.fill('b_pz', b.p.pz)

                                    self.mc_truth_tree.fill('kstar_px', kstar.p.px)
                                    self.mc_truth_tree.fill('kstar_py', kstar.p.py)
                                    self.mc_truth_tree.fill('kstar_pz', kstar.p.pz)

                                    self.mc_truth_tree.fill('k_q', k.charge)
                                    self.mc_truth_tree.fill('k_px', k.p.px)
                                    self.mc_truth_tree.fill('k_py', k.p.py)
                                    self.mc_truth_tree.fill('k_pz', k.p.pz)

                                    self.mc_truth_tree.fill('pi_kstar_q', pi_kstar.charge)
                                    self.mc_truth_tree.fill('pi_kstar_px', pi_kstar.p.px)
                                    self.mc_truth_tree.fill('pi_kstar_py', pi_kstar.p.py)
                                    self.mc_truth_tree.fill('pi_kstar_pz', pi_kstar.p.pz)

                                    self.mc_truth_tree.fill('d_q', d.charge)
                                    self.mc_truth_tree.fill('d_px', d.p.px)
                                    self.mc_truth_tree.fill('d_py', d.p.py)
                                    self.mc_truth_tree.fill('d_pz', d.p.pz)

                                    self.mc_truth_tree.fill('pi1_d_q', pi1_d.charge)
                                    self.mc_truth_tree.fill('pi1_d_px', pi1_d.p.px)
                                    self.mc_truth_tree.fill('pi1_d_py', pi1_d.p.py)
                                    self.mc_truth_tree.fill('pi1_d_pz', pi1_d.p.pz)

                                    self.mc_truth_tree.fill('pi2_d_q', pi2_d.charge)
                                    self.mc_truth_tree.fill('pi2_d_px', pi2_d.p.px)
                                    self.mc_truth_tree.fill('pi2_d_py', pi2_d.p.py)
                                    self.mc_truth_tree.fill('pi2_d_pz', pi2_d.p.pz)

                                    self.mc_truth_tree.fill('pi3_d_q', pi3_d.charge)
                                    self.mc_truth_tree.fill('pi3_d_px', pi3_d.p.px)
                                    self.mc_truth_tree.fill('pi3_d_py', pi3_d.p.py)
                                    self.mc_truth_tree.fill('pi3_d_pz', pi3_d.p.pz)

                                    self.mc_truth_tree.fill('pi0_d_px', pi0_d.p.px)
                                    self.mc_truth_tree.fill('pi0_d_py', pi0_d.p.py)
                                    self.mc_truth_tree.fill('pi0_d_pz', pi0_d.p.pz)

                                    self.mc_truth_tree.fill('tau_q', tau.charge)
                                    self.mc_truth_tree.fill('tau_px', tau.p.px)
                                    self.mc_truth_tree.fill('tau_py', tau.p.py)
                                    self.mc_truth_tree.fill('tau_pz', tau.p.pz)

                                    self.mc_truth_tree.fill('pi1_tau_q', pi1_tau.charge)
                                    self.mc_truth_tree.fill('pi1_tau_px', pi1_tau.p.px)
                                    self.mc_truth_tree.fill('pi1_tau_py', pi1_tau.p.py)
                                    self.mc_truth_tree.fill('pi1_tau_pz', pi1_tau.p.pz)

                                    self.mc_truth_tree.fill('pi2_tau_q', pi2_tau.charge)
                                    self.mc_truth_tree.fill('pi2_tau_px', pi2_tau.p.px)
                                    self.mc_truth_tree.fill('pi2_tau_py', pi2_tau.p.py)
                                    self.mc_truth_tree.fill('pi2_tau_pz', pi2_tau.p.pz)

                                    self.mc_truth_tree.fill('pi3_tau_q', pi3_tau.charge)
                                    self.mc_truth_tree.fill('pi3_tau_px', pi3_tau.p.px)
                                    self.mc_truth_tree.fill('pi3_tau_py', pi3_tau.p.py)
                                    self.mc_truth_tree.fill('pi3_tau_pz', pi3_tau.p.pz)

                                    self.mc_truth_tree.fill('nu_tau_px', nu_tau.p.px)
                                    self.mc_truth_tree.fill('nu_tau_py', nu_tau.p.py)
                                    self.mc_truth_tree.fill('nu_tau_pz', nu_tau.p.pz)

                                    self.mc_truth_tree.fill('nu_px', nu.p.px)
                                    self.mc_truth_tree.fill('nu_py', nu.p.py)
                                    self.mc_truth_tree.fill('nu_pz', nu.p.pz)

                                    self.mc_truth_tree.tree.Fill()

                                    # matching visible particles and MC truth ones
                                    if tau.charge < 0:
                                        tv_tauminus = tv_tau
                                        pi1_tauminus = pi1_tau
                                        pi2_tauminus = pi2_tau
                                        pi3_tauminus = pi3_tau
                                    else:
                                        tv_tauplus = tv_tau
                                        pi1_tauplus = pi1_tau
                                        pi2_tauplus = pi2_tau
                                        pi3_tauplus = pi3_tau

                                    if d.charge < 0:
                                        tv_tauminus = tv_d
                                        pi1_tauminus = pi1_d
                                        pi2_tauminus = pi2_d
                                        pi3_tauminus = pi3_d
                                    else:
                                        tv_tauplus = tv_d
                                        pi1_tauplus = pi1_d
                                        pi2_tauplus = pi2_d
                                        pi3_tauplus = pi3_d

                                    # filling event information
                                    self.tree.fill('event_number', event_number)
                                    self.tree.fill('n_particles', n_particles)

                                    self.tree.fill('pv_x', numpy.random.normal(pv.x, self.cfg_ana.pv_x_resolution) if self.cfg_ana.smear_pv else pv.x)
                                    self.tree.fill('pv_y', numpy.random.normal(pv.y, self.cfg_ana.pv_y_resolution) if self.cfg_ana.smear_pv else pv.y)
                                    self.tree.fill('pv_z', numpy.random.normal(pv.z, self.cfg_ana.pv_z_resolution) if self.cfg_ana.smear_pv else pv.z)
                                    self.tree.fill('sv_x', numpy.random.normal(sv.x, self.cfg_ana.sv_x_resolution) if self.cfg_ana.smear_sv else sv.x)
                                    self.tree.fill('sv_y', numpy.random.normal(sv.y, self.cfg_ana.sv_y_resolution) if self.cfg_ana.smear_sv else sv.y)
                                    self.tree.fill('sv_z', numpy.random.normal(sv.z, self.cfg_ana.sv_z_resolution) if self.cfg_ana.smear_sv else sv.z)
                                    self.tree.fill('tv_tauplus_x', numpy.random.normal(tv_tauplus.x, self.cfg_ana.tv_x_resolution) if self.cfg_ana.smear_tv else tv_tauplus.x)
                                    self.tree.fill('tv_tauplus_y', numpy.random.normal(tv_tauplus.y, self.cfg_ana.tv_y_resolution) if self.cfg_ana.smear_tv else tv_tauplus.y)
                                    self.tree.fill('tv_tauplus_z', numpy.random.normal(tv_tauplus.z, self.cfg_ana.tv_z_resolution) if self.cfg_ana.smear_tv else tv_tauplus.z)
                                    self.tree.fill('tv_tauminus_x', numpy.random.normal(tv_tauminus.x, self.cfg_ana.tv_x_resolution) if self.cfg_ana.smear_tv else tv_tauminus.x)
                                    self.tree.fill('tv_tauminus_y', numpy.random.normal(tv_tauminus.y, self.cfg_ana.tv_y_resolution) if self.cfg_ana.smear_tv else tv_tauminus.y)
                                    self.tree.fill('tv_tauminus_z', numpy.random.normal(tv_tauminus.z, self.cfg_ana.tv_z_resolution) if self.cfg_ana.smear_tv else tv_tauminus.z)

                                    self.tree.fill('pi1_tauplus_q', pi1_tauplus.charge)
                                    self.tree.fill('pi1_tauplus_px', self.cfg_ana.momentum_smearer(pi1_tauplus.p.px, pi1_tauplus.p) if self.cfg_ana.smear_momentum else pi1_tauplus.p.px)
                                    self.tree.fill('pi1_tauplus_py', self.cfg_ana.momentum_smearer(pi1_tauplus.p.py, pi1_tauplus.p) if self.cfg_ana.smear_momentum else pi1_tauplus.p.py)
                                    self.tree.fill('pi1_tauplus_pz', self.cfg_ana.momentum_smearer(pi1_tauplus.p.pz, pi1_tauplus.p) if self.cfg_ana.smear_momentum else pi1_tauplus.p.pz)

                                    self.tree.fill('pi2_tauplus_q', pi2_tauplus.charge)
                                    self.tree.fill('pi2_tauplus_px', self.cfg_ana.momentum_smearer(pi2_tauplus.p.px, pi2_tauplus.p) if self.cfg_ana.smear_momentum else pi2_tauplus.p.px)
                                    self.tree.fill('pi2_tauplus_py', self.cfg_ana.momentum_smearer(pi2_tauplus.p.py, pi2_tauplus.p) if self.cfg_ana.smear_momentum else pi2_tauplus.p.py)
                                    self.tree.fill('pi2_tauplus_pz', self.cfg_ana.momentum_smearer(pi2_tauplus.p.pz, pi2_tauplus.p) if self.cfg_ana.smear_momentum else pi2_tauplus.p.pz)

                                    self.tree.fill('pi3_tauplus_q', pi3_tauplus.charge)
                                    self.tree.fill('pi3_tauplus_px', self.cfg_ana.momentum_smearer(pi3_tauplus.p.px, pi3_tauplus.p) if self.cfg_ana.smear_momentum else pi3_tauplus.p.px)
                                    self.tree.fill('pi3_tauplus_py', self.cfg_ana.momentum_smearer(pi3_tauplus.p.px, pi3_tauplus.p) if self.cfg_ana.smear_momentum else pi3_tauplus.p.py)
                                    self.tree.fill('pi3_tauplus_pz', self.cfg_ana.momentum_smearer(pi3_tauplus.p.px, pi3_tauplus.p) if self.cfg_ana.smear_momentum else pi3_tauplus.p.pz)

                                    self.tree.fill('pi1_tauminus_q', pi1_tauminus.charge)
                                    self.tree.fill('pi1_tauminus_px', self.cfg_ana.momentum_smearer(pi1_tauminus.p.px, pi1_tauminus.p) if self.cfg_ana.smear_momentum else pi1_tauminus.p.px)
                                    self.tree.fill('pi1_tauminus_py', self.cfg_ana.momentum_smearer(pi1_tauminus.p.py, pi1_tauminus.p) if self.cfg_ana.smear_momentum else pi1_tauminus.p.py)
                                    self.tree.fill('pi1_tauminus_pz', self.cfg_ana.momentum_smearer(pi1_tauminus.p.pz, pi1_tauminus.p) if self.cfg_ana.smear_momentum else pi1_tauminus.p.pz)

                                    self.tree.fill('pi2_tauminus_q', pi2_tauminus.charge)
                                    self.tree.fill('pi2_tauminus_px', self.cfg_ana.momentum_smearer(pi2_tauminus.p.px, pi2_tauminus.p) if self.cfg_ana.smear_momentum else pi2_tauminus.p.px)
                                    self.tree.fill('pi2_tauminus_py', self.cfg_ana.momentum_smearer(pi2_tauminus.p.py, pi2_tauminus.p) if self.cfg_ana.smear_momentum else pi2_tauminus.p.py)
                                    self.tree.fill('pi2_tauminus_pz', self.cfg_ana.momentum_smearer(pi2_tauminus.p.pz, pi2_tauminus.p) if self.cfg_ana.smear_momentum else pi2_tauminus.p.pz)

                                    self.tree.fill('pi3_tauminus_q', pi3_tauminus.charge)
                                    self.tree.fill('pi3_tauminus_px', self.cfg_ana.momentum_smearer(pi3_tauminus.p.px, pi3_tauminus.p) if self.cfg_ana.smear_momentum else pi3_tauminus.p.px)
                                    self.tree.fill('pi3_tauminus_py', self.cfg_ana.momentum_smearer(pi3_tauminus.p.py, pi3_tauminus.p) if self.cfg_ana.smear_momentum else pi3_tauminus.p.py)
                                    self.tree.fill('pi3_tauminus_pz', self.cfg_ana.momentum_smearer(pi3_tauminus.p.pz, pi3_tauminus.p) if self.cfg_ana.smear_momentum else pi3_tauminus.p.pz)

                                    self.tree.fill('k_q', k.charge)
                                    self.tree.fill('k_px', self.cfg_ana.momentum_smearer(k.p.px, k.p) if self.cfg_ana.smear_momentum else k.p.px)
                                    self.tree.fill('k_py', self.cfg_ana.momentum_smearer(k.p.py, k.p) if self.cfg_ana.smear_momentum else k.p.py)
                                    self.tree.fill('k_pz', self.cfg_ana.momentum_smearer(k.p.pz, k.p) if self.cfg_ana.smear_momentum else k.p.pz)

                                    self.tree.fill('pi_kstar_q', pi_kstar.charge)
                                    self.tree.fill('pi_kstar_px', self.cfg_ana.momentum_smearer(pi_kstar.p.px, pi_kstar.p) if self.cfg_ana.smear_momentum else pi_kstar.p.px)
                                    self.tree.fill('pi_kstar_py', self.cfg_ana.momentum_smearer(pi_kstar.p.py, pi_kstar.p) if self.cfg_ana.smear_momentum else pi_kstar.p.py)
                                    self.tree.fill('pi_kstar_pz', self.cfg_ana.momentum_smearer(pi_kstar.p.pz, pi_kstar.p) if self.cfg_ana.smear_momentum else pi_kstar.p.pz)

                                    self.tree.tree.Fill()
