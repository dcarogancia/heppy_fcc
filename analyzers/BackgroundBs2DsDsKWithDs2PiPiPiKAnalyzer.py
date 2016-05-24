#!/usr/bin/env python

"""
	Analyzer of B0d -> K*0 Ds+ Ds- events
	                    |   |   |-> pi- pi- pi+ K0L
						|   |-> pi+ pi+ pi- K0L
						|->	K+ pi-

	Note: it is supposed to be used within heppy_fcc framework
"""

import math
import time

import numpy

from heppy_fcc.utility.CommonAnalyzer import CommonAnalyzer
from heppy_fcc.utility.Particle import Particle

class BackgroundBs2DsDsKWithDs2PiPiPiKAnalyzer(CommonAnalyzer):
	"""
		Analyzer of B0d -> K*0 Ds+ Ds- background events
		                    |   |   |-> pi- pi- pi+ K0L
							|   |-> pi+ pi+ pi- K0L
							|->	K+ pi-

		Inherits from heppy_fcc.utility.CommonAnalyzer. Extends the base class to cover analysis-specific needs
	"""

	def __init__(self, cfg_ana, cfg_comp, looper_name):
		"""
			Constructor

			Arguments:
			cfg_ana: passed to the base class
			cfg_comp: passed to the base class
			looper_name: passed to the base class
		"""

		super(BackgroundBs2DsDsKWithDs2PiPiPiKAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

		# MC truth values
		self.mc_truth_tree.var('n_particles')
		self.mc_truth_tree.var('event_number')
		self.mc_truth_tree.var('pv_x')
		self.mc_truth_tree.var('pv_y')
		self.mc_truth_tree.var('pv_z')
		self.mc_truth_tree.var('sv_x')
		self.mc_truth_tree.var('sv_y')
		self.mc_truth_tree.var('sv_z')
		self.mc_truth_tree.var('tv_dplus_x')
		self.mc_truth_tree.var('tv_dplus_y')
		self.mc_truth_tree.var('tv_dplus_z')
		self.mc_truth_tree.var('tv_dminus_x')
		self.mc_truth_tree.var('tv_dminus_y')
		self.mc_truth_tree.var('tv_dminus_z')
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
		self.mc_truth_tree.var('dplus_px')
		self.mc_truth_tree.var('dplus_py')
		self.mc_truth_tree.var('dplus_pz')
		self.mc_truth_tree.var('pi1_dplus_px')
		self.mc_truth_tree.var('pi1_dplus_py')
		self.mc_truth_tree.var('pi1_dplus_pz')
		self.mc_truth_tree.var('pi1_dplus_q')
		self.mc_truth_tree.var('pi2_dplus_px')
		self.mc_truth_tree.var('pi2_dplus_py')
		self.mc_truth_tree.var('pi2_dplus_pz')
		self.mc_truth_tree.var('pi2_dplus_q')
		self.mc_truth_tree.var('pi3_dplus_px')
		self.mc_truth_tree.var('pi3_dplus_py')
		self.mc_truth_tree.var('pi3_dplus_pz')
		self.mc_truth_tree.var('pi3_dplus_q')
		self.mc_truth_tree.var('k0_dplus_px')
		self.mc_truth_tree.var('k0_dplus_py')
		self.mc_truth_tree.var('k0_dplus_pz')
		self.mc_truth_tree.var('dminus_px')
		self.mc_truth_tree.var('dminus_py')
		self.mc_truth_tree.var('dminus_pz')
		self.mc_truth_tree.var('pi1_dminus_px')
		self.mc_truth_tree.var('pi1_dminus_py')
		self.mc_truth_tree.var('pi1_dminus_pz')
		self.mc_truth_tree.var('pi1_dminus_q')
		self.mc_truth_tree.var('pi2_dminus_px')
		self.mc_truth_tree.var('pi2_dminus_py')
		self.mc_truth_tree.var('pi2_dminus_pz')
		self.mc_truth_tree.var('pi2_dminus_q')
		self.mc_truth_tree.var('pi3_dminus_px')
		self.mc_truth_tree.var('pi3_dminus_py')
		self.mc_truth_tree.var('pi3_dminus_pz')
		self.mc_truth_tree.var('pi3_dminus_q')
		self.mc_truth_tree.var('k0_dminus_px')
		self.mc_truth_tree.var('k0_dminus_py')
		self.mc_truth_tree.var('k0_dminus_pz')

	def process(self, event):
		"""
            Overriden base class function

			Processes the event

            Arguments:
			event: unused
        """

		b = None # B particle
		kstar = None # K* from B decay
		k = None # K from K* decay
		pi_kstar = None # pi from K* decay
		dplus= None # Ds+ from Bs decay
		dminus = None # Ds- from Bs decay
		pi1_dplus = None # pi from Ds+ decay
		pi2_dplus = None # pi from Ds+ decay
		pi3_dplus = None # pi from Ds+ decay
		k0_dplus = None # pi0 from Ds+ decay
		pi1_dminus = None # pi from Ds- decay
		pi2_dminus = None # pi from Ds- decay
		pi3_dminus = None # pi from Ds- decay
		k0_dminus = None # pi0 from Ds- decay

		pv = None # primary vertex
		sv = None # secondary vertex
		tv_dplus = None # tau+ decay vertex
		tv_dminus = None # tau- decay vertex

		pvsv_distance = 0. # distance between PV and SV
		pb = 0. # B momentum
		max_svtv_distance = 0. # maximal distance between SV and TV

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

				pb = b.p.absvalue()

				if pb > 25.: # Select only events with large momentum of the B
					self.pb_counter += 1

					pv = b.start_vertex
					sv = b.end_vertex
					pvsv_distance = math.sqrt((sv.x - pv.x) ** 2 + (sv.y - pv.y) ** 2 + (sv.z - pv.z) ** 2)

					if pvsv_distance > 1.: # Select only events with long flight distance of the B
						self.pvsv_distance_counter += 1

						for ptc_gen2 in ptcs:
							if ptc_gen2.start_vertex == b.end_vertex:
								# looking for Ds+
								if ptc_gen2.pdgid == 431:
									dplus = ptc_gen2
									tv_dplus = dplus.end_vertex

								# looking for Ds-
								if ptc_gen2.pdgid == -431:
									dminus = ptc_gen2
									tv_dminus = dminus.end_vertex

								# looking for K*
								if abs(ptc_gen2.pdgid) == 313:
									kstar = ptc_gen2

						max_svtv_distance = max(math.sqrt((tv_dplus.x - sv.x) ** 2 + (tv_dplus.y - sv.y) ** 2 + (tv_dplus.z - sv.z) ** 2), math.sqrt((tv_dminus.x - sv.x) ** 2 + (tv_dminus.y - sv.y) ** 2 + (tv_dminus.z - sv.z) ** 2))

						if max_svtv_distance > 0.5: # select only events with long flight distance of tau
							self.max_svtv_distance_counter += 1

							pis_dplus = []
							pis_dminus = []

							for ptc_gen3 in ptcs:
								if ptc_gen3.start_vertex == kstar.end_vertex:
									# looking for K
									if abs(ptc_gen3.pdgid) == 321:
										k = ptc_gen3

									# looking for pi
									if abs(ptc_gen3.pdgid) == 211:
										pi_kstar = ptc_gen3

								if ptc_gen3.start_vertex == dplus.end_vertex:
									# looking for pi+/-
									if abs(ptc_gen3.pdgid) == 211:
										pis_dplus.append(ptc_gen3)

									# looking for K0L
									if ptc_gen3.pdgid == 130:
										k0_dplus = ptc_gen3

								if ptc_gen3.start_vertex == dminus.end_vertex:
									# looking for pi+/-
									if abs(ptc_gen3.pdgid) == 211:
										pis_dminus.append(ptc_gen3)

									# looking for K0L
									if ptc_gen3.pdgid == 130:
										k0_dminus = ptc_gen3


							if len(pis_dplus) == 3:
								pi1_dplus, pi2_dplus, pi3_dplus = pis_dplus[0], pis_dplus[1], pis_dplus[2]

							if len(pis_dminus) == 3:
								pi1_dminus, pi2_dminus, pi3_dminus = pis_dminus[0], pis_dminus[1], pis_dminus[2]

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
							self.mc_truth_tree.fill('tv_dplus_x', tv_dplus.x)
							self.mc_truth_tree.fill('tv_dplus_y', tv_dplus.y)
							self.mc_truth_tree.fill('tv_dplus_z', tv_dplus.z)
							self.mc_truth_tree.fill('tv_dminus_x', tv_dminus.x)
							self.mc_truth_tree.fill('tv_dminus_y', tv_dminus.y)
							self.mc_truth_tree.fill('tv_dminus_z', tv_dminus.z)

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

							self.mc_truth_tree.fill('dplus_px', dplus.p.px)
							self.mc_truth_tree.fill('dplus_py', dplus.p.py)
							self.mc_truth_tree.fill('dplus_pz', dplus.p.pz)

							self.mc_truth_tree.fill('pi1_dplus_q', pi1_dplus.charge)
							self.mc_truth_tree.fill('pi1_dplus_px', pi1_dplus.p.px)
							self.mc_truth_tree.fill('pi1_dplus_py', pi1_dplus.p.py)
							self.mc_truth_tree.fill('pi1_dplus_pz', pi1_dplus.p.pz)

							self.mc_truth_tree.fill('pi2_dplus_q', pi2_dplus.charge)
							self.mc_truth_tree.fill('pi2_dplus_px', pi2_dplus.p.px)
							self.mc_truth_tree.fill('pi2_dplus_py', pi2_dplus.p.py)
							self.mc_truth_tree.fill('pi2_dplus_pz', pi2_dplus.p.pz)

							self.mc_truth_tree.fill('pi3_dplus_q', pi3_dplus.charge)
							self.mc_truth_tree.fill('pi3_dplus_px', pi3_dplus.p.px)
							self.mc_truth_tree.fill('pi3_dplus_py', pi3_dplus.p.py)
							self.mc_truth_tree.fill('pi3_dplus_pz', pi3_dplus.p.pz)

							self.mc_truth_tree.fill('k0_dplus_px', k0_dplus.p.px)
							self.mc_truth_tree.fill('k0_dplus_py', k0_dplus.p.py)
							self.mc_truth_tree.fill('k0_dplus_pz', k0_dplus.p.pz)

							self.mc_truth_tree.fill('dminus_px', dminus.p.px)
							self.mc_truth_tree.fill('dminus_py', dminus.p.py)
							self.mc_truth_tree.fill('dminus_pz', dminus.p.pz)

							self.mc_truth_tree.fill('pi1_dminus_q', pi1_dminus.charge)
							self.mc_truth_tree.fill('pi1_dminus_px', pi1_dminus.p.px)
							self.mc_truth_tree.fill('pi1_dminus_py', pi1_dminus.p.py)
							self.mc_truth_tree.fill('pi1_dminus_pz', pi1_dminus.p.pz)

							self.mc_truth_tree.fill('pi2_dminus_q', pi2_dminus.charge)
							self.mc_truth_tree.fill('pi2_dminus_px', pi2_dminus.p.px)
							self.mc_truth_tree.fill('pi2_dminus_py', pi2_dminus.p.py)
							self.mc_truth_tree.fill('pi2_dminus_pz', pi2_dminus.p.pz)

							self.mc_truth_tree.fill('pi3_dminus_q', pi3_dminus.charge)
							self.mc_truth_tree.fill('pi3_dminus_px', pi3_dminus.p.px)
							self.mc_truth_tree.fill('pi3_dminus_py', pi3_dminus.p.py)
							self.mc_truth_tree.fill('pi3_dminus_pz', pi3_dminus.p.pz)

							self.mc_truth_tree.fill('k0_dminus_px', k0_dminus.p.px)
							self.mc_truth_tree.fill('k0_dminus_py', k0_dminus.p.py)
							self.mc_truth_tree.fill('k0_dminus_pz', k0_dminus.p.pz)

							self.mc_truth_tree.tree.Fill()

							# matching visible particles and MC truth ones
							tv_tauplus = tv_dplus
							pi1_tauplus = pi1_dplus
							pi2_tauplus = pi2_dplus
							pi3_tauplus = pi3_dplus

							tv_tauminus = tv_dminus
							pi1_tauminus = pi1_dminus
							pi2_tauminus = pi2_dminus
							pi3_tauminus = pi3_dminus

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
							self.tree.fill('pi1_tauplus_px', numpy.random.normal(pi1_tauplus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi1_tauplus.p.px)
							self.tree.fill('pi1_tauplus_py', numpy.random.normal(pi1_tauplus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi1_tauplus.p.py)
							self.tree.fill('pi1_tauplus_pz', numpy.random.normal(pi1_tauplus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi1_tauplus.p.pz)

							self.tree.fill('pi2_tauplus_q', pi2_tauplus.charge)
							self.tree.fill('pi2_tauplus_px', numpy.random.normal(pi2_tauplus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi2_tauplus.p.px)
							self.tree.fill('pi2_tauplus_py', numpy.random.normal(pi2_tauplus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi2_tauplus.p.py)
							self.tree.fill('pi2_tauplus_pz', numpy.random.normal(pi2_tauplus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi2_tauplus.p.pz)

							self.tree.fill('pi3_tauplus_q', pi3_tauplus.charge)
							self.tree.fill('pi3_tauplus_px', numpy.random.normal(pi3_tauplus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi3_tauplus.p.px)
							self.tree.fill('pi3_tauplus_py', numpy.random.normal(pi3_tauplus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi3_tauplus.p.py)
							self.tree.fill('pi3_tauplus_pz', numpy.random.normal(pi3_tauplus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi3_tauplus.p.pz)

							self.tree.fill('pi1_tauminus_q', pi1_tauminus.charge)
							self.tree.fill('pi1_tauminus_px', numpy.random.normal(pi1_tauminus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi1_tauminus.p.px)
							self.tree.fill('pi1_tauminus_py', numpy.random.normal(pi1_tauminus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi1_tauminus.p.py)
							self.tree.fill('pi1_tauminus_pz', numpy.random.normal(pi1_tauminus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi1_tauminus.p.pz)

							self.tree.fill('pi2_tauminus_q', pi2_tauminus.charge)
							self.tree.fill('pi2_tauminus_px', numpy.random.normal(pi2_tauminus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi2_tauminus.p.px)
							self.tree.fill('pi2_tauminus_py', numpy.random.normal(pi2_tauminus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi2_tauminus.p.py)
							self.tree.fill('pi2_tauminus_pz', numpy.random.normal(pi2_tauminus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi2_tauminus.p.pz)

							self.tree.fill('pi3_tauminus_q', pi3_tauminus.charge)
							self.tree.fill('pi3_tauminus_px', numpy.random.normal(pi3_tauminus.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi3_tauminus.p.px)
							self.tree.fill('pi3_tauminus_py', numpy.random.normal(pi3_tauminus.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi3_tauminus.p.py)
							self.tree.fill('pi3_tauminus_pz', numpy.random.normal(pi3_tauminus.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi3_tauminus.p.pz)

							self.tree.fill('k_q', k.charge)
							self.tree.fill('k_px', numpy.random.normal(k.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else k.p.px)
							self.tree.fill('k_py', numpy.random.normal(k.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else k.p.py)
							self.tree.fill('k_pz', numpy.random.normal(k.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else k.p.pz)

							self.tree.fill('pi_kstar_q', pi_kstar.charge)
							self.tree.fill('pi_kstar_px', numpy.random.normal(pi_kstar.p.px, self.cfg_ana.momentum_x_resolution) if self.cfg_ana.smear_momentum else pi_kstar.p.px)
							self.tree.fill('pi_kstar_py', numpy.random.normal(pi_kstar.p.py, self.cfg_ana.momentum_y_resolution) if self.cfg_ana.smear_momentum else pi_kstar.p.py)
							self.tree.fill('pi_kstar_pz', numpy.random.normal(pi_kstar.p.pz, self.cfg_ana.momentum_z_resolution) if self.cfg_ana.smear_momentum else pi_kstar.p.pz)

							self.tree.tree.Fill()
