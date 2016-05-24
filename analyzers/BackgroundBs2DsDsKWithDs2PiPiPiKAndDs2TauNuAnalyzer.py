#!/usr/bin/env python

"""
	Analyzer of B0d -> K*0 Ds+ Ds- events
	                    |   |   |-> tau- nu
						|   |        |-> pi- pi- pi+ nu
						|   |-> pi+ pi+ pi- K0L
						|->	K+ pi-

	Note: it is supposed to be used within heppy_fcc framework
"""

import math
import time

import numpy

from heppy_fcc.utility.CommonAnalyzer import CommonAnalyzer
from heppy_fcc.utility.Particle import Particle

class BackgroundBs2DsDsKWithDs2PiPiPiKAndDs2TauNuAnalyzer(CommonAnalyzer):
	"""
		Analyzer of B0d -> K*0 Ds+ Ds- background events
		                    |   |   |-> tau- nu
							|   |        |-> pi- pi- pi+ nu
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

		super(BackgroundBs2DsDsKWithDs2PiPiPiKAndDs2TauNuAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

		# MC truth values
		self.mc_truth_tree.var('n_particles')
		self.mc_truth_tree.var('event_number')
		self.mc_truth_tree.var('pv_x')
		self.mc_truth_tree.var('pv_y')
		self.mc_truth_tree.var('pv_z')
		self.mc_truth_tree.var('sv_x')
		self.mc_truth_tree.var('sv_y')
		self.mc_truth_tree.var('sv_z')
		self.mc_truth_tree.var('tv_d_x')
		self.mc_truth_tree.var('tv_d_y')
		self.mc_truth_tree.var('tv_d_z')
		self.mc_truth_tree.var('tv_tau_d_x')
		self.mc_truth_tree.var('tv_tau_d_y')
		self.mc_truth_tree.var('tv_tau_d_z')
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
		self.mc_truth_tree.var('k0_d_px')
		self.mc_truth_tree.var('k0_d_py')
		self.mc_truth_tree.var('k0_d_pz')
		self.mc_truth_tree.var('dminus_px')
		self.mc_truth_tree.var('dminus_py')
		self.mc_truth_tree.var('dminus_pz')
		self.mc_truth_tree.var('tau_d_px')
		self.mc_truth_tree.var('tau_d_py')
		self.mc_truth_tree.var('tau_d_pz')
		self.mc_truth_tree.var('tau_d_q')
		self.mc_truth_tree.var('pi1_tau_d_px')
		self.mc_truth_tree.var('pi1_tau_d_py')
		self.mc_truth_tree.var('pi1_tau_d_pz')
		self.mc_truth_tree.var('pi1_tau_d_q')
		self.mc_truth_tree.var('pi2_tau_d_px')
		self.mc_truth_tree.var('pi2_tau_d_py')
		self.mc_truth_tree.var('pi2_tau_d_pz')
		self.mc_truth_tree.var('pi2_tau_d_q')
		self.mc_truth_tree.var('pi3_tau_d_px')
		self.mc_truth_tree.var('pi3_tau_d_py')
		self.mc_truth_tree.var('pi3_tau_d_pz')
		self.mc_truth_tree.var('pi3_tau_d_q')
		self.mc_truth_tree.var('nu_tau_d_px')
		self.mc_truth_tree.var('nu_tau_d_py')
		self.mc_truth_tree.var('nu_tau_d_pz')
		self.mc_truth_tree.var('nu_d_px')
		self.mc_truth_tree.var('nu_d_py')
		self.mc_truth_tree.var('nu_d_pz')

	def process(self, event):
		b = None # B particle
		kstar = None # K* from B decay
		k = None # K from K* decay
		pi_kstar = None # pi from K* decay
		dplus = None # Ds+ from Bs decay
		pi1_d = None # pi from Ds decay
		pi2_d = None # pi from Ds decay
		pi3_d = None # pi from Ds decay
		k0_d = None # pi0 from Ds decay
		dminus = None # Ds- from Bs decay
		tau_d = None # tau+ from Ds decay
		pi1_tau_d = None # pi from tau decay
		pi2_tau_d = None # pi from tau decay
		pi3_tau_d = None # pi from tau decay
		nu_tau_d = None # nu from tau decay
		nu_d = None # nu from Ds decay

		pv = None # primary vertex
		sv = None # secondary vertex
		tv_d = None # Ds decay vertex
		tv_tau_d= None # tau decay vertex

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
			if abs(ptc_gen1.pdgid) == 531 and ptc_gen1.start_vertex != ptc_gen1.end_vertex: # if B found and it's not an oscillation
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

								# looking for Ds-
								if ptc_gen2.pdgid == -431:
									dminus = ptc_gen2

								# looking for K*
								if abs(ptc_gen2.pdgid) == 313:
									kstar = ptc_gen2

						pis_d = []
						for ptc_gen3 in ptcs:
							if ptc_gen3.start_vertex == kstar.end_vertex:
								# looking for K
								if abs(ptc_gen3.pdgid) == 321:
									k = ptc_gen3

								# looking for pi
								if abs(ptc_gen3.pdgid) == 211:
									pi_kstar = ptc_gen3

							if ptc_gen3.start_vertex == dplus.end_vertex or ptc_gen3.start_vertex == dminus.end_vertex:
								# looking for tau
								if abs(ptc_gen3.pdgid) == 15:
									tau_d = ptc_gen3
									tv_tau_d = tau_d.end_vertex

								# looking for nu
								if abs(ptc_gen3.pdgid) == 16:
									nu_d = ptc_gen3

								# looking for pi+/-
								if abs(ptc_gen3.pdgid) == 211:
									pis_d.append(ptc_gen3)

								# looking for pi0
								if abs(ptc_gen3.pdgid) == 130:
									k0_d = ptc_gen3

						if len(pis_d) == 3:
							pi1_d, pi2_d, pi3_d = pis_d[0], pis_d[1], pis_d[2]
							tv_d = k0_d.start_vertex

						max_svtv_distance = max(math.sqrt((tv_d.x - sv.x) ** 2 + (tv_d.y - sv.y) ** 2 + (tv_d.z - sv.z) ** 2), math.sqrt((tv_tau_d.x - sv.x) ** 2 + (tv_tau_d.y - sv.y) ** 2 + (tv_tau_d.z - sv.z) ** 2))

						if max_svtv_distance > 0.5: # select only events with long flight distance of tau
							self.max_svtv_distance_counter += 1

							pis_tau_d = []
							for ptc_gen4 in ptcs:
								if ptc_gen4.start_vertex == tau_d.end_vertex:
									# looking for pi+/-
									if abs(ptc_gen4.pdgid) == 211:
										pis_tau_d.append(ptc_gen4)

									# looking for nu
									if abs(ptc_gen4.pdgid) == 16:
										nu_tau_d = ptc_gen4

							if len(pis_tau_d) == 3:
								pi1_tau_d, pi2_tau_d, pi3_tau_d = pis_tau_d[0], pis_tau_d[1], pis_tau_d[2]

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
							self.mc_truth_tree.fill('tv_tau_d_x', tv_tau_d.x)
							self.mc_truth_tree.fill('tv_tau_d_y', tv_tau_d.y)
							self.mc_truth_tree.fill('tv_tau_d_z', tv_tau_d.z)
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

							self.mc_truth_tree.fill('dplus_px', dplus.p.px)
							self.mc_truth_tree.fill('dplus_py', dplus.p.py)
							self.mc_truth_tree.fill('dplus_pz', dplus.p.pz)

							self.mc_truth_tree.fill('tau_d_px', tau_d.p.px)
							self.mc_truth_tree.fill('tau_d_py', tau_d.p.py)
							self.mc_truth_tree.fill('tau_d_pz', tau_d.p.pz)

							self.mc_truth_tree.fill('pi1_tau_d_q', pi1_tau_d.charge)
							self.mc_truth_tree.fill('pi1_tau_d_px', pi1_tau_d.p.px)
							self.mc_truth_tree.fill('pi1_tau_d_py', pi1_tau_d.p.py)
							self.mc_truth_tree.fill('pi1_tau_d_pz', pi1_tau_d.p.pz)

							self.mc_truth_tree.fill('pi2_tau_d_q', pi2_tau_d.charge)
							self.mc_truth_tree.fill('pi2_tau_d_px', pi2_tau_d.p.px)
							self.mc_truth_tree.fill('pi2_tau_d_py', pi2_tau_d.p.py)
							self.mc_truth_tree.fill('pi2_tau_d_pz', pi2_tau_d.p.pz)

							self.mc_truth_tree.fill('pi3_tau_d_q', pi3_tau_d.charge)
							self.mc_truth_tree.fill('pi3_tau_d_px', pi3_tau_d.p.px)
							self.mc_truth_tree.fill('pi3_tau_d_py', pi3_tau_d.p.py)
							self.mc_truth_tree.fill('pi3_tau_d_pz', pi3_tau_d.p.pz)

							self.mc_truth_tree.fill('nu_tau_d_px', nu_tau_d.p.px)
							self.mc_truth_tree.fill('nu_tau_d_py', nu_tau_d.p.py)
							self.mc_truth_tree.fill('nu_tau_d_pz', nu_tau_d.p.pz)

							self.mc_truth_tree.fill('nu_d_px', nu_d.p.px)
							self.mc_truth_tree.fill('nu_d_py', nu_d.p.py)
							self.mc_truth_tree.fill('nu_d_pz', nu_d.p.pz)

							self.mc_truth_tree.fill('dminus_px', dminus.p.px)
							self.mc_truth_tree.fill('dminus_py', dminus.p.py)
							self.mc_truth_tree.fill('dminus_pz', dminus.p.pz)

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

							self.mc_truth_tree.fill('k0_d_px', k0_d.p.px)
							self.mc_truth_tree.fill('k0_d_py', k0_d.p.py)
							self.mc_truth_tree.fill('k0_d_pz', k0_d.p.pz)

							self.mc_truth_tree.tree.Fill()

							# matching visible particles and MC truth ones
							if tau_d.charge < 0:
								tv_tauminus = tv_tau_d
								pi1_tauminus = pi1_tau_d
								pi2_tauminus = pi2_tau_d
								pi3_tauminus = pi3_tau_d

								tv_tauplus = tv_d
								pi1_tauplus = pi1_d
								pi2_tauplus = pi2_d
								pi3_tauplus = pi3_d
							else:
								tv_tauplus = tv_tau_d
								pi1_tauplus = pi1_tau_d
								pi2_tauplus = pi2_tau_d
								pi3_tauplus = pi3_tau_d

								tv_tauminus = tv_d
								pi1_tauminus = pi1_d
								pi2_tauminus = pi2_d
								pi3_tauminus = pi3_d

							# filling event information
							self.tree.fill('event_number', event_number)
							self.tree.fill('n_particles', n_particles)

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
