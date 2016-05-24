#!/usr/bin/env python

"""
	Analyzer of signal (B0d -> K*0 tau+ tau- nu) events
	                            |   |    |-> pi- pi- pi+ nu
						        |	|-> pi+ pi+ pi- nu
						        |->	K+ pi-

	Note: it is supposed to be used within heppy_fcc framework
"""

import math
import time

import numpy

from ROOT import gROOT, TFile, TH1F, TCanvas

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree

from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex
from heppy_fcc.utility.Particle import Particle

class SignalAnalyzer(Analyzer):
	"""
		Analyzer of signal (B0d -> K*0 tau+ tau- nu) events
		                            |   |    |-> pi- pi- pi+ nu
							        |	|-> pi+ pi+ pi- nu
							        |->	K+ pi-

		Inherits from heppy.framework.analyzer.Analyzer. Overrides base class methods to cover analysis-specific needs
	"""

	def __init__(self, cfg_ana, cfg_comp, looper_name):
		"""
			Constructor

			Just initializes the base class
		"""

		super(SignalAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

	def beginLoop(self, setup):
		"""Overriden base class function"""

		self.rootfile = TFile('/'.join([self.dirName, 'output.root']), 'recreate')

		# tree to store MC truth values and its branches
		self.mc_truth_tree = Tree(self.cfg_ana.mc_truth_tree_name, self.cfg_ana.mc_truth_tree_title)
		self.mc_truth_tree.var('n_particles')
		self.mc_truth_tree.var('event_number')
		self.mc_truth_tree.var('pv_x')
		self.mc_truth_tree.var('pv_y')
		self.mc_truth_tree.var('pv_z')
		self.mc_truth_tree.var('sv_x')
		self.mc_truth_tree.var('sv_y')
		self.mc_truth_tree.var('sv_z')
		self.mc_truth_tree.var('tv_tauplus_x')
		self.mc_truth_tree.var('tv_tauplus_y')
		self.mc_truth_tree.var('tv_tauplus_z')
		self.mc_truth_tree.var('tv_tauminus_x')
		self.mc_truth_tree.var('tv_tauminus_y')
		self.mc_truth_tree.var('tv_tauminus_z')
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
		self.mc_truth_tree.var('tauplus_px')
		self.mc_truth_tree.var('tauplus_py')
		self.mc_truth_tree.var('tauplus_pz')
		self.mc_truth_tree.var('pi1_tauplus_px')
		self.mc_truth_tree.var('pi1_tauplus_py')
		self.mc_truth_tree.var('pi1_tauplus_pz')
		self.mc_truth_tree.var('pi1_tauplus_q')
		self.mc_truth_tree.var('pi2_tauplus_px')
		self.mc_truth_tree.var('pi2_tauplus_py')
		self.mc_truth_tree.var('pi2_tauplus_pz')
		self.mc_truth_tree.var('pi2_tauplus_q')
		self.mc_truth_tree.var('pi3_tauplus_px')
		self.mc_truth_tree.var('pi3_tauplus_py')
		self.mc_truth_tree.var('pi3_tauplus_pz')
		self.mc_truth_tree.var('pi3_tauplus_q')
		self.mc_truth_tree.var('nu_tauplus_px')
		self.mc_truth_tree.var('nu_tauplus_py')
		self.mc_truth_tree.var('nu_tauplus_pz')
		self.mc_truth_tree.var('tauminus_px')
		self.mc_truth_tree.var('tauminus_py')
		self.mc_truth_tree.var('tauminus_pz')
		self.mc_truth_tree.var('pi1_tauminus_px')
		self.mc_truth_tree.var('pi1_tauminus_py')
		self.mc_truth_tree.var('pi1_tauminus_pz')
		self.mc_truth_tree.var('pi1_tauminus_q')
		self.mc_truth_tree.var('pi2_tauminus_px')
		self.mc_truth_tree.var('pi2_tauminus_py')
		self.mc_truth_tree.var('pi2_tauminus_pz')
		self.mc_truth_tree.var('pi2_tauminus_q')
		self.mc_truth_tree.var('pi3_tauminus_px')
		self.mc_truth_tree.var('pi3_tauminus_py')
		self.mc_truth_tree.var('pi3_tauminus_pz')
		self.mc_truth_tree.var('pi3_tauminus_q')
		self.mc_truth_tree.var('nu_tauminus_px')
		self.mc_truth_tree.var('nu_tauminus_py')
		self.mc_truth_tree.var('nu_tauminus_pz')

		# the same for smeared values
		self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)
		self.tree.var('n_particles')
		self.tree.var('event_number')
		self.tree.var('pv_x')
		self.tree.var('pv_y')
		self.tree.var('pv_z')
		self.tree.var('sv_x')
		self.tree.var('sv_y')
		self.tree.var('sv_z')
		self.tree.var('tv_tauplus_x')
		self.tree.var('tv_tauplus_y')
		self.tree.var('tv_tauplus_z')
		self.tree.var('tv_tauminus_x')
		self.tree.var('tv_tauminus_y')
		self.tree.var('tv_tauminus_z')
		self.tree.var('k_px')
		self.tree.var('k_py')
		self.tree.var('k_pz')
		self.tree.var('k_q')
		self.tree.var('pi_kstar_px')
		self.tree.var('pi_kstar_py')
		self.tree.var('pi_kstar_pz')
		self.tree.var('pi_kstar_q')
		self.tree.var('pi1_tauplus_px')
		self.tree.var('pi1_tauplus_py')
		self.tree.var('pi1_tauplus_pz')
		self.tree.var('pi1_tauplus_q')
		self.tree.var('pi2_tauplus_px')
		self.tree.var('pi2_tauplus_py')
		self.tree.var('pi2_tauplus_pz')
		self.tree.var('pi2_tauplus_q')
		self.tree.var('pi3_tauplus_px')
		self.tree.var('pi3_tauplus_py')
		self.tree.var('pi3_tauplus_pz')
		self.tree.var('pi3_tauplus_q')
		self.tree.var('pi1_tauminus_px')
		self.tree.var('pi1_tauminus_py')
		self.tree.var('pi1_tauminus_pz')
		self.tree.var('pi1_tauminus_q')
		self.tree.var('pi2_tauminus_px')
		self.tree.var('pi2_tauminus_py')
		self.tree.var('pi2_tauminus_pz')
		self.tree.var('pi2_tauminus_q')
		self.tree.var('pi3_tauminus_px')
		self.tree.var('pi3_tauminus_py')
		self.tree.var('pi3_tauminus_pz')
		self.tree.var('pi3_tauminus_q')

		# statistics
		self.counter = 0 # Total number of processed decays
		self.pb_counter = 0 # Number of events with B momentum > 25 GeV
		self.pvsv_distance_counter = 0 # Number of events with distance between PV and SV > 1 mm
		self.max_svtv_distance_counter = 0 # Number of events with any distance between SV and TV > 0.5 mm
		# histograms to visualize cuts
		TH1F.AddDirectory(False) # not to link histograms to files
		gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots
		self.pb_hist = TH1F('pb_hist', 'P_{B}', 100, 0, 50)
		self.pvsv_distance_hist = TH1F('pvsv_distance_hist', 'FD_{B}', 100, 0, 10)
		self.max_svtv_distance_hist = TH1F('max_svtv_distance_hist', 'Max FD_{#tau}', 100, 0, 5)

		super(SignalAnalyzer, self).beginLoop(setup)

		#time
		self.start_time = time.time()
		self.last_timestamp = time.time()

	def process(self, event):
		"""Overriden base class function"""

		b = None # B0d particle
		kstar = None # K*0 from B0d decay
		k = None # K from K*0 decay
		pi_kstar = None # pi from K* decay
		tauplus = None # tau+ from B0d decay
		pi1_tauplus = None # pi from tau+ decay
		pi2_tauplus = None # pi from tau+ decay
		pi3_tauplus = None # pi from tau+ decay
		nu_tauplus = None # nu from tau+ decay
		tauminus = None # tau- from B0d decay
		pi1_tauminus = None # pi from tau- decay
		pi2_tauminus = None # pi from tau- decay
		pi3_tauminus = None # pi from tau- decay
		nu_tauminus = None # nu from tau- decay

		pv = None # primary vertex
		sv = None # secondary vertex
		tv_tauplus = None # tau+ decay vertex
		tv_tauminus = None # tau- decay vertex

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
			if abs(ptc_gen1.pdgid) == 511 and ptc_gen1.start_vertex != ptc_gen1.end_vertex: # if B0d found and it's not an oscillation
				self.counter += 1
				if self.counter % 100 == 0:
					print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
					self.last_timestamp = time.time()

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
							if ptc_gen2.start_vertex == b.end_vertex:
								# looking for tau+
								if ptc_gen2.pdgid == -15:
									tauplus = ptc_gen2
									tv_tauplus = tauplus.end_vertex

								# looking for tau-
								if ptc_gen2.pdgid == 15:
									tauminus = ptc_gen2
									tv_tauminus = tauminus.end_vertex

								# looking for K*
								if abs(ptc_gen2.pdgid) == 313:
									kstar = ptc_gen2

						max_svtv_distance = max(math.sqrt((tv_tauplus.x - sv.x) ** 2 + (tv_tauplus.y - sv.y) ** 2 + (tv_tauplus.z - sv.z) ** 2), math.sqrt((tv_tauminus.x - sv.x) ** 2 + (tv_tauminus.y - sv.y) ** 2 + (tv_tauminus.z - sv.z) ** 2))

						if max_svtv_distance > 0.5: # select only events with long flight distance of tau
							self.max_svtv_distance_counter += 1

							pis_tauplus = []
							pis_tauminus = []

							for ptc_gen3 in ptcs:
								if ptc_gen3.start_vertex == kstar.end_vertex:
									# looking for K
									if abs(ptc_gen3.pdgid) == 321:
										k = ptc_gen3

									# looking for pi
									if abs(ptc_gen3.pdgid) == 211:
										pi_kstar = ptc_gen3

								if  ptc_gen3.start_vertex == tauplus.end_vertex:
									# looking for pions from tau+ decay
									if abs(ptc_gen3.pdgid) == 211:
										pis_tauplus.append(ptc_gen3)

									# looking for nu from tau+ decay
									if ptc_gen3.pdgid == -16:
										nu_tauplus = ptc_gen3

								if ptc_gen3.start_vertex == tauminus.end_vertex:
									# looking for pions from tau- decay
									if abs(ptc_gen3.pdgid) == 211:
										pis_tauminus.append(ptc_gen3)

									# looking for nu from tau- decay
									if ptc_gen3.pdgid == 16:
										nu_tauminus = ptc_gen3

							if len(pis_tauplus) == 3:
								pi1_tauplus, pi2_tauplus, pi3_tauplus = pis_tauplus[0], pis_tauplus[1], pis_tauplus[2]

							if len(pis_tauminus) == 3:
								pi1_tauminus, pi2_tauminus, pi3_tauminus = pis_tauminus[0], pis_tauminus[1], pis_tauminus[2]

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
							self.mc_truth_tree.fill('tv_tauplus_x', tv_tauplus.x)
							self.mc_truth_tree.fill('tv_tauplus_y', tv_tauplus.y)
							self.mc_truth_tree.fill('tv_tauplus_z', tv_tauplus.z)
							self.mc_truth_tree.fill('tv_tauminus_x', tv_tauminus.x)
							self.mc_truth_tree.fill('tv_tauminus_y', tv_tauminus.y)
							self.mc_truth_tree.fill('tv_tauminus_z', tv_tauminus.z)

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

							self.mc_truth_tree.fill('tauplus_px', tauplus.p.px)
							self.mc_truth_tree.fill('tauplus_py', tauplus.p.py)
							self.mc_truth_tree.fill('tauplus_pz', tauplus.p.pz)

							self.mc_truth_tree.fill('pi1_tauplus_q', pi1_tauplus.charge)
							self.mc_truth_tree.fill('pi1_tauplus_px', pi1_tauplus.p.px)
							self.mc_truth_tree.fill('pi1_tauplus_py', pi1_tauplus.p.py)
							self.mc_truth_tree.fill('pi1_tauplus_pz', pi1_tauplus.p.pz)

							self.mc_truth_tree.fill('pi2_tauplus_q', pi2_tauplus.charge)
							self.mc_truth_tree.fill('pi2_tauplus_px', pi2_tauplus.p.px)
							self.mc_truth_tree.fill('pi2_tauplus_py', pi2_tauplus.p.py)
							self.mc_truth_tree.fill('pi2_tauplus_pz', pi2_tauplus.p.pz)

							self.mc_truth_tree.fill('pi3_tauplus_q', pi3_tauplus.charge)
							self.mc_truth_tree.fill('pi3_tauplus_px', pi3_tauplus.p.px)
							self.mc_truth_tree.fill('pi3_tauplus_py', pi3_tauplus.p.py)
							self.mc_truth_tree.fill('pi3_tauplus_pz', pi3_tauplus.p.pz)

							self.mc_truth_tree.fill('nu_tauplus_px', nu_tauplus.p.px)
							self.mc_truth_tree.fill('nu_tauplus_py', nu_tauplus.p.py)
							self.mc_truth_tree.fill('nu_tauplus_pz', nu_tauplus.p.pz)

							self.mc_truth_tree.fill('tauminus_px', tauminus.p.px)
							self.mc_truth_tree.fill('tauminus_py', tauminus.p.py)
							self.mc_truth_tree.fill('tauminus_pz', tauminus.p.pz)

							self.mc_truth_tree.fill('pi1_tauminus_q', pi1_tauminus.charge)
							self.mc_truth_tree.fill('pi1_tauminus_px', pi1_tauminus.p.px)
							self.mc_truth_tree.fill('pi1_tauminus_py', pi1_tauminus.p.py)
							self.mc_truth_tree.fill('pi1_tauminus_pz', pi1_tauminus.p.pz)

							self.mc_truth_tree.fill('pi2_tauminus_q', pi2_tauminus.charge)
							self.mc_truth_tree.fill('pi2_tauminus_px', pi2_tauminus.p.px)
							self.mc_truth_tree.fill('pi2_tauminus_py', pi2_tauminus.p.py)
							self.mc_truth_tree.fill('pi2_tauminus_pz', pi2_tauminus.p.pz)

							self.mc_truth_tree.fill('pi3_tauminus_q', pi3_tauminus.charge)
							self.mc_truth_tree.fill('pi3_tauminus_px', pi3_tauminus.p.px)
							self.mc_truth_tree.fill('pi3_tauminus_py', pi3_tauminus.p.py)
							self.mc_truth_tree.fill('pi3_tauminus_pz', pi3_tauminus.p.pz)

							self.mc_truth_tree.fill('nu_tauminus_px', nu_tauminus.p.px)
							self.mc_truth_tree.fill('nu_tauminus_py', nu_tauminus.p.py)
							self.mc_truth_tree.fill('nu_tauminus_pz', nu_tauminus.p.pz)

							self.mc_truth_tree.tree.Fill()

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

	def write(self, setup):
		"""Overriden base class function"""

		# finalizing writing to the file
		self.rootfile.Write()
		self.rootfile.Close()

		# drawing the histograms
		pb_canvas = TCanvas('pb_canvas', 'B momentum', 640, 480)
		pb_canvas.cd()
		self.pb_hist.GetXaxis().SetTitle('p_{B}, GeV/#it{c}')
		self.pb_hist.GetYaxis().SetTitle('Events / ({} GeV/#it{{c}})'.format((self.pb_hist.GetXaxis().GetXmax() - self.pb_hist.GetXaxis().GetXmin()) / self.pb_hist.GetNbinsX()))
		self.pb_hist.Draw()
		pb_canvas.Update()

		pvsv_distance_canvas = TCanvas('pvsv_distance_canvas', 'Distance between PV and SV', 640, 480)
		pvsv_distance_canvas.cd()
		self.pvsv_distance_hist.GetXaxis().SetTitle('mm')
		self.pvsv_distance_hist.GetYaxis().SetTitle('Events / ({} mm)'.format((self.pvsv_distance_hist.GetXaxis().GetXmax() - self.pvsv_distance_hist.GetXaxis().GetXmin()) / self.pvsv_distance_hist.GetNbinsX()))
		self.pvsv_distance_hist.Draw()
		pvsv_distance_canvas.Update()

		max_svtv_distance_canvas = TCanvas('max_svtv_distance_canvas', 'Max distance between SV and TV', 640, 480)
		max_svtv_distance_canvas.cd()
		self.max_svtv_distance_hist.GetXaxis().SetTitle('mm')
		self.max_svtv_distance_hist.GetYaxis().SetTitle('Events / ({} mm)'.format((self.max_svtv_distance_hist.GetXaxis().GetXmax() - self.max_svtv_distance_hist.GetXaxis().GetXmin()) / self.max_svtv_distance_hist.GetNbinsX()))
		self.max_svtv_distance_hist.Draw()
		max_svtv_distance_canvas.Update()

		# some useful statistics
		print('Total decays processed: {}'.format(self.counter))
		print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
		print('Efficiency:\n\tMomentum of B cut: {:.3f}\n\tDistance between PV and SV cut: {:.3f}\n\tMax distance between SV and TV cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter), float(self.pvsv_distance_counter)/float(self.counter), float(self.max_svtv_distance_counter)/float(self.counter)))
		raw_input('Press ENTER when finished')
