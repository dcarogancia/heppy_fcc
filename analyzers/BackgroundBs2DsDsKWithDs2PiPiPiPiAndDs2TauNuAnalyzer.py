## Analyzer of B0s -> Ds+ (-> pi+ pi+ pi+ pi0) Ds- (-> tau nu) K*0 events
#  It is supposed to be used within heppy_fcc framework

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree

import math
import copy
import time

import numpy

from ROOT import gROOT
from ROOT import TFile
from ROOT import TH1F
from ROOT import TCanvas

from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex
from heppy_fcc.utility.Particle import Particle

def smear_momentum(p, px_resolution, py_resolution, pz_resolution):
	return Momentum(numpy.random.normal(p.px, px_resolution), numpy.random.normal(p.py, py_resolution), numpy.random.normal(p.pz, pz_resolution))

def smear_vertex(v, x_resolution, y_resolution, z_resolution):
	return Vertex(numpy.random.normal(v.x, x_resolution), numpy.random.normal(v.y, y_resolution), numpy.random.normal(v.z, z_resolution))

class BackgroundBs2DsDsKWithDs2PiPiPiPiAndDs2TauNuAnalyzer(Analyzer):
	def beginLoop(self, setup):
		self.start_time = time.time()
		self.last_timestamp = time.time()

		self.counter = 0 # Total number of processed decays
		self.pb_counter = 0 # Number of events with B momentum > 25 GeV
		self.pvsv_distance_counter = 0 # Number of events with distance between PV and SV > 1 mm
		self.max_svtv_distance_counter = 0 # Number of events with any distance between SV and TV > 0.5 mm

		gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

		# histograms to visualize cuts
		self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)
		self.pvsv_distance_hist = TH1F('pvsv_distance_hist', 'FD_{B}', 500, 0, 10)
		self.max_svtv_distance_hist = TH1F('max_svtv_distance_hist', 'Max FD_{#tau}', 500, 0, 5)

		super(BackgroundBs2DsDsKWithDs2PiPiPiPiAndDs2TauNuAnalyzer, self).beginLoop(setup)
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
		self.mc_truth_tree.var('pi_k_px')
		self.mc_truth_tree.var('pi_k_py')
		self.mc_truth_tree.var('pi_k_pz')
		self.mc_truth_tree.var('pi_k_q')
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
		self.mc_truth_tree.var('pi0_d_px')
		self.mc_truth_tree.var('pi0_d_py')
		self.mc_truth_tree.var('pi0_d_pz')
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

		# same for smeared values
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
		self.tree.var('pi_k_px')
		self.tree.var('pi_k_py')
		self.tree.var('pi_k_pz')
		self.tree.var('pi_k_q')
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

	def process(self, event):
		pv_mc_truth = None # primary vertex (MC truth)
		sv_mc_truth = None # secondary vertex (MC truth)
		tv_d_mc_truth = None # Ds decay vertex (MC truth)
		tv_tau_d_mc_truth = None # tau decay vertex (MC truth)
		b_mc_truth = None # B particle (MC truth)
		kstar_mc_truth = None # K* from B decay (MC truth)
		k_mc_truth = None # K from K* decay (MC truth)
		pi_k_mc_truth = None # pi from K* decay (MC truth)
		dplus_mc_truth = None # Ds+ from Bs decay
		pi1_d_mc_truth = None # pi from Ds decay (MC truth)
		pi2_d_mc_truth = None # pi from Ds decay (MC truth)
		pi3_d_mc_truth = None # pi from Ds decay (MC truth)
		pi0_d_mc_truth = None # pi0 from Ds decay (MC truth)
		dminus_mc_truth = None # Ds- from Bs decay
		tau_d_mc_truth = None # tau+ from Ds decay (MC truth)
		pi1_tau_d_mc_truth = None # pi from tau decay (MC truth)
		pi2_tau_d_mc_truth = None # pi from tau decay (MC truth)
		pi3_tau_d_mc_truth = None # pi from tau decay (MC truth)
		nu_tau_d_mc_truth = None # nu from tau decay (MC truth)
		nu_d_mc_truth = None # nu from Ds decay (MC truth)

		pv = None # primary vertex
		sv = None # secondary vertex
		tv_tauplus = None # tau+ decay vertex
		tv_tauminus = None # tau- decay vertex
		k = None # K from K*0 decay
		pi_k = None # pi from K*0 decay
		pi1_tauplus = None # pi from tau+ decay
		pi2_tauplus = None # pi from tau+ decay
		pi3_tauplus = None # pi from tau+ decay
		pi1_tauminus = None # pi from tau- decay
		pi2_tauminus = None # pi from tau- decay
		pi3_tauminus = None # pi from tau- decay

		pvsv_distance = 0. # distance between PV and SV
		pb = 0. # B momentum
		svtv_tauplus_distance = 0. # distance between SV and tau+ decay vertex
		svtv_tauminus_distance = 0. # distance between SV and tau- decay vertex

		store = event.input # This is just a shortcut
		event_info = store.get("EventInfo")
		particles_info = store.get("GenParticle")
		vertices_info = store.get("GenVertex")

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

				b_mc_truth = ptc_gen1

				pb = b_mc_truth.p.absvalue()

				if pb > 25.: # Select only events with large momentum of the B
					self.pb_counter += 1

					pv_mc_truth = b_mc_truth.start_vertex
					sv_mc_truth = b_mc_truth.end_vertex
					pvsv_distance = math.sqrt((sv_mc_truth.x - pv_mc_truth.x) ** 2 + (sv_mc_truth.y - pv_mc_truth.y) ** 2 + (sv_mc_truth.z - pv_mc_truth.z) ** 2)

					if pvsv_distance > 1.: # Select only events with long flight distance of the B
						self.pvsv_distance_counter += 1

						pv = copy.deepcopy(pv_mc_truth)
						sv = copy.deepcopy(sv_mc_truth)

						for ptc_gen2 in ptcs:
							# checking B-decay products
							if ptc_gen2.start_vertex == b_mc_truth.end_vertex:
								# looking for Ds+
								if ptc_gen2.pdgid == 431:
									dplus_mc_truth = ptc_gen2

								# looking for Ds-
								if ptc_gen2.pdgid == -431:
									dminus_mc_truth = ptc_gen2

								# looking for K*
								if abs(ptc_gen2.pdgid) == 313:
									kstar_mc_truth = ptc_gen2

						pis_d = list([])
						for ptc_gen3 in ptcs:
							# checking Ds-decay products
							if ptc_gen3.start_vertex == dplus_mc_truth.end_vertex or ptc_gen3.start_vertex == dminus_mc_truth.end_vertex:
								# looking for tau
								if abs(ptc_gen3.pdgid) == 15:
									tau_d_mc_truth = ptc_gen3
									tv_tau_d_mc_truth = ptc_gen3.end_vertex
									if tau_d_mc_truth.pdgid < 0:
										tv_tauplus = copy.deepcopy(tv_tau_d_mc_truth) # copy is needed in order to keep initial vertex properties after smearing
										svtv_tauplus_distance = math.sqrt((tv_tau_d_mc_truth.x - sv_mc_truth.x) ** 2 + (tv_tau_d_mc_truth.y - sv_mc_truth.y) ** 2 + (tv_tau_d_mc_truth.z - sv_mc_truth.z) ** 2)
									else:
										tv_tauminus = copy.deepcopy(tv_tau_d_mc_truth) # copy is needed in order to keep initial vertex properties after smearing
										svtv_tauminus_distance = math.sqrt((tv_tau_d_mc_truth.x - sv_mc_truth.x) ** 2 + (tv_tau_d_mc_truth.y - sv_mc_truth.y) ** 2 + (tv_tau_d_mc_truth.z - sv_mc_truth.z) ** 2)

								# looking for nu
								if abs(ptc_gen3.pdgid) == 16:
									nu_d_mc_truth = ptc_gen3

								# looking for charged pions
								if abs(ptc_gen3.pdgid) == 211:
									pis_d.append(ptc_gen3)

								# looking for pi0
								if abs(ptc_gen3.pdgid) == 111:
									pi0_d_mc_truth = ptc_gen3

						if len(pis_d) == 3:
							pi1_d_mc_truth, pi2_d_mc_truth, pi3_d_mc_truth = pis_d[0], pis_d[1], pis_d[2]
							tv_d_mc_truth = pi0_d_mc_truth.start_vertex

							total_q = pi1_d_mc_truth.charge + pi2_d_mc_truth.charge + pi3_d_mc_truth.charge
							if total_q < 0:
								pi1_tauminus, pi2_tauminus, pi3_tauminus = copy.deepcopy(pi1_d_mc_truth), copy.deepcopy(pi2_d_mc_truth), copy.deepcopy(pi3_d_mc_truth)
								tv_tauminus = copy.deepcopy(tv_d_mc_truth)
								svtv_tauminus_distance = math.sqrt((tv_d_mc_truth.x - sv_mc_truth.x) ** 2 + (tv_d_mc_truth.y - sv_mc_truth.y) ** 2 + (tv_d_mc_truth.z - sv_mc_truth.z) ** 2)
							else:
								pi1_tauplus, pi2_tauplus, pi3_tauplus = copy.deepcopy(pi1_d_mc_truth), copy.deepcopy(pi2_d_mc_truth), copy.deepcopy(pi3_d_mc_truth)
								tv_tauplus = copy.deepcopy(tv_d_mc_truth)
								svtv_tauplus_distance = math.sqrt((tv_d_mc_truth.x - sv_mc_truth.x) ** 2 + (tv_d_mc_truth.y - sv_mc_truth.y) ** 2 + (tv_d_mc_truth.z - sv_mc_truth.z) ** 2)

						if max(svtv_tauplus_distance, svtv_tauminus_distance) > 0.5: # select only events with long flight distance of tau
							self.max_svtv_distance_counter += 1

							pis_tau_d_mc_truth = list([])

							for ptc in ptcs:
								# checking K*-decay products
								if ptc.start_vertex == kstar_mc_truth.end_vertex:
									# looking for K
									if abs(ptc.pdgid) == 321:
										k_mc_truth = ptc
										k = copy.deepcopy(k_mc_truth) # copy is needed in order to keep initial particle properties after smearing

									# looking for pi_K
									if abs(ptc.pdgid) == 211:
										pi_k_mc_truth = ptc
										pi_k = copy.deepcopy(pi_k_mc_truth) # copy is needed in order to keep initial particle properties after smearing

								# checking for tau (from Ds-decay) decay products
								if ptc.start_vertex == tau_d_mc_truth.end_vertex:
									# looking for pions from tau+ decay
									if abs(ptc.pdgid) == 211:
										pis_tau_d_mc_truth.append(ptc)

									# looking for nu from tau decay
									if abs(ptc.pdgid) == 16:
										nu_tau_d_mc_truth = ptc

							if len(pis_tau_d_mc_truth) == 3:
								pi1_tau_d_mc_truth, pi2_tau_d_mc_truth, pi3_tau_d_mc_truth = pis_tau_d_mc_truth[0], pis_tau_d_mc_truth[1], pis_tau_d_mc_truth[2]
								if tau_d_mc_truth.charge < 0:
									pi1_tauminus, pi2_tauminus, pi3_tauminus = copy.deepcopy(pi1_tau_d_mc_truth), copy.deepcopy(pi2_tau_d_mc_truth), copy.deepcopy(pi3_tau_d_mc_truth)
								else:
									pi1_tauplus, pi2_tauplus, pi3_tauplus = copy.deepcopy(pi1_tau_d_mc_truth), copy.deepcopy(pi2_tau_d_mc_truth), copy.deepcopy(pi3_tau_d_mc_truth)


							# applying smearing
							if self.cfg_ana.smear_momentum:
								k.p = smear_momentum(k.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi_k.p = smear_momentum(pi_k.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi1_tauplus.p = smear_momentum(pi1_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi2_tauplus.p = smear_momentum(pi2_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi3_tauplus.p = smear_momentum(pi3_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi1_tauminus.p = smear_momentum(pi1_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi2_tauminus.p = smear_momentum(pi2_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
								pi3_tauminus.p = smear_momentum(pi3_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
							if self.cfg_ana.smear_pv:
								pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)
							if self.cfg_ana.smear_sv:
								sv = smear_vertex(sv, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)

								# to keep consistency
								k.start_vertex = sv
								pi_k.start_vertex = sv
							if self.cfg_ana.smear_tv:
								tv_tauplus = smear_vertex(tv_tauplus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
								tv_tauminus = smear_vertex(tv_tauminus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)

								# to keep consistency
								pi1_tauplus.start_vertex, pi2_tauplus.start_vertex, pi3_tauplus.start_vertex = tv_tauplus, tv_tauplus, tv_tauplus
								pi1_tauminus.start_vertex, pi2_tauminus.start_vertex, pi3_tauminus.start_vertex = tv_tauminus, tv_tauminus, tv_tauminus

							# filling histograms
							self.pvsv_distance_hist.Fill(pvsv_distance)
							self.pb_hist.Fill(pb)
							self.max_svtv_distance_hist.Fill(max(svtv_tauplus_distance, svtv_tauminus_distance))

							# filling MC truth information
							self.mc_truth_tree.fill('event_number', event_number)
							self.mc_truth_tree.fill('n_particles', n_particles)

							self.mc_truth_tree.fill('pv_x', pv_mc_truth.x)
							self.mc_truth_tree.fill('pv_y', pv_mc_truth.y)
							self.mc_truth_tree.fill('pv_z', pv_mc_truth.z)
							self.mc_truth_tree.fill('sv_x', sv_mc_truth.x)
							self.mc_truth_tree.fill('sv_y', sv_mc_truth.y)
							self.mc_truth_tree.fill('sv_z', sv_mc_truth.z)
							self.mc_truth_tree.fill('tv_tau_d_x', tv_tau_d_mc_truth.x)
							self.mc_truth_tree.fill('tv_tau_d_y', tv_tau_d_mc_truth.y)
							self.mc_truth_tree.fill('tv_tau_d_z', tv_tau_d_mc_truth.z)
							self.mc_truth_tree.fill('tv_d_x', tv_d_mc_truth.x)
							self.mc_truth_tree.fill('tv_d_y', tv_d_mc_truth.y)
							self.mc_truth_tree.fill('tv_d_z', tv_d_mc_truth.z)

							self.mc_truth_tree.fill('b_px', b_mc_truth.p.px)
							self.mc_truth_tree.fill('b_py', b_mc_truth.p.py)
							self.mc_truth_tree.fill('b_pz', b_mc_truth.p.pz)

							self.mc_truth_tree.fill('kstar_px', kstar_mc_truth.p.px)
							self.mc_truth_tree.fill('kstar_py', kstar_mc_truth.p.py)
							self.mc_truth_tree.fill('kstar_pz', kstar_mc_truth.p.pz)

							self.mc_truth_tree.fill('k_q', k_mc_truth.charge)
							self.mc_truth_tree.fill('k_px', k_mc_truth.p.px)
							self.mc_truth_tree.fill('k_py', k_mc_truth.p.py)
							self.mc_truth_tree.fill('k_pz', k_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi_k_q', pi_k_mc_truth.charge)
							self.mc_truth_tree.fill('pi_k_px', pi_k_mc_truth.p.px)
							self.mc_truth_tree.fill('pi_k_py', pi_k_mc_truth.p.py)
							self.mc_truth_tree.fill('pi_k_pz', pi_k_mc_truth.p.pz)

							self.mc_truth_tree.fill('dplus_px', dplus_mc_truth.p.px)
							self.mc_truth_tree.fill('dplus_py', dplus_mc_truth.p.py)
							self.mc_truth_tree.fill('dplus_pz', dplus_mc_truth.p.pz)

							self.mc_truth_tree.fill('tau_d_px', tau_d_mc_truth.p.px)
							self.mc_truth_tree.fill('tau_d_py', tau_d_mc_truth.p.py)
							self.mc_truth_tree.fill('tau_d_pz', tau_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi1_tau_d_q', pi1_tau_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi1_tau_d_px', pi1_tau_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi1_tau_d_py', pi1_tau_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi1_tau_d_pz', pi1_tau_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi2_tau_d_q', pi2_tau_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi2_tau_d_px', pi2_tau_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi2_tau_d_py', pi2_tau_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi2_tau_d_pz', pi2_tau_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi3_tau_d_q', pi3_tau_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi3_tau_d_px', pi3_tau_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi3_tau_d_py', pi3_tau_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi3_tau_d_pz', pi3_tau_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('nu_tau_d_px', nu_tau_d_mc_truth.p.px)
							self.mc_truth_tree.fill('nu_tau_d_py', nu_tau_d_mc_truth.p.py)
							self.mc_truth_tree.fill('nu_tau_d_pz', nu_tau_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('nu_d_px', nu_d_mc_truth.p.px)
							self.mc_truth_tree.fill('nu_d_py', nu_d_mc_truth.p.py)
							self.mc_truth_tree.fill('nu_d_pz', nu_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('dminus_px', dminus_mc_truth.p.px)
							self.mc_truth_tree.fill('dminus_py', dminus_mc_truth.p.py)
							self.mc_truth_tree.fill('dminus_pz', dminus_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi1_d_q', pi1_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi1_d_px', pi1_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi1_d_py', pi1_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi1_d_pz', pi1_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi2_d_q', pi2_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi2_d_px', pi2_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi2_d_py', pi2_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi2_d_pz', pi2_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi3_d_q', pi3_d_mc_truth.charge)
							self.mc_truth_tree.fill('pi3_d_px', pi3_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi3_d_py', pi3_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi3_d_pz', pi3_d_mc_truth.p.pz)

							self.mc_truth_tree.fill('pi0_d_px', pi0_d_mc_truth.p.px)
							self.mc_truth_tree.fill('pi0_d_py', pi0_d_mc_truth.p.py)
							self.mc_truth_tree.fill('pi0_d_pz', pi0_d_mc_truth.p.pz)

							self.mc_truth_tree.tree.Fill()

							# filling event information
							self.tree.fill('event_number', event_number)
							self.tree.fill('n_particles', n_particles)

							self.tree.fill('pv_x', pv.x)
							self.tree.fill('pv_y', pv.y)
							self.tree.fill('pv_z', pv.z)
							self.tree.fill('sv_x', sv.x)
							self.tree.fill('sv_y', sv.y)
							self.tree.fill('sv_z', sv.z)
							self.tree.fill('tv_tauplus_x', tv_tauplus.x)
							self.tree.fill('tv_tauplus_y', tv_tauplus.y)
							self.tree.fill('tv_tauplus_z', tv_tauplus.z)
							self.tree.fill('tv_tauminus_x', tv_tauminus.x)
							self.tree.fill('tv_tauminus_y', tv_tauminus.y)
							self.tree.fill('tv_tauminus_z', tv_tauminus.z)

							self.tree.fill('pi1_tauplus_q', pi1_tauplus.charge)
							self.tree.fill('pi1_tauplus_px', pi1_tauplus.p.px)
							self.tree.fill('pi1_tauplus_py', pi1_tauplus.p.py)
							self.tree.fill('pi1_tauplus_pz', pi1_tauplus.p.pz)

							self.tree.fill('pi2_tauplus_q', pi2_tauplus.charge)
							self.tree.fill('pi2_tauplus_px', pi2_tauplus.p.px)
							self.tree.fill('pi2_tauplus_py', pi2_tauplus.p.py)
							self.tree.fill('pi2_tauplus_pz', pi2_tauplus.p.pz)

							self.tree.fill('pi3_tauplus_q', pi3_tauplus.charge)
							self.tree.fill('pi3_tauplus_px', pi3_tauplus.p.px)
							self.tree.fill('pi3_tauplus_py', pi3_tauplus.p.py)
							self.tree.fill('pi3_tauplus_pz', pi3_tauplus.p.pz)

							self.tree.fill('pi1_tauminus_q', pi1_tauminus.charge)
							self.tree.fill('pi1_tauminus_px', pi1_tauminus.p.px)
							self.tree.fill('pi1_tauminus_py', pi1_tauminus.p.py)
							self.tree.fill('pi1_tauminus_pz', pi1_tauminus.p.pz)

							self.tree.fill('pi2_tauminus_q', pi2_tauminus.charge)
							self.tree.fill('pi2_tauminus_px', pi2_tauminus.p.px)
							self.tree.fill('pi2_tauminus_py', pi2_tauminus.p.py)
							self.tree.fill('pi2_tauminus_pz', pi2_tauminus.p.pz)

							self.tree.fill('pi3_tauminus_q', pi3_tauminus.charge)
							self.tree.fill('pi3_tauminus_px', pi3_tauminus.p.px)
							self.tree.fill('pi3_tauminus_py', pi3_tauminus.p.py)
							self.tree.fill('pi3_tauminus_pz', pi3_tauminus.p.pz)

							self.tree.fill('k_q', k.charge)
							self.tree.fill('k_px', k.p.px)
							self.tree.fill('k_py', k.p.py)
							self.tree.fill('k_pz', k.p.pz)

							self.tree.fill('pi_k_q', pi_k.charge)
							self.tree.fill('pi_k_px', pi_k.p.px)
							self.tree.fill('pi_k_py', pi_k.p.py)
							self.tree.fill('pi_k_pz', pi_k.p.pz)

							self.tree.tree.Fill()

	def write(self, unusefulVar):
		self.rootfile.Write()
		self.rootfile.Close()

		pb_canvas = TCanvas('pb_canvas', 'B momentum', 600, 400)
		pb_canvas.cd()
		self.pb_hist.Draw()
		pb_canvas.Update()

		pvsv_distance_canvas = TCanvas('pvsv_distance_canvas', 'Distance between PV and SV', 600, 400)
		pvsv_distance_canvas.cd()
		self.pvsv_distance_hist.Draw()
		pvsv_distance_canvas.Update()

		max_svtv_distance_canvas = TCanvas('max_svtv_distance_canvas', 'Max distance between SV and TV', 600, 400)
		max_svtv_distance_canvas.cd()
		self.max_svtv_distance_hist.Draw()
		max_svtv_distance_canvas.Update()

		print('Total decays processed: {}'.format(self.counter))
		print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
		print('Efficiency:\n\tMomentum of B cut: {:.3f}\n\tDistance between PV and SV cut: {:.3f}\n\tMax distance between SV and TV cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter), float(self.pvsv_distance_counter)/float(self.counter), float(self.max_svtv_distance_counter)/float(self.counter)))
		raw_input('Press ENTER when finished')