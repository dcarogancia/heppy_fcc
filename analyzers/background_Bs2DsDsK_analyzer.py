## Analyzer of Bs -> Ds Ds K* events
#  It is supposed to be used within heppy_fcc framework

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree

import math
import copy

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

class background_Bs2DsDsK_analyzer(Analyzer):
	def beginLoop(self, setup):
		self.counter = 0 # Total number of processed decays
		self.pb_counter = 0 # Number of events with B momentum > 25 GeV
		self.fdb_counter = 0 # Number of events with B flight distance > 1 mm
		self.fdtau_counter = 0 # Number of events with any tau flight distance > 0.5 mm

		gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath)
		self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)
		self.fdb_hist = TH1F('fdb_hist', 'FD_{B}', 500, 0, 10)
		self.fdtau_hist = TH1F('fdtau_hist', 'Max FD_{#tau}', 500, 0, 5)

		super(background_Bs2DsDsK_analyzer, self).beginLoop(setup)
		self.rootfile = TFile('/'.join([self.dirName, 'output.root']), 'recreate')
		self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)

		# GLOBAL EVENT INFORMATION
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

		# DECAY INFORMATION

		# K from K* decay
		self.tree.var('k_px')
		self.tree.var('k_py')
		self.tree.var('k_pz')
		self.tree.var('k_q')

		# pi from K* decay
		self.tree.var('pi_k_px')
		self.tree.var('pi_k_py')
		self.tree.var('pi_k_pz')
		self.tree.var('pi_k_q')

		# pi from tau+ decay
		self.tree.var('pi1_tauplus_px')
		self.tree.var('pi1_tauplus_py')
		self.tree.var('pi1_tauplus_pz')
		self.tree.var('pi1_tauplus_q')

		# pi from tau+ decay
		self.tree.var('pi2_tauplus_px')
		self.tree.var('pi2_tauplus_py')
		self.tree.var('pi2_tauplus_pz')
		self.tree.var('pi2_tauplus_q')

		# pi from tau+ decay
		self.tree.var('pi3_tauplus_px')
		self.tree.var('pi3_tauplus_py')
		self.tree.var('pi3_tauplus_pz')
		self.tree.var('pi3_tauplus_q')

		# pi from tau- decay
		self.tree.var('pi1_tauminus_px')
		self.tree.var('pi1_tauminus_py')
		self.tree.var('pi1_tauminus_pz')
		self.tree.var('pi1_tauminus_q')

		# pi from tau- decay
		self.tree.var('pi2_tauminus_px')
		self.tree.var('pi2_tauminus_py')
		self.tree.var('pi2_tauminus_pz')
		self.tree.var('pi2_tauminus_q')

		# pi from tau- decay
		self.tree.var('pi3_tauminus_px')
		self.tree.var('pi3_tauminus_py')
		self.tree.var('pi3_tauminus_pz')
		self.tree.var('pi3_tauminus_q')

	def process(self, event):
		b = Particle() # B particle
		k = Particle() # K from K* decay
		pi_k = Particle() # pi from K* decay
		tauplus = Particle() # tau+
		tauminus = Particle() # tau-
		pi1_tauplus = Particle() # pi from tau+ decay
		pi2_tauplus = Particle() # pi from tau+ decay
		pi3_tauplus = Particle() # pi from tau+ decay
		pi1_tauminus = Particle() # pi from tau- decay
		pi2_tauminus = Particle() # pi from tau- decay
		pi3_tauminus = Particle() # pi from tau- decay

		pv = Vertex() # primary vertex
		sv = Vertex() # secondary vertex
		tv_tauplus = Vertex() # tau+ decay vertex
		tv_tauminus = Vertex() # tau- decay vertex

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

				b = copy.deepcopy(ptc_gen1) # copy is needed in order to keep initial particle properties after smearing

				pb = b.p.absvalue()

				if self.cfg_ana.smear_momentum:
					b.p = smear_momentum(b.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

				if pb > 25.: # Select only events with large momentum of the B
					self.pb_counter += 1

					pv = copy.deepcopy(b.start_vertex) # copy is needed in order to keep initial vertex properties after smearing
					if self.cfg_ana.smear_pv:
						pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)
					sv = copy.deepcopy(b.end_vertex)  # copy is needed in order to keep initial vertex properties after smearing
					if self.cfg_ana.smear_sv:
						sv = smear_vertex(sv, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)

					pvsv_distance = math.sqrt((b.end_vertex.x - b.start_vertex.x) ** 2 + (b.end_vertex.y - b.start_vertex.y) ** 2 + (b.end_vertex.z - b.start_vertex.z) ** 2)

					if pvsv_distance > 1.: # Select only events with long flight distance of the B
						self.fdb_counter += 1

						for ptc_gen2 in ptcs:
							# looking for Ds+
							if ptc_gen2.pdgid == 431 and ptc_gen2.start_vertex == b.end_vertex:
								# looking for tauplus
								for ptc_gen3 in ptcs:
									if ptc_gen3.pdgid == -15 and ptc_gen3.start_vertex == ptc_gen2.end_vertex:
										tauplus = copy.deepcopy(ptc_gen3) # copy is needed in order to keep initial particle properties after smearing

										if self.cfg_ana.smear_momentum:
											tauplus.p = smear_momentum(tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

										tv_tauplus = copy.deepcopy(tauplus.end_vertex) # copy is needed in order to keep initial vertex properties after smearing

										if self.cfg_ana.smear_tv:
											tv_tauplus = smear_vertex(tv_tauplus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)

										svtv_tauplus_distance = math.sqrt((tauplus.end_vertex.x - b.end_vertex.x) ** 2 + (tauplus.end_vertex.y - b.end_vertex.y) ** 2 + (tauplus.end_vertex.z - b.end_vertex.z) ** 2)

										# looking for pis_tauplus
										pis_tauplus = list([])
										for ptc_gen4 in ptcs:
											if abs(ptc_gen4.pdgid) == 211 and ptc_gen4.start_vertex == tauplus.end_vertex:
												pi_tauplus = copy.deepcopy(ptc_gen4) # copy is needed in order to keep initial particle properties after smearing

												if self.cfg_ana.smear_momentum:
													pi_tauplus.p = smear_momentum(pi_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

												pis_tauplus.append(pi_tauplus)

										if len(pis_tauplus) == 3:
											pi1_tauplus, pi2_tauplus, pi3_tauplus = pis_tauplus[0], pis_tauplus[1], pis_tauplus[2]

							# looking for Ds-
							if ptc_gen2.pdgid == -431 and ptc_gen2.start_vertex == b.end_vertex:
								# looking for tauMinus
								for ptc_gen3 in ptcs:
									if ptc_gen3.pdgid == 15 and ptc_gen3.start_vertex == ptc_gen2.end_vertex:
										tauminus = copy.deepcopy(ptc_gen3) # copy is needed in order to keep initial particle properties after smearing

										if self.cfg_ana.smear_momentum:
											tauminus.p = smear_momentum(tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

										tv_tauminus = copy.deepcopy(tauminus.end_vertex) # copy is needed in order to keep initial vertex properties after smearing

										if self.cfg_ana.smear_tv:
											tv_tauminus = smear_vertex(tv_tauminus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)

										svtv_tauminus_distance = math.sqrt((tauminus.end_vertex.x - b.end_vertex.x) ** 2 + (tauminus.end_vertex.y - b.end_vertex.y) ** 2 + (tauminus.end_vertex.z - b.end_vertex.z) ** 2)

										# looking for pi_tauminus
										pis_tauminus = list([])
										for ptc_gen4 in ptcs:
											if abs(ptc_gen4.pdgid) == 211 and ptc_gen4.start_vertex == tauminus.end_vertex:
												pi_tauminus = copy.deepcopy(ptc_gen4) # copy is needed in order to keep initial particle properties after smearing

												if self.cfg_ana.smear_momentum:
													pi_tauminus.p = smear_momentum(pi_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

												pis_tauminus.append(pi_tauminus)

										if len(pis_tauminus) == 3:
											pi1_tauminus, pi2_tauminus, pi3_tauminus = pis_tauminus[0], pis_tauminus[1], pis_tauminus[2]

							# looking for K*
							if abs(ptc_gen2.pdgid) == 313 and ptc_gen2.start_vertex == b.end_vertex:
								# looking for K
								for ptc_gen3 in ptcs:
									if abs(ptc_gen3.pdgid) == 321 and ptc_gen3.start_vertex == ptc_gen2.end_vertex:
										k = copy.deepcopy(ptc_gen3) # copy is needed in order to keep initial particle properties after smearing
										if self.cfg_ana.smear_momentum:
											k.p = smear_momentum(k.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

								# looking for pi_K
								for ptc_gen3 in ptcs:
									if abs(ptc_gen3.pdgid) == 211 and ptc_gen3.start_vertex == ptc_gen2.end_vertex:
										pi_k = copy.deepcopy(ptc_gen3) # copy is needed in order to keep initial particle properties after smearing
										if self.cfg_ana.smear_momentum:
											pi_k.p = smear_momentum(pi_k.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)

						if max(svtv_tauplus_distance, svtv_tauminus_distance) > 0.5 and k.is_valid() and pi_k.is_valid() and pi1_tauplus.is_valid() and pi2_tauplus.is_valid() and pi3_tauplus.is_valid() and pi1_tauminus.is_valid() and pi2_tauminus.is_valid() and pi3_tauminus.is_valid(): # Select only events with long flight distance of tau

							self.fdtau_counter += 1

							# filling histograms
							self.fdb_hist.Fill(pvsv_distance)
							self.pb_hist.Fill(pb)
							self.fdtau_hist.Fill(max(svtv_tauplus_distance, svtv_tauminus_distance))

							# filling general event information
							self.tree.fill('event_number', event_number)
							self.tree.fill('n_particles', n_particles)

							# filling particle information
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

		fdb_canvas = TCanvas('fdb_canvas', 'B flight distance', 600, 400)
		fdb_canvas.cd()
		self.fdb_hist.Draw()
		fdb_canvas.Update()

		fdtau_canvas = TCanvas('fdtau_canvas', 'Max tau flight distance', 600, 400)
		fdtau_canvas.cd()
		self.fdtau_hist.Draw()
		fdtau_canvas.Update()

		print('Total decays processed: {}'.format(self.counter))

		print('Efficiency:\n\tMomentum of B cut: {}\n\tDistance between PV and SV cut: {}\n\tMax distance between SV and TV cut: {}'.format (float(self.pb_counter)/float(self.counter), float(self.fdb_counter)/float(self.counter), float(self.fdtau_counter)/float(self.counter)))
		raw_input('Press ENTER when finished')
