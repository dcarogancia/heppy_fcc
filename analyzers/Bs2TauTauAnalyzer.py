## Analyzer of B0d -> K*0 tau+ tau- events
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

class Bs2TauTauAnalyzer(Analyzer):
	def beginLoop(self, setup):
		self.start_time = time.time()
		self.last_timestamp = time.time()

		self.counter = 0 # Total number of processed decays
		self.pb_counter = 0 # Number of events with B momentum > 25 GeV

		gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

		# histograms to visualize cuts
		self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)

		super(Bs2TauTauAnalyzer, self).beginLoop(setup)
		self.rootfile = TFile('/'.join([self.dirName, 'output.root']), 'recreate')

		# tree to store MC truth values and its branches
		self.mc_truth_tree = Tree(self.cfg_ana.mc_truth_tree_name, self.cfg_ana.mc_truth_tree_title)
		self.mc_truth_tree.var('n_particles')
		self.mc_truth_tree.var('event_number')
		self.mc_truth_tree.var('pv_x')
		self.mc_truth_tree.var('pv_y')
		self.mc_truth_tree.var('pv_z')
		self.mc_truth_tree.var('tv_tauplus_x')
		self.mc_truth_tree.var('tv_tauplus_y')
		self.mc_truth_tree.var('tv_tauplus_z')
		self.mc_truth_tree.var('tv_tauminus_x')
		self.mc_truth_tree.var('tv_tauminus_y')
		self.mc_truth_tree.var('tv_tauminus_z')
		self.mc_truth_tree.var('b_px')
		self.mc_truth_tree.var('b_py')
		self.mc_truth_tree.var('b_pz')
		self.mc_truth_tree.var('opposite_b_quark_mc_truth_px')
		self.mc_truth_tree.var('opposite_b_quark_mc_truth_py')
		self.mc_truth_tree.var('opposite_b_quark_mc_truth_pz')
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

		# same for smeared values
		self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)
		self.tree.var('n_particles')
		self.tree.var('event_number')
		self.tree.var('pv_x')
		self.tree.var('pv_y')
		self.tree.var('pv_z')
		self.tree.var('tv_tauplus_x')
		self.tree.var('tv_tauplus_y')
		self.tree.var('tv_tauplus_z')
		self.tree.var('tv_tauminus_x')
		self.tree.var('tv_tauminus_y')
		self.tree.var('tv_tauminus_z')
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
		b_mc_truth = None # B0s particle (MC truth)
		opposite_b_quark_mc_truth = None # b quark opposite to B0s (MC truth)
		tauplus_mc_truth = None # tau+ from B0s decay (MC truth)
		tauminus_mc_truth = None # tau- from B0s decay (MC truth)
		pi1_tauplus_mc_truth = None # pi from tau+ decay (MC truth)
		pi2_tauplus_mc_truth = None # pi from tau+ decay (MC truth)
		pi3_tauplus_mc_truth = None # pi from tau+ decay (MC truth)
		nu_tauplus_mc_truth = None # nu from tau+ decay (MC truth)
		pi1_tauminus_mc_truth = None # pi from tau- decay (MC truth)
		pi2_tauminus_mc_truth = None # pi from tau- decay (MC truth)
		pi3_tauminus_mc_truth = None # pi from tau- decay (MC truth)
		nu_tauminus_mc_truth = None # nu from tau- decay (MC truth)
		pv_mc_truth = None # primary vertex (MC truth)
		tv_tauplus_mc_truth = None # tau+ decay vertex (MC truth)
		tv_tauminus_mc_truth = None # tau- decay vertex (MC truth)

		pi1_tauplus = None # pi from tau+ decay
		pi2_tauplus = None # pi from tau+ decay
		pi3_tauplus = None # pi from tau+ decay
		pi1_tauminus = None # pi from tau- decay
		pi2_tauminus = None # pi from tau- decay
		pi3_tauminus = None # pi from tau- decay
		pv = None # primary vertex
		tv_tauplus = None # tau+ decay vertex
		tv_tauminus = None # tau- decay vertex

		pb = 0. # B momentum

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

				if pb > 25.: # select only events with large momentum of the B
					self.pb_counter += 1

					# looking for opposite b quark. This is a dirty hack. Works only because both PYTHIA/HepMC and PODIO store particles ordered. But IT'S NOT GUARANTEED
					index = 0
					while opposite_b_quark_mc_truth == None and index < len(ptcs):
						if abs(ptcs[index].pdgid) == 5 and numpy.dot([b_mc_truth.p.px, b_mc_truth.p.py, b_mc_truth.p.pz], [ptcs[index].p.px, ptcs[index].p.py, ptcs[index].p.pz]) < 0:
							opposite_b_quark_mc_truth = ptcs[index]

						index += 1

					pv_mc_truth = b_mc_truth.start_vertex
					pv = copy.deepcopy(pv_mc_truth)

					for ptc_gen2 in ptcs:
						# looking for tauplus
						if ptc_gen2.pdgid == -15 and ptc_gen2.start_vertex == b_mc_truth.end_vertex:
							tauplus_mc_truth = ptc_gen2
							tv_tauplus_mc_truth = ptc_gen2.end_vertex
							tv_tauplus = copy.deepcopy(tv_tauplus_mc_truth) # copy is needed in order to keep initial vertex properties after smearing

						# looking for tauMinus
						if ptc_gen2.pdgid == 15 and ptc_gen2.start_vertex == b_mc_truth.end_vertex:
							tauminus_mc_truth = ptc_gen2
							tv_tauminus_mc_truth = ptc_gen2.end_vertex
							tv_tauminus = copy.deepcopy(tv_tauminus_mc_truth) # copy is needed in order to keep initial vertex properties after smearing

					# looking for pions and nu from tau+ decay
					pis_tauplus_mc_truth = list([])
					pis_tauplus = list([])
					for ptc_gen3 in ptcs:
						if abs(ptc_gen3.pdgid) == 211 and ptc_gen3.start_vertex == tauplus_mc_truth.end_vertex:
							pis_tauplus_mc_truth.append(ptc_gen3)
							pis_tauplus.append(copy.deepcopy(ptc_gen3)) # copy is needed in order to keep initial particle properties after smearing

						if ptc_gen3.pdgid == 16:
							nu_tauplus_mc_truth = ptc_gen3

					if len(pis_tauplus_mc_truth) == 3:
						pi1_tauplus_mc_truth, pi2_tauplus_mc_truth, pi3_tauplus_mc_truth = pis_tauplus_mc_truth[0], pis_tauplus_mc_truth[1], pis_tauplus_mc_truth[2]
					if len(pis_tauplus) == 3:
						pi1_tauplus, pi2_tauplus, pi3_tauplus = pis_tauplus[0], pis_tauplus[1], pis_tauplus[2]

					# looking for pions and nu from tau- decay
					pis_tauminus_mc_truth = list([])
					pis_tauminus = list([])
					for ptc_gen3 in ptcs:
						if abs(ptc_gen3.pdgid) == 211 and ptc_gen3.start_vertex == tauminus_mc_truth.end_vertex:
							pis_tauminus_mc_truth.append(ptc_gen3)
							pis_tauminus.append(copy.deepcopy(ptc_gen3)) # copy is needed in order to keep initial particle properties after smearing

						if ptc_gen3.pdgid == -16:
							nu_tauminus_mc_truth = ptc_gen3

					if len(pis_tauminus_mc_truth) == 3:
						pi1_tauminus_mc_truth, pi2_tauminus_mc_truth, pi3_tauminus_mc_truth = pis_tauminus_mc_truth[0], pis_tauminus_mc_truth[1], pis_tauminus_mc_truth[2]
					if len(pis_tauplus) == 3:
						pi1_tauminus, pi2_tauminus, pi3_tauminus = pis_tauminus[0], pis_tauminus[1], pis_tauminus[2]

					# applying smearing
					if self.cfg_ana.smear_momentum:
						pi1_tauplus.p = smear_momentum(pi1_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
						pi2_tauplus.p = smear_momentum(pi2_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
						pi3_tauplus.p = smear_momentum(pi3_tauplus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
						pi1_tauminus.p = smear_momentum(pi1_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
						pi2_tauminus.p = smear_momentum(pi2_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
						pi3_tauminus.p = smear_momentum(pi3_tauminus.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
					if self.cfg_ana.smear_pv:
						pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)
					if self.cfg_ana.smear_tv:
						tv_tauplus = smear_vertex(tv_tauplus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
						tv_tauminus = smear_vertex(tv_tauminus, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)

						# to keep consistency
						pi1_tauplus.start_vertex, pi2_tauplus.start_vertex, pi3_tauplus.start_vertex = tv_tauplus, tv_tauplus, tv_tauplus
						pi1_tauminus.start_vertex, pi2_tauminus.start_vertex, pi3_tauminus.start_vertex = tv_tauminus, tv_tauminus, tv_tauminus

					if pi1_tauplus.is_valid() and pi2_tauplus.is_valid() and pi3_tauplus.is_valid() and pi1_tauminus.is_valid() and pi2_tauminus.is_valid() and pi3_tauminus.is_valid():
						# filling histogram
						self.pb_hist.Fill(pb)

						# filling MC truth information
						self.mc_truth_tree.fill('event_number', event_number)
						self.mc_truth_tree.fill('n_particles', n_particles)

						self.mc_truth_tree.fill('pv_x', pv_mc_truth.x)
						self.mc_truth_tree.fill('pv_y', pv_mc_truth.y)
						self.mc_truth_tree.fill('pv_z', pv_mc_truth.z)
						self.mc_truth_tree.fill('tv_tauplus_x', tv_tauplus_mc_truth.x)
						self.mc_truth_tree.fill('tv_tauplus_y', tv_tauplus_mc_truth.y)
						self.mc_truth_tree.fill('tv_tauplus_z', tv_tauplus_mc_truth.z)
						self.mc_truth_tree.fill('tv_tauminus_x', tv_tauminus_mc_truth.x)
						self.mc_truth_tree.fill('tv_tauminus_y', tv_tauminus_mc_truth.y)
						self.mc_truth_tree.fill('tv_tauminus_z', tv_tauminus_mc_truth.z)

						self.mc_truth_tree.fill('b_px', b_mc_truth.p.px)
						self.mc_truth_tree.fill('b_py', b_mc_truth.p.py)
						self.mc_truth_tree.fill('b_pz', b_mc_truth.p.pz)

						self.mc_truth_tree.fill('opposite_b_quark_mc_truth_px', opposite_b_quark_mc_truth.p.px)
						self.mc_truth_tree.fill('opposite_b_quark_mc_truth_py', opposite_b_quark_mc_truth.p.py)
						self.mc_truth_tree.fill('opposite_b_quark_mc_truth_pz', opposite_b_quark_mc_truth.p.pz)

						self.mc_truth_tree.fill('tauplus_px', tauplus_mc_truth.p.px)
						self.mc_truth_tree.fill('tauplus_py', tauplus_mc_truth.p.py)
						self.mc_truth_tree.fill('tauplus_pz', tauplus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi1_tauplus_q', pi1_tauplus_mc_truth.charge)
						self.mc_truth_tree.fill('pi1_tauplus_px', pi1_tauplus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi1_tauplus_py', pi1_tauplus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi1_tauplus_pz', pi1_tauplus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi2_tauplus_q', pi2_tauplus_mc_truth.charge)
						self.mc_truth_tree.fill('pi2_tauplus_px', pi2_tauplus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi2_tauplus_py', pi2_tauplus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi2_tauplus_pz', pi2_tauplus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi3_tauplus_q', pi3_tauplus_mc_truth.charge)
						self.mc_truth_tree.fill('pi3_tauplus_px', pi3_tauplus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi3_tauplus_py', pi3_tauplus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi3_tauplus_pz', pi3_tauplus_mc_truth.p.pz)

						self.mc_truth_tree.fill('nu_tauplus_px', nu_tauplus_mc_truth.p.px)
						self.mc_truth_tree.fill('nu_tauplus_py', nu_tauplus_mc_truth.p.py)
						self.mc_truth_tree.fill('nu_tauplus_pz', nu_tauplus_mc_truth.p.pz)

						self.mc_truth_tree.fill('tauminus_px', tauminus_mc_truth.p.px)
						self.mc_truth_tree.fill('tauminus_py', tauminus_mc_truth.p.py)
						self.mc_truth_tree.fill('tauminus_pz', tauminus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi1_tauminus_q', pi1_tauminus_mc_truth.charge)
						self.mc_truth_tree.fill('pi1_tauminus_px', pi1_tauminus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi1_tauminus_py', pi1_tauminus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi1_tauminus_pz', pi1_tauminus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi2_tauminus_q', pi2_tauminus_mc_truth.charge)
						self.mc_truth_tree.fill('pi2_tauminus_px', pi2_tauminus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi2_tauminus_py', pi2_tauminus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi2_tauminus_pz', pi2_tauminus_mc_truth.p.pz)

						self.mc_truth_tree.fill('pi3_tauminus_q', pi3_tauminus_mc_truth.charge)
						self.mc_truth_tree.fill('pi3_tauminus_px', pi3_tauminus_mc_truth.p.px)
						self.mc_truth_tree.fill('pi3_tauminus_py', pi3_tauminus_mc_truth.p.py)
						self.mc_truth_tree.fill('pi3_tauminus_pz', pi3_tauminus_mc_truth.p.pz)

						self.mc_truth_tree.fill('nu_tauminus_px', nu_tauminus_mc_truth.p.px)
						self.mc_truth_tree.fill('nu_tauminus_py', nu_tauminus_mc_truth.p.py)
						self.mc_truth_tree.fill('nu_tauminus_pz', nu_tauminus_mc_truth.p.pz)

						self.mc_truth_tree.tree.Fill()

						# filling event information
						self.tree.fill('event_number', event_number)
						self.tree.fill('n_particles', n_particles)

						self.tree.fill('pv_x', pv.x)
						self.tree.fill('pv_y', pv.y)
						self.tree.fill('pv_z', pv.z)
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

						self.tree.tree.Fill()

	def write(self, unusefulVar):
		self.rootfile.Write()
		self.rootfile.Close()

		pb_canvas = TCanvas('pb_canvas', 'B momentum', 600, 400)
		pb_canvas.cd()
		self.pb_hist.Draw()
		pb_canvas.Update()

		print('Total decays processed: {}'.format(self.counter))
		print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
		print('Efficiency:\n\tMomentum of B cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter)))
		raw_input('Press ENTER when finished')
