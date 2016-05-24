#!/usr/bin/env python

"""
	Contains CommonAnalyzer class definition

	CommonAnalyzer - an utility class that embraces the features common for all analyzers
"""

import time

from ROOT import gROOT, TFile, TH1F, TCanvas

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree

class CommonAnalyzer(Analyzer):
	"""
		CommonAnalyzer - an utility class that embraces the features common for all analyzers

		Attributes:
		rootfile (ROOT.TFile): output ROOT file
		tree (heppy.statistics.Tree): the tree with visible (smeared) values
		mc_truth_tree (heppy.statistics.Tree): the tree with MC truth values
		counter (int): total number of processed decays
		pb_counter (int): number of events that survived B momentum cut
		pvsv_distance_counter (int): number of events that survived SV-PV distance cut
		max_svtv_distance_counter (int): number of events that survived TV-SV distance cut
		pb_hist (ROOT.TH1F): histogram to visualize B momentum cut
		pvsv_distance_hist (ROOT.TH1F): histogram to visualize SV-PV distance cut
		max_svtv_distance_hist (ROOT.TH1F): histogram to visualize TV-SV distance cut
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
		super(CommonAnalyzer, self).__init__(cfg_ana, cfg_comp, looper_name)

		self.rootfile = TFile('/'.join([self.dirName, 'output.root']), 'recreate')

		# tree to store smeared values
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

		# MC truth tree
		self.mc_truth_tree = Tree(self.cfg_ana.mc_truth_tree_name, self.cfg_ana.mc_truth_tree_title)

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

		super(CommonAnalyzer, self).beginLoop(setup)

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
