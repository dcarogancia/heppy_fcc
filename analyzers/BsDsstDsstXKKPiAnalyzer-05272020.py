## Analyzer of B0d -> K*0 tau+ tau- events
#  It is supposed to be used within heppy_fcc framework

from heppy.framework.analyzer import Analyzer
from heppy.statistics.tree import Tree

import math
import copy
import time

import numpy as np

from ROOT import gROOT
from ROOT import TFile
from ROOT import TH1F
from ROOT import TCanvas

from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex
from heppy_fcc.utility.Particle import Particle

#From FastFit
from PyFastFit import FastFit
import scipy.stats


def smear_momentum(p, px_resolution, py_resolution, pz_resolution):
    return Momentum(np.random.normal(p.px, px_resolution), np.random.normal(p.py, py_resolution), np.random.normal(p.pz, pz_resolution))

def momentum_res(pt):
    return (2.0e-5/pt + 1.0e-3)

def getEt(pEn):
    return np.sqrt(pEn.p.absvalue*pEn.p.absvalue + pEn.mass*pEn.mass)

def fdistance(pVtx):
    return np.sqrt((pVtx.end_vertex.x - pVtx.start_vertex.x)**2 + (pVtx.end_vertex.y - pVtx.start_vertex.y)**2 + (pVtx.end_vertex.z - pVtx.start_vertex.z)**2)

def smear_vertex(v, x_resolution, y_resolution, z_resolution):
    return Vertex(np.random.normal(v.x, x_resolution), np.random.normal(v.y, y_resolution), np.random.normal(v.z, z_resolution))

def Ds_PCA(r_Ds,r_mu,p_Ds,p_mu):
    pmag_Ds = 1/np.sqrt(p_Ds[0]**2 + p_Ds[1]**2 + p_Ds[2]**2)
    pmag_mu = 1/np.sqrt(p_mu.p.px**2 + p_mu.p.py**2 + p_mu.p.pz**2)
    r = np.subtract([r_Ds[0],r_Ds[1],r_Ds[2]],[r_mu.x,r_mu.y,r_mu.z])
    t_Ds = (pmag_Ds*pmag_mu*pmag_mu*(np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[p_mu.p.px,p_mu.p.py,p_mu.p.pz]))*(np.dot([p_mu.p.px,p_mu.p.py,p_mu.p.pz],[r[0],r[1],r[2]])) - pmag_Ds*np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[r[0],r[1],r[2]]))/(1 - (pmag_Ds*pmag_mu*np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[p_mu.p.px,p_mu.p.py,p_mu.p.pz]))**2)
    
    return t_Ds

def mu_PCA(r_Ds,r_mu,p_Ds,p_mu):
    pmag_Ds = 1/np.sqrt(p_Ds[0]**2 + p_Ds[1]**2 + p_Ds[2]**2)
    pmag_mu = 1/np.sqrt(p_mu.p.px**2 + p_mu.p.py**2 + p_mu.p.pz**2)
    r = np.subtract([r_Ds[0],r_Ds[1],r_Ds[2]],[r_mu.x,r_mu.y,r_mu.z])
    t_mu = (pmag_mu*np.dot([p_mu.p.px,p_mu.p.py,p_mu.p.pz],[r[0],r[1],r[2]]) - pmag_Ds*pmag_Ds*pmag_mu*(np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[p_mu.p.px,p_mu.p.py,p_mu.p.pz]))*(np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[r[0],r[1],r[2]])))/(1 - (pmag_Ds*pmag_mu*np.dot([p_Ds[0],p_Ds[1],p_Ds[2]],[p_mu.p.px,p_mu.p.py,p_mu.p.pz]))**2)
    
    return t_mu

class BsDsstDsstXKKPiAnalyzer(Analyzer):
    def beginLoop(self, setup):
        self.start_time = time.time()
        self.last_timestamp = time.time()

        self.counter = 0 # Total number of processed decays
        self.pb_counter = 0 # Number of events with B momentum > 25 GeV
        self.Ds_counter = 0
        self.Dsst_counter = 0
        self.Ds0st_counter = 0
        self.Ds1_counter = 0
        self.Ds1prime_counter = 0
        self.DBop_counter = 0
        self.K_counter = 0
        self.Bmu_counter = 0
        self.taumu_counter = 0
        self.mu_counter = 0
        self.munu_counter = 0
        self.tau_counter = 0
        self.pi_counter = 0
        self.pi0_counter = 0
        self.gamma_counter = 0


        gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

        # histograms to visualize cuts
        self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)

        super(BsDsstDsstXKKPiAnalyzer, self).beginLoop(setup)
        #self.rootfile = TFile('/'.join([self.dirName, 'Bd2DmunuKpipi-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsstDsstbkgAnl-evtshp-500k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsstDsstbkgAnl_vtxfit-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsstDsstbkgAnl_vtxfit_bkg-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsstDsstbkgAnl_vtxfit_bkg_newevtgen-86.9k.root']), 'recreate')
        self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsstDsst_KKpi_bkg_evtpdl2019-100k.root']), 'recreate')
        
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
        self.mc_truth_tree.var('tv_x')
        self.mc_truth_tree.var('tv_y')
        self.mc_truth_tree.var('tv_z')
        self.mc_truth_tree.var('pvDs_x')
        self.mc_truth_tree.var('pvDs_y')
        self.mc_truth_tree.var('pvDs_z')
        self.mc_truth_tree.var('svDs_x')
        self.mc_truth_tree.var('svDs_y')
        self.mc_truth_tree.var('svDs_z')
        self.mc_truth_tree.var('pvsv_distance')
        self.mc_truth_tree.var('bquark_px')
        self.mc_truth_tree.var('bquark_py')
        self.mc_truth_tree.var('bquark_pz')
        self.mc_truth_tree.var('bquark_npx')
        self.mc_truth_tree.var('bquark_npy')
        self.mc_truth_tree.var('bquark_npz')
        self.mc_truth_tree.var('bquark_p') 
        self.mc_truth_tree.var('bquark_E') 
        self.mc_truth_tree.var('bquarkSS_px')
        self.mc_truth_tree.var('bquarkSS_py')
        self.mc_truth_tree.var('bquarkSS_pz')
        self.mc_truth_tree.var('bquarkSS_npx')
        self.mc_truth_tree.var('bquarkSS_npy')
        self.mc_truth_tree.var('bquarkSS_npz')
        self.mc_truth_tree.var('bquarkSS_p') 
        self.mc_truth_tree.var('bquarkSS_E') 
        self.mc_truth_tree.var('bquarkOS_px')
        self.mc_truth_tree.var('bquarkOS_py')
        self.mc_truth_tree.var('bquarkOS_pz')
        self.mc_truth_tree.var('bquarkOS_npx')
        self.mc_truth_tree.var('bquarkOS_npy')
        self.mc_truth_tree.var('bquarkOS_npz')
        self.mc_truth_tree.var('bquarkOS_p') 
        self.mc_truth_tree.var('bquarkOS_E')
        self.mc_truth_tree.var('B_px')
        self.mc_truth_tree.var('B_py')
        self.mc_truth_tree.var('B_pz')
        self.mc_truth_tree.var('B_m')
        self.mc_truth_tree.var('B_p')
        self.mc_truth_tree.var('B_ID')
    	self.mc_truth_tree.var('B_q')
    	self.mc_truth_tree.var('Ds_E')
    	self.mc_truth_tree.var('Ds_px')
        self.mc_truth_tree.var('Ds_py')
        self.mc_truth_tree.var('Ds_pz')
        self.mc_truth_tree.var('Ds_p')
        self.mc_truth_tree.var('Ds_pT')
    	self.mc_truth_tree.var('Ds1_E')
    	self.mc_truth_tree.var('Ds1_px')
        self.mc_truth_tree.var('Ds1_py')
        self.mc_truth_tree.var('Ds1_pz')
        self.mc_truth_tree.var('Ds1_p')
        self.mc_truth_tree.var('Ds1_q')
    	self.mc_truth_tree.var('Ds1_m')
    	self.mc_truth_tree.var('Ds1_ID')
    	self.mc_truth_tree.var('Ds2_E')
    	self.mc_truth_tree.var('Ds2_px')
        self.mc_truth_tree.var('Ds2_py')
        self.mc_truth_tree.var('Ds2_pz')
        self.mc_truth_tree.var('Ds2_p')
        self.mc_truth_tree.var('Ds2_q')
    	self.mc_truth_tree.var('Ds2_m')
    	self.mc_truth_tree.var('Ds2_ID')
    	self.mc_truth_tree.var('mu_px')
        self.mc_truth_tree.var('mu_py')
        self.mc_truth_tree.var('mu_pz')
        self.mc_truth_tree.var('mu_E')        
    	self.mc_truth_tree.var('mu_m')
    	self.mc_truth_tree.var('mu_p')
    	self.mc_truth_tree.var('mu_pT')
    	self.mc_truth_tree.var('mu_ID')
        self.mc_truth_tree.var('mu_q')
    	self.mc_truth_tree.var('BsDs1Ds2_m')
    	self.mc_truth_tree.var('Ds_KKpi_m')
    	self.mc_truth_tree.var('Bs_KKpimu_m')
    	self.mc_truth_tree.var('Bs_KKpimu_mcorr')
    	self.mc_truth_tree.var('Pvis_SS_p')
    	self.mc_truth_tree.var('Pvis_SS_px')
    	self.mc_truth_tree.var('Pvis_SS_py')
    	self.mc_truth_tree.var('Pvis_SS_pz')
    	self.mc_truth_tree.var('Pvis_SS_E')
    	self.mc_truth_tree.var('Pvis_SS_ID')
    	self.mc_truth_tree.var('Pvis_OS_p')
    	self.mc_truth_tree.var('Pvis_OS_px')
    	self.mc_truth_tree.var('Pvis_OS_py')
    	self.mc_truth_tree.var('Pvis_OS_pz')
    	self.mc_truth_tree.var('Pvis_OS_E')
    	self.mc_truth_tree.var('Pvis_OS_ID')
    	self.mc_truth_tree.var('Pnu_SS_p')
    	self.mc_truth_tree.var('Pnu_OS_p')
    	self.mc_truth_tree.var('Pnu_px')
    	self.mc_truth_tree.var('Pnu_py')
    	self.mc_truth_tree.var('Pnu_pz')
    	self.mc_truth_tree.var('Pnu_p')
    	self.mc_truth_tree.var('Thr1')
    	self.mc_truth_tree.var('Thr1_px')
    	self.mc_truth_tree.var('Thr1_py')
    	self.mc_truth_tree.var('Thr1_pz')
    	self.mc_truth_tree.var('Thr2')
    	self.mc_truth_tree.var('Thr2_px')
    	self.mc_truth_tree.var('Thr2_py')
    	self.mc_truth_tree.var('Thr2_pz')
    	self.mc_truth_tree.var('Thr3')
    	self.mc_truth_tree.var('Thr3_px')
    	self.mc_truth_tree.var('Thr3_py')
    	self.mc_truth_tree.var('Thr3_pz')
    	self.mc_truth_tree.var('OblThr1')
    	self.mc_truth_tree.var('OblThr2')
    	self.mc_truth_tree.var('OblThr3')
    	self.mc_truth_tree.var('Sph1')
    	self.mc_truth_tree.var('Sph1_px')
    	self.mc_truth_tree.var('Sph1_py')
    	self.mc_truth_tree.var('Sph1_pz')
    	self.mc_truth_tree.var('Sph2')
    	self.mc_truth_tree.var('Sph2_px')
    	self.mc_truth_tree.var('Sph2_py')
    	self.mc_truth_tree.var('Sph2_pz')
    	self.mc_truth_tree.var('Sph3')
    	self.mc_truth_tree.var('Sph3_px')
    	self.mc_truth_tree.var('Sph3_py')
    	self.mc_truth_tree.var('Sph3_pz')
    	self.mc_truth_tree.var('LinSph1')
    	self.mc_truth_tree.var('LinSph1_px')
    	self.mc_truth_tree.var('LinSph1_py')
    	self.mc_truth_tree.var('LinSph1_pz')
    	self.mc_truth_tree.var('LinSph2')
    	self.mc_truth_tree.var('LinSph2_px')
    	self.mc_truth_tree.var('LinSph2_py')
    	self.mc_truth_tree.var('LinSph2_pz')
    	self.mc_truth_tree.var('LinSph3')
    	self.mc_truth_tree.var('LinSph3_px')
    	self.mc_truth_tree.var('LinSph3_py')
    	self.mc_truth_tree.var('LinSph3_pz')
    	self.mc_truth_tree.var('mu_IPdist')
    	self.mc_truth_tree.var('Ds_IPdist')
    	

    	self.smeared_tree = Tree(self.cfg_ana.smeared_tree_name, self.cfg_ana.smeared_tree_title)
        self.smeared_tree.var('n_particles')
        self.smeared_tree.var('event_number')
        self.smeared_tree.var('pv_sx')
        self.smeared_tree.var('pv_sy')
        self.smeared_tree.var('pv_sz')
        self.smeared_tree.var('sv_sx')
        self.smeared_tree.var('sv_sy')
        self.smeared_tree.var('sv_sz')
        self.smeared_tree.var('tv_sx')
        self.smeared_tree.var('tv_sy')
        self.smeared_tree.var('tv_sz')
        self.smeared_tree.var('pvDs_sx')
        self.smeared_tree.var('pvDs_sy')
        self.smeared_tree.var('pvDs_sz')
        self.smeared_tree.var('svDs_sx')
        self.smeared_tree.var('svDs_sy')
        self.smeared_tree.var('svDs_sz')
        self.smeared_tree.var('pvsv_sdistance')
        self.smeared_tree.var('Bs_Ds_sEn')
        self.smeared_tree.var('Bs_Ds_spx')
        self.smeared_tree.var('Bs_Ds_spy')
        self.smeared_tree.var('Bs_Ds_spz')
        self.smeared_tree.var('Bs_Ds_sp')
        self.smeared_tree.var('Bs_Ds_spT')
        self.smeared_tree.var('DsK1_sE')
    	self.smeared_tree.var('DsK1_spx')
    	self.smeared_tree.var('DsK1_spy')
    	self.smeared_tree.var('DsK1_spz')
    	self.smeared_tree.var('DsK1_sp')
    	self.smeared_tree.var('DsK2_sE')
    	self.smeared_tree.var('DsK2_spx')
    	self.smeared_tree.var('DsK2_spy')
    	self.smeared_tree.var('DsK2_spz')
    	self.smeared_tree.var('DsK2_sp')
    	self.smeared_tree.var('Dspi_sE')
    	self.smeared_tree.var('Dspi_spx')
    	self.smeared_tree.var('Dspi_spy')
    	self.smeared_tree.var('Dspi_spz')
    	self.smeared_tree.var('Dspi_sp')
    	self.smeared_tree.var('Dsmu_sE')
    	self.smeared_tree.var('Dsmu_spx')
    	self.smeared_tree.var('Dsmu_spy')
    	self.smeared_tree.var('Dsmu_spz')
    	self.smeared_tree.var('Dsmu_sp')
    	self.smeared_tree.var('Dsmu_spT')
    	self.smeared_tree.var('Ds_KKpi_sm')
    	self.smeared_tree.var('Bs_KKpimu_sm')
    	self.smeared_tree.var('Bs_KKpimu_smcorr')
    	self.smeared_tree.var('Pvis_SS_sp')
        self.smeared_tree.var('Pvis_SS_sE')
        self.smeared_tree.var('Pvis_OS_sp')
        self.smeared_tree.var('Pvis_OS_sE')
        self.smeared_tree.var('mu_sIPdist')
        self.smeared_tree.var('Ds_sIPdist')
        self.smeared_tree.var('Dsmu_sDOCA')
        self.smeared_tree.var('Dsmu_Chi2')
        self.smeared_tree.var('Dsmu_NDF')
        self.smeared_tree.var('Dsmu_FitTest')        
        self.smeared_tree.var('Dsmu_fitvtx_x')
        self.smeared_tree.var('Dsmu_fitvtx_y')
        self.smeared_tree.var('Dsmu_fitvtx_z')
        self.smeared_tree.var('Dsmu_fitvtx_diffx')
        self.smeared_tree.var('Dsmu_fitvtx_diffy')
        self.smeared_tree.var('Dsmu_fitvtx_diffz')
        
        self.smeared_vtx = Tree(self.cfg_ana.smeared_vtx_name, self.cfg_ana.smeared_vtx_title)
        self.smeared_vtx.var('pv_sx')
        self.smeared_vtx.var('pv_sy')
        self.smeared_vtx.var('pv_sz')
        self.smeared_vtx.var('sv_sx')
        self.smeared_vtx.var('sv_sy')
        self.smeared_vtx.var('sv_sz')
        self.smeared_vtx.var('pv_sdiffx')
        self.smeared_vtx.var('pv_sdiffy')
        self.smeared_vtx.var('pv_sdiffz')
        self.smeared_vtx.var('sv_sdiffx')
        self.smeared_vtx.var('sv_sdiffy')
        self.smeared_vtx.var('sv_sdiffz')
        self.smeared_vtx.var('pvsv_sdistance')
        self.smeared_vtx.var('B_KKpimu_sm')
    	self.smeared_vtx.var('B_KKpimu_smcorr')        
    	self.smeared_vtx.var('mu_spx')
    	self.smeared_vtx.var('mu_spy')
    	self.smeared_vtx.var('mu_spz')
    	self.smeared_vtx.var('mu_sp')
    	self.smeared_vtx.var('mu_spT')
    	self.smeared_vtx.var('mu_spT_B')
        self.smeared_vtx.var('mu_sIPdist')
        self.smeared_vtx.var('Ds_Chi2')
        self.smeared_vtx.var('Ds_NDF')
        self.smeared_vtx.var('Ds_CDF')
        self.smeared_vtx.var('Ds_fitvtx_x')
        self.smeared_vtx.var('Ds_fitvtx_y')
        self.smeared_vtx.var('Ds_fitvtx_z')
        self.smeared_vtx.var('Ds_fitvtx_diffx')
        self.smeared_vtx.var('Ds_fitvtx_diffy')
        self.smeared_vtx.var('Ds_fitvtx_diffz')
        self.smeared_vtx.var('B_fitvtx_Chi2')
        self.smeared_vtx.var('B_fitvtx_NDF')
        self.smeared_vtx.var('B_fitvtx_CDF')
        self.smeared_vtx.var('B_fitvtx_x')
        self.smeared_vtx.var('B_fitvtx_y')
        self.smeared_vtx.var('B_fitvtx_z')
        self.smeared_vtx.var('B_fitvtx_diffx')
        self.smeared_vtx.var('B_fitvtx_diffy')
        self.smeared_vtx.var('B_fitvtx_diffz')
        self.smeared_vtx.var('B_Dsmu_fitvtx_Chi2')
        self.smeared_vtx.var('B_Dsmu_fitvtx_NDF')
        self.smeared_vtx.var('B_Dsmu_fitvtx_CDF')
        self.smeared_vtx.var('B_Dsmu_fitvtx_x')
        self.smeared_vtx.var('B_Dsmu_fitvtx_y')
        self.smeared_vtx.var('B_Dsmu_fitvtx_z')
        self.smeared_vtx.var('B_Dsmu_fitvtx_diffx')
        self.smeared_vtx.var('B_Dsmu_fitvtx_diffy')
        self.smeared_vtx.var('B_Dsmu_fitvtx_diffz')
        self.smeared_vtx.var('pv_DsPCA_x')   
        self.smeared_vtx.var('pv_DsPCA_y')
        self.smeared_vtx.var('pv_DsPCA_z')
        self.smeared_vtx.var('pv_muPCA_x')   
        self.smeared_vtx.var('pv_muPCA_y')
        self.smeared_vtx.var('pv_muPCA_z')        
        self.smeared_vtx.var('Dsmu_DCA')
        self.smeared_vtx.var('Thr1')
    	self.smeared_vtx.var('Thr1_px')
    	self.smeared_vtx.var('Thr1_py')
    	self.smeared_vtx.var('Thr1_pz')
    	self.smeared_vtx.var('Thr2')
    	self.smeared_vtx.var('Thr2_px')
    	self.smeared_vtx.var('Thr2_py')
    	self.smeared_vtx.var('Thr2_pz')
    	self.smeared_vtx.var('Thr3')
    	self.smeared_vtx.var('Thr3_px')
    	self.smeared_vtx.var('Thr3_py')
    	self.smeared_vtx.var('Thr3_pz')
        self.smeared_vtx.var('ThrMaj1')
        self.smeared_vtx.var('ThrMaj1_x')
        self.smeared_vtx.var('ThrMaj1_y')
        self.smeared_vtx.var('ThrMaj1_z')
        self.smeared_vtx.var('ThrMaj2')
        self.smeared_vtx.var('ThrMaj2_x')
        self.smeared_vtx.var('ThrMaj2_y')
        self.smeared_vtx.var('ThrMaj2_z')
        self.smeared_vtx.var('ThrMaj3')
        self.smeared_vtx.var('ThrMaj3_x')
        self.smeared_vtx.var('ThrMaj3_y')
        self.smeared_vtx.var('ThrMaj3_z')
        self.smeared_vtx.var('ThrMaj2_SS_spT')
        self.smeared_vtx.var('ThrMaj2_OS_spT')
        self.smeared_vtx.var('Pvis_SS_sp')
        self.smeared_vtx.var('Pvis_SS_sE')
        self.smeared_vtx.var('Pvis_OS_sp')
        self.smeared_vtx.var('Pvis_OS_sE')

        # same for smeared values
        #self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)
        #self.tree.var('n_particles')
        #self.tree.var('event_number')
        #self.tree.var('pv_x')
        #self.tree.var('pv_y')
        #self.tree.var('pv_z')

    def process(self, event):
        Bs_mc_truth = None # B0s particle (MC truth)
        opposite_b_quark_mc_truth = None # b quark opposite to B0s (MC truth)
        b_quarkSS_mc_truth = None
        b_quarkOS_mc_truth = None
        os_b_quark_mc_truth = None # b quark opposite side to B0s (MC truth)
        ss_b_quark_mc_truth = None # b quark same side to B0s (MC truth)

        pv_mctruth = None
        pv = None # primary vertex
        sv_mctruth = None
        sv = None
        tv_mc_truth = None
        tv = None
        
        pv_Ds_mctruth = None
        pv_Ds = None
        sv_Ds_mctruth = None
        sv_Ds = None
        
        pv_mu_mctruth = None
        pv_mu = None
        
        Ds_B = None
        DsK1 = None
        DsK2 = None
        Dspi = None
        Dsmu = None
        Ds_mctruth = None
        Ds1_mc_truth = None
        Ds2_mc_truth = None

        K_mc_truth = None

        Dst_mc_truth = None
        D0st_mc_truth = None
        pi_mc_truth = None
        pi0_mc_truth = None
        mu_mc_truth = None
        mu1_mc_truth = None
        munu_mc_truth = None
        D_B_mc_truth = None
        mu_B_mc_truth = None
        munu_B_mc_truth = None
        D_B = None
        mu_B = None
        


        pb = 0. # B momentum
        pvsv_distance = 0.0
        pvsv_sdistance = 0.0
        
        ##b quark kinematic variables
        bquark_px = 0.
        bquark_py = 0.
        bquark_pz = 0.
        bquark_npx = 0.
        bquark_npy = 0.
        bquark_npz = 0.
        bquark_p = 0.
        bquark_E = 0.
        bquarkSS_px = 0.
        bquarkSS_py = 0.
        bquarkSS_pz = 0.
        bquarkSS_npx = 0.
        bquarkSS_npy = 0.
        bquarkSS_npz = 0.
        bquarkSS_p = 0.
        bquarkSS_E = 0.
        bquarkOS_px = 0.
        bquarkOS_py = 0.
        bquarkOS_pz = 0.
        bquarkOS_npx = 0.
        bquarkOS_npy = 0.
        bquarkOS_npz = 0.
        bquarkOS_p = 0.
        bquarkOS_E = 0.
        #========================#
        
        diffx = 0.0
        diffy = 0.0
        sdiffx = 0.0
        sdiffy = 0.0
        mu_IPdist = 0.0
        mu_sIPdist = 0.0
        
        Ds_En = 0. #Ds energy
        Ds_px = 0.
        Ds_py = 0.
        Ds_pz = 0.
        Ds_p = 0.
        Ds_pT = 0.
        
        Ds1_En = 0. #D*+ energy
        Ds1_px = 0.
        Ds1_py = 0.
        Ds1_pz = 0.
        Ds1_p = 0.
        Ds2_En = 0. #D*+ energy
        Ds2_px = 0.
        Ds2_py = 0.
        Ds2_pz = 0.
        Ds2_p = 0.
        
        Ds1_ptot = 0.0
        Ds2_ptot = 0.0
        Ds1_m = 0.0
        Ds2_m = 0.0
        Ds1Ds2_ptot = 0.0
        BsDs1Ds2_m = 0.0

        DsK1_En = 0. #Kaon energy
        DsK1_px = 0.
        DsK1_py = 0.
        DsK1_pz = 0.
        DsK1_p = 0.

        DsK2_En = 0. #Kaon energy
        DsK2_px = 0.
        DsK2_py = 0.
        DsK2_pz = 0.
        DsK2_p = 0.

        Dspi_En = 0. #pion energy
        Dspi_px = 0.
        Dspi_py = 0.
        Dspi_pz = 0.
        Dspi_p = 0.

        Dsmu_En = 0. #muon energy
        Dsmu_px = 0.
        Dsmu_py = 0.
        Dsmu_pz = 0.
        Dsmu_p = 0.
        Dsmu_pT = 0.
        mu_IPdist = 0.0
        diffx = 0.
        diffy = 0.
        
        Pvis_SS_p = 0.0
        Pvis_SS_ptot = 0.0
        Pvis_SS_px = 0.0
        Pvis_SS_py = 0.0
        Pvis_SS_pz = 0.0
        Pvis_SS_E = 0.0        
        Pvis_SS_sp = 0.0
        Pvis_SS_sE = 0.0
        Pvis_SS_pT = 0.0
        Pvis_OS_p = 0.0
        Pvis_OS_ptot = 0.0
        Pvis_OS_px = 0.0
        Pvis_OS_py = 0.0
        Pvis_OS_pz = 0.0
        Pvis_OS_E = 0.0        
        Pvis_OS_sp = 0.0
        Pvis_OS_sE = 0.0
        Pvis_OS_pT = 0.0
        Pnu_SS_p = 0.
        Pnu_OS_p = 0.
        Pnu_px = 0.
        Pnu_py = 0.
        Pnu_pz = 0.
        Pnu_p = 0.
        
        ThrMaj1 = 0.
        ThrMaj1_x = 0.
        ThrMaj1_y = 0.
        ThrMaj1_z = 0.
        ThrMaj2 = 0.
        ThrMaj2_x = 0.
        ThrMaj2_y = 0.
        ThrMaj2_z = 0.
        ThrMaj3 = 0.
        ThrMaj3_x = 0.
        ThrMaj3_y = 0.
        ThrMaj3_z = 0.
        ThrMaj2_SS_spT = 0.
        ThrMaj2_OS_spT = 0.
        Pvis_SS_spx = 0.0
        Pvis_SS_spy = 0.0
        Pvis_SS_spz = 0.0
        Pvis_OS_spx = 0.0
        Pvis_OS_spy = 0.0
        Pvis_OS_spz = 0.0

        Ds_KKpi_ptot = 0.0
        Ds_KKpi_m = 0.0

        Bs_KKpimu_ptot = 0.0
        Bs_KKpimu_m = 0.0

        KKpimu_par = 0.0
        KKpimu_per = 0.0
        Bs_KKpimu_mcorr = 0.0

        Bs_gammapi_En = 0.0
        Bs_gamma_En = 0.0
        Bs_pi_En = 0.0
        Bs_gammapi_px = 0.0
        Bs_gammapi_py = 0.0
        Bs_gammapi_pz = 0.0
        Bs_gammapi_p = 0.0

        #resolutions
        Ds_resol = 0.0
        DsK1_resol = 0.0
        DsK2_resol = 0.0
        Dspi_resol = 0.0
        Dsmu_resol = 0.0
        Pvis_SS_resol = 0.0
        Pvis_OS_resol = 0.0

        DsK1_pt = 0.0
        DsK2_pt = 0.0
        Dspi_pt = 0.0
        Dsmu_pT = 0.0

        DsX_En = 0. #pion energy
        DsX_px = 0.
        DsX_py = 0.
        DsX_pz = 0.
        DsX_p = 0.

        #==smeared variables==
        Ds_sEn = 0. #Kaon energy
        Ds_spx = 0.
        Ds_spy = 0.
        Ds_spz = 0.
        Ds_sp = 0.
        Ds_spT = 0.
        
        DsK1_sEn = 0. #Kaon energy
        DsK1_spx = 0.
        DsK1_spy = 0.
        DsK1_spz = 0.
        DsK1_sp = 0.

        DsK2_sEn = 0. #Kaon energy
        DsK2_spx = 0.
        DsK2_spy = 0.
        DsK2_spz = 0.
        DsK2_sp = 0.

        Dspi_sEn = 0. #pion energy
        Dspi_spx = 0.
        Dspi_spy = 0.
        Dspi_spz = 0.
        Dspi_sp = 0.

        Dsmu_sEn = 0. #muon energy
        Dsmu_spx = 0.
        Dsmu_spy = 0.
        Dsmu_spz = 0.
        Dsmu_sp = 0.
        Dsmu_spT = 0.
        
        Ds_IPdist = 0.0
        Ds_sIPdist = 0.0
        
        Ds_diffx = 0.0
        Ds_diffy = 0.0
        Ds_diffz = 0.0
        Ds_sdiffx = 0.0
        Ds_sdiffy = 0.0
        Ds_sdiffz = 0.0
        
        mu_IPdist = 0.
        mu_sIPdist = 0.
        diffx = 0.
        diffy = 0.
        sdiffx = 0.
        sdiffy = 0.
        
        Ds_spxdir = 0.0
        Ds_spydir = 0.0
        Ds_spzdir = 0.0
        
        mu_spxdir = 0.0
        mu_spydir = 0.0
        mu_spzdir = 0.0  

        Dsst1_gp_En = 0.0
        Dsst1_gp_px = 0.0
        Dsst1_gp_py = 0.0
        Dsst1_gp_pz = 0.0
        Dsst1_gp_p = 0.0

        Dsst2_gp_En = 0.0
        Dsst2_gp_px = 0.0
        Dsst2_gp_py = 0.0
        Dsst2_gp_pz = 0.0
        Dsst2_gp_p = 0.0

        Ds_KKpi_sptot = 0.0
        Ds_KKpi_sm = 0.0

        KKpimu_spar = 0.0
        KKpimu_sper = 0.0
        Bs_KKpimu_smcorr = 0.0
        Bs_KKpimu_sptot = 0.0
        
        fitvtx = 0.0
        
        mu_spT_FdB = 0.0
        mu_sppar_B = 0.0

        #====================

        BD_vdistance = 0.0
        Bdec_vdistance = 0.0
        E_total = 0.0
        p_total = 0.0
        B_prod = 0.0

        store = event.input # This is just a shortcut
        event_info = store.get("EventInfo")
        particles_info = store.get("GenParticle")
        vertices_info = store.get("GenVertex")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        plist_bquark_mc_truth = list([])
        plist_bquarkSS_mc_truth = list([])
        plist_bquarkOS_mc_truth = list([])
        
        plist_D_mc_truth = list([])
        plist_Dsstp_mc_truth = list([])
        plist_Dsstm_mc_truth = list([])
        plist_Ds_mctruth = list([])
        plist_DsBs_mctruth = list([])
        plist_Ds1_mc_truth = list([])
        plist_Ds2_mc_truth = list([])

        plist_DsK_mc_truth = list([])
        plist_Dspi_mc_truth = list([])
        plist_Dsmu_mc_truth = list([])
        plist_gamma_mc_truth = list([])



        plist_Bdaughters = list([])
        plist_B0s = list([])

        plist = [11, 14, 22, 111, 113, 130, 310, 211, 213, 223, 321]
        #plist2 = [221, 311, 331, 333, 9000311]
        plist2 = [221, 333]

        for ptc_gen1 in ptcs:
            if abs(ptc_gen1.pdgid) == 531 and (ptc_gen1.start_vertex != ptc_gen1.end_vertex): # if B found and it's not an oscillation

                self.counter += 1
                if self.counter %100 == 0:
                    print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
                    self.last_timestamp = time.time()

                Bs_mc_truth = ptc_gen1
                pb = Bs_mc_truth.p.absvalue()

                plist_B0s.append(ptc_gen1)

                #print len(plist_B0s)
                
                Dsmu_vtx = FastFit.FastFit(2, 0)
                Ds_vtx = FastFit.FastFit(3, 0)
                Bs_Dsmu_vtx = FastFit.FastFit(2, 0)
                Bs_vtx = FastFit.FastFit(4, 0)

                if pb > 0.0: # select only events with large momentum of the B
                    self.pb_counter += 1

                    # looking for opposite b quark. This is a dirty hack. Works only because both PYTHIA/HepMC and PODIO store particles ordered. But IT'S NOT GUARANTEED
		            # need to find better algorithm to look for the opposite b-quark
                    index = 0
                    while os_b_quark_mc_truth == None and index < len(ptcs):
                        if (abs(ptcs[index].pdgid) == 5 and ptcs[index].status == 23) and np.dot([Bs_mc_truth.p.px, Bs_mc_truth.p.py, Bs_mc_truth.p.pz], [ptcs[index].p.px, ptcs[index].p.py, ptcs[index].p.pz]) < 0:
                            os_b_quark_mc_truth = ptcs[index]
                            #print os_b_quark_mc_truth.start_vertex
                        index += 1
                    #print opposite_b_quark_mc_truth.pdgid

                    #print opposite_b_quark_mc_truth.start_vertex
                    pv_mctruth = Bs_mc_truth.start_vertex
                    pv = copy.deepcopy(pv_mctruth)

                    sv_mctruth = Bs_mc_truth.end_vertex
                    sv = copy.deepcopy(sv_mctruth)

                    pvsv_distance = np.sqrt((sv_mctruth.x - pv_mctruth.x)**2 + (sv_mctruth.y - pv_mctruth.y)**2 + (sv_mctruth.z - pv_mctruth.z)**2)

                    #print len(plist_B0s), ' ', pvsv_distance
                    #plist_B_mc_truth = list([])

                    #for ptc_gen4 in ptcs:
                    #print ptc_gen4.pdgid



                    if pvsv_distance > 0.0:
                        
                        Thr1 = event_info.at(0).Thrust().Thr1
                        Thr2 = event_info.at(0).Thrust().Thr2
                        Thr3 = event_info.at(0).Thrust().Thr3
                        ThrMaj1 = event_info.at(0).Thrust().ThrMaj1
                        ThrMaj2 = event_info.at(0).Thrust().ThrMaj2
                        ThrMaj3 = event_info.at(0).Thrust().ThrMaj3
                        ThrMaj1_x = event_info.at(0).Thrust().ThrMaj1_axis.X
                        ThrMaj1_y = event_info.at(0).Thrust().ThrMaj1_axis.Y
                        ThrMaj1_z = event_info.at(0).Thrust().ThrMaj1_axis.Z
                        ThrMaj2_x = event_info.at(0).Thrust().ThrMaj2_axis.X
                        ThrMaj2_y = event_info.at(0).Thrust().ThrMaj2_axis.Y
                        ThrMaj2_z = event_info.at(0).Thrust().ThrMaj2_axis.Z
                        ThrMaj3_x = event_info.at(0).Thrust().ThrMaj3_axis.X
                        ThrMaj3_y = event_info.at(0).Thrust().ThrMaj3_axis.Y
                        ThrMaj3_z = event_info.at(0).Thrust().ThrMaj3_axis.Z
                        OblThr1 = event_info.at(0).Thrust().OblThr1
                        OblThr2 = event_info.at(0).Thrust().OblThr2
                        OblThr3 = event_info.at(0).Thrust().OblThr3
                        Thr1_px = event_info.at(0).Thrust().Thr1_axis.X
                        Thr1_py = event_info.at(0).Thrust().Thr1_axis.Y
                        Thr1_pz = event_info.at(0).Thrust().Thr1_axis.Z
                        Thr2_px = event_info.at(0).Thrust().Thr2_axis.X
                        Thr2_py = event_info.at(0).Thrust().Thr2_axis.Y
                        Thr2_pz = event_info.at(0).Thrust().Thr2_axis.Z
                        Thr3_px = event_info.at(0).Thrust().Thr3_axis.X
                        Thr3_py = event_info.at(0).Thrust().Thr3_axis.Y
                        Thr3_pz = event_info.at(0).Thrust().Thr3_axis.Z
                        Sph1 = event_info.at(0).Sphericity().Sph1
                        Sph2 = event_info.at(0).Sphericity().Sph2
                        Sph3 = event_info.at(0).Sphericity().Sph3
                        Sph1_px = event_info.at(0).Sphericity().Sph1_axis.X
                        Sph1_py = event_info.at(0).Sphericity().Sph1_axis.Y
                        Sph1_pz = event_info.at(0).Sphericity().Sph1_axis.Z
                        Sph2_px = event_info.at(0).Sphericity().Sph2_axis.X
                        Sph2_py = event_info.at(0).Sphericity().Sph2_axis.Y
                        Sph2_pz = event_info.at(0).Sphericity().Sph2_axis.Z
                        Sph3_px = event_info.at(0).Sphericity().Sph3_axis.X
                        Sph3_py = event_info.at(0).Sphericity().Sph3_axis.Y
                        Sph3_pz = event_info.at(0).Sphericity().Sph3_axis.Z
                        LinSph1 = event_info.at(0).Sphericity().LinSph1
                        LinSph2 = event_info.at(0).Sphericity().LinSph2
                        LinSph3 = event_info.at(0).Sphericity().LinSph3
                        LinSph1_px = event_info.at(0).Sphericity().LinSph1_axis.X
                        LinSph1_py = event_info.at(0).Sphericity().LinSph1_axis.Y
                        LinSph1_pz = event_info.at(0).Sphericity().LinSph1_axis.Z
                        LinSph2_px = event_info.at(0).Sphericity().LinSph2_axis.X
                        LinSph2_py = event_info.at(0).Sphericity().LinSph2_axis.Y
                        LinSph2_pz = event_info.at(0).Sphericity().LinSph2_axis.Z
                        LinSph3_px = event_info.at(0).Sphericity().LinSph3_axis.X
                        LinSph3_py = event_info.at(0).Sphericity().LinSph3_axis.Y
                        LinSph3_pz = event_info.at(0).Sphericity().LinSph3_axis.Z

                        for ptc_gen2 in ptcs:
                            
                            if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23):
                                
                                plist_bquark_mc_truth.append(ptc_gen2)
                                b_quark_mc_truth = ptc_gen2                                
                                bquark_px = b_quark_mc_truth.p.px
                                bquark_py = b_quark_mc_truth.p.py
                                bquark_pz = b_quark_mc_truth.p.pz
                                bquark_p = b_quark_mc_truth.p.absvalue()
                                bquark_npx = b_quark_mc_truth.p.px/b_quark_mc_truth.p.absvalue()
                                bquark_npy = b_quark_mc_truth.p.py/b_quark_mc_truth.p.absvalue()
                                bquark_npz = b_quark_mc_truth.p.pz/b_quark_mc_truth.p.absvalue()
                                bquark_E = b_quark_mc_truth.energy
                            
                            if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                                
                                plist_bquarkSS_mc_truth.append(ptc_gen2)
                                b_quarkSS_mc_truth = ptc_gen2                                
                                bquarkSS_px = b_quarkSS_mc_truth.p.px
                                bquarkSS_py = b_quarkSS_mc_truth.p.py
                                bquarkSS_pz = b_quarkSS_mc_truth.p.pz
                                bquarkSS_p = b_quarkSS_mc_truth.p.absvalue()
                                bquarkSS_npx = b_quarkSS_mc_truth.p.px/b_quarkSS_mc_truth.p.absvalue()
                                bquarkSS_npy = b_quarkSS_mc_truth.p.py/b_quarkSS_mc_truth.p.absvalue()
                                bquarkSS_npz = b_quarkSS_mc_truth.p.pz/b_quarkSS_mc_truth.p.absvalue()
                                bquarkSS_E = b_quarkSS_mc_truth.energy
                                
                            #if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) > 0):
                            if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) < 0):
                            
                                plist_bquarkOS_mc_truth.append(ptc_gen2)
                                b_quarkOS_mc_truth = ptc_gen2                                
                                bquarkOS_px = b_quarkOS_mc_truth.p.px
                                bquarkOS_py = b_quarkOS_mc_truth.p.py
                                bquarkOS_pz = b_quarkOS_mc_truth.p.pz
                                bquarkOS_p = b_quarkOS_mc_truth.p.absvalue()
                                bquarkOS_npx = b_quarkOS_mc_truth.p.px/b_quarkOS_mc_truth.p.absvalue()
                                bquarkOS_npy = b_quarkOS_mc_truth.p.py/b_quarkOS_mc_truth.p.absvalue()
                                bquarkOS_npz = b_quarkOS_mc_truth.p.pz/b_quarkOS_mc_truth.p.absvalue()
                                bquarkOS_E = b_quarkOS_mc_truth.energy
                                #print np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz])
                                
                            #print 'os: ', os_b_quark_mc_truth.pdgid, os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py, os_b_quark_mc_truth.p.pz
                            #print 'ss: ', ss_b_quark_mc_truth.pdgid, ss_b_quark_mc_truth.p.px, ss_b_quark_mc_truth.p.py, ss_b_quark_mc_truth.p.pz
                            
                            #if np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0:
                            #    if abs(ptc_gen2.pdgid) == 531:
                            #        print ptc_gen2.pdgid
                            
                            #if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) not in [12,14,16]) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0):
                            
                            #if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) in [12,14,16]) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                            if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) == 14) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                                
                                Pnu_px = ptc_gen2.p.px 
                                Pnu_py = ptc_gen2.p.py 
                                Pnu_pz = ptc_gen2.p.pz
                                Pnu_p  = ptc_gen2.p.absvalue()
                            
                            if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) not in [12,14,16]) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                                
                                #if abs(ptc_gen2.pdgid) == 13:
                                #print 'SS', ptc_gen2.pdgid, 
                                Pvis_SS_ID = ptc_gen2.pdgid 
                                Pvis_SS_mc_truth = ptc_gen2
                                Pvis_SS = copy.deepcopy(Pvis_SS_mc_truth)
                                Pvis_SS_pT = np.sqrt(ptc_gen2.p.px*ptc_gen2.p.px + ptc_gen2.p.py*ptc_gen2.p.py)
                                Pvis_SS_resol = momentum_res(Pvis_SS_pT)
                                
                                Pvis_SS_px += ptc_gen2.p.px
                                Pvis_SS_py += ptc_gen2.p.py
                                Pvis_SS_pz += ptc_gen2.p.pz
                                
                                Pvis_SS_p += ptc_gen2.p.absvalue()
                                Pvis_SS_E += ptc_gen2.energy
                                
                                if self.cfg_ana.smear_momentum:
                                    Pvis_SS.p = smear_momentum(Pvis_SS.p, Pvis_SS_resol, Pvis_SS_resol, Pvis_SS_resol)
                                
                                Pvis_SS_sp += Pvis_SS.p.absvalue()
                                Pvis_SS_sE += np.sqrt(Pvis_SS.p.absvalue()*Pvis_SS.p.absvalue() + Pvis_SS.mass*Pvis_SS.mass)
                            
                            #if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) not in [12,14,16]) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) > 0):
                            if (ptc_gen2.status == 1 and abs(ptc_gen2.pdgid) not in [12,14,16]) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) < 0):
                                
                                #if abs(ptc_gen2.pdgid) == 13:
                                #    print 'OS', ptc_gen2.pdgid
                                Pvis_OS_ID = ptc_gen2.pdgid 
                                Pvis_OS_mc_truth = ptc_gen2
                                Pvis_OS = copy.deepcopy(Pvis_OS_mc_truth)
                                Pvis_OS_pT = np.sqrt(ptc_gen2.p.px*ptc_gen2.p.px + ptc_gen2.p.py*ptc_gen2.p.py)
                                Pvis_OS_resol = momentum_res(Pvis_OS_pT)
                                
                                Pvis_OS_px += ptc_gen2.p.px
                                Pvis_OS_py += ptc_gen2.p.py
                                Pvis_OS_pz += ptc_gen2.p.pz
                                
                                Pvis_OS_p += ptc_gen2.p.absvalue()
                                Pvis_OS_E += ptc_gen2.energy
                                
                                if self.cfg_ana.smear_momentum:
                                    Pvis_OS.p = smear_momentum(Pvis_OS.p, Pvis_OS_resol, Pvis_OS_resol, Pvis_OS_resol)
                                
                                Pvis_OS_sp += Pvis_OS.p.absvalue()
                                Pvis_OS_sE += np.sqrt(Pvis_OS.p.absvalue()*Pvis_OS.p.absvalue() + Pvis_OS.mass*Pvis_OS.mass)
                                

                            #if (ptc_gen2.start_vertex == B_mc_truth.end_vertex) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0):
                            
                            Pnu_SS_p = np.sqrt(np.dot([Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz],[Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz]))
                            
                            Pnu_OS_p = np.sqrt(np.dot([Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz],[Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz]))
                            
                            
                            
                            if (ptc_gen2.start_vertex == Bs_mc_truth.end_vertex) and (ptc_gen2.start_vertex != ptc_gen2.end_vertex):


                                BD_vdistance = np.sqrt((Bs_mc_truth.end_vertex.x - ptc_gen2.start_vertex.x)**2 + (Bs_mc_truth.end_vertex.y - ptc_gen2.start_vertex.y)**2 + (Bs_mc_truth.end_vertex.z - ptc_gen2.start_vertex.z)**2)
        
                                if abs(ptc_gen2.pdgid) == 22 or abs(ptc_gen2.pdgid) == 111:
                                    #print 'B0s daugthers:', ptc_gen2.pdgid
                                    plist_gamma_mc_truth.append(ptc_gen2)
                                    self.gamma_counter += 1
        
                                plist_Bdaughters.append(ptc_gen2)
                                
        
                                #if (abs(ptc_gen2.pdgid) == 22 or abs(ptc_gen2.pdgid) == 111):
                                #if abs(ptc_gen2.pdgid) == 111 and BD_vdistance < 0.25:
                                #    Bs_gammapi_px = Bs_gammapi_px + ptc_gen2.p.px
                                #    Bs_gammapi_py =  Bs_gammapi_py + ptc_gen2.p.py
                                #    Bs_gammapi_pz = Bs_gammapi_pz + ptc_gen2.p.pz
                                #    Bs_gammapi_p = Bs_gammapi_p + ptc_gen2.p.absvalue()
                                #
                                #    Bs_gammapi_En = Bs_gammapi_En + np.sqrt(ptc_gen2.p.absvalue()*ptc_gen2.p.absvalue() + ptc_gen2.mass*ptc_gen2.mass)

                                if abs(ptc_gen2.pdgid) == 431 and BD_vdistance < 0.25:                                    
                                    plist_Ds_mctruth.append(ptc_gen2)
                                    
                        
                        if len(plist_Ds_mctruth) == 2:
                            self.Ds_counter += 1
                        #print Bs_gammapi_px, Bs_gammapi_py, Bs_gammapi_pz, Bs_gammapi_En
        
                        for ptc_gen3 in ptcs:

                            if (ptc_gen3.start_vertex == plist_Ds_mctruth[0].end_vertex):
                                plist_Ds1_mc_truth.append(ptc_gen3)
                                #print 'Ds1 daugthers: ', ptc_gen3.pdgid, ptc_gen3.start_vertex
                            if (ptc_gen3.start_vertex == plist_Ds_mctruth[1].end_vertex):
                                plist_Ds2_mc_truth.append(ptc_gen3)
                                #print 'Ds2 daugthers: ', ptc_gen3.pdgid, ptc_gen3.start_vertex

                            if (ptc_gen3.start_vertex == plist_Ds_mctruth[0].end_vertex) or (ptc_gen3.start_vertex == plist_Ds_mctruth[1].end_vertex):

                                #print 'Ds daugthers: ', ptc_gen3.pdgid

                                if abs(ptc_gen3.pdgid) == 13:
                                    plist_Dsmu_mc_truth.append(ptc_gen3)
                                    
                                    pv_mu_mctruth = ptc_gen3.start_vertex
                                    pv_mu = copy.deepcopy(pv_mu_mctruth)
                                  
                                    diffx = sv_mctruth.x - pv_mu_mctruth.x
                                    diffy = sv_mctruth.y - pv_mu_mctruth.y                                    
                                    
                                    mu_IPdist = abs(ptc_gen3.p.px*diffy - ptc_gen3.p.py*diffx)/np.sqrt(ptc_gen3.p.px**2 + ptc_gen3.p.py**2)
                                    
                                    self.mu_counter += 1
                        
                        #print 'No. of Ds1 daugthers: ', len( plist_Ds1_mc_truth)
                        #print 'No. of Ds2 daugthers: ', len( plist_Ds2_mc_truth)
                        
                        for ptc_gen4 in ptcs:
                            
                            if (ptc_gen4.start_vertex == Bs_mc_truth.end_vertex):
                                if abs(ptc_gen4.pdgid) == 431 and ptc_gen4.end_vertex != pv_mu_mctruth:                                    
                                    Ds_mctruth = ptc_gen4
                                    plist_DsBs_mctruth.append(ptc_gen4)
                                    pv_Ds_mctruth = Ds_mctruth.start_vertex
                                    pv_Ds = copy.deepcopy(pv_Ds_mctruth)
                                    sv_Ds_mctruth = Ds_mctruth.end_vertex
                                    sv_Ds = copy.deepcopy(sv_Ds_mctruth)
                                    
                                    Ds_diffx = sv_mctruth.x - sv_Ds_mctruth.x
                                    Ds_diffy = sv_mctruth.y - sv_Ds_mctruth.y
                                    Ds_diffz = sv_mctruth.z - sv_Ds_mctruth.z
                                                                        
                                    Ds_IPdist = abs(ptc_gen4.p.px*Ds_diffy - ptc_gen4.p.py*Ds_diffx)/np.sqrt(ptc_gen4.p.px**2 + ptc_gen4.p.py**2)       

                            if (ptc_gen4.start_vertex == plist_Ds_mctruth[0].end_vertex) or (ptc_gen4.start_vertex == plist_Ds_mctruth[1].end_vertex):

                                if abs(ptc_gen4.pdgid) == 321 and (ptc_gen4.start_vertex != plist_Dsmu_mc_truth[0].start_vertex):
                                    plist_DsK_mc_truth.append(ptc_gen4)

                                if abs(ptc_gen4.pdgid) == 211 and (ptc_gen4.start_vertex != plist_Dsmu_mc_truth[0].start_vertex):
                                    plist_Dspi_mc_truth.append(ptc_gen4)

                                if ptc_gen4.start_vertex == plist_Dsmu_mc_truth[0].start_vertex and (abs(ptc_gen4.pdgid) in plist2):
                                    DsX_px = DsX_px + ptc_gen4.p.px
                                    DsX_py =  DsX_py + ptc_gen4.p.py
                                    DsX_pz = DsX_pz + ptc_gen4.p.pz
                                    DsX_p = DsX_p + ptc_gen4.p.absvalue()

                                    DsX_En = DsX_En + np.sqrt(ptc_gen4.p.absvalue()*ptc_gen4.p.absvalue() + ptc_gen4.mass*ptc_gen4.mass)
                                    #print 'B0s daugthers:', ptc_gen4.pdgid

                            if (ptc_gen4.start_vertex == plist_Ds_mctruth[0].start_vertex):
                                if abs(ptc_gen4.pdgid) == 22 or abs(ptc_gen4.pdgid) == 111:
                                    Dsst1_gp_px = Dsst1_gp_px + ptc_gen4.p.px
                                    Dsst1_gp_py = Dsst1_gp_py + ptc_gen4.p.py
                                    Dsst1_gp_pz = Dsst1_gp_pz + ptc_gen4.p.pz
                                    Dsst1_gp_p = Dsst1_gp_p + ptc_gen4.p.absvalue()
                                    Dsst1_gp_En = Dsst1_gp_En + np.sqrt(ptc_gen4.p.absvalue()*ptc_gen4.p.absvalue() + ptc_gen4.mass*ptc_gen4.mass)
                                #print 'D*s1 daugthers:', ptc_gen4.pdgid


                            if (ptc_gen4.start_vertex == plist_Ds_mctruth[1].start_vertex):
                                if abs(ptc_gen4.pdgid) == 22 or abs(ptc_gen4.pdgid) == 111:
                                    Dsst2_gp_px = Dsst2_gp_px + ptc_gen4.p.px
                                    Dsst2_gp_py = Dsst2_gp_py + ptc_gen4.p.py
                                    Dsst2_gp_pz = Dsst2_gp_pz + ptc_gen4.p.pz
                                    Dsst2_gp_p = Dsst2_gp_p + ptc_gen4.p.absvalue()
                                    Dsst2_gp_En = Dsst2_gp_En + np.sqrt(ptc_gen4.p.absvalue()*ptc_gen4.p.absvalue() + ptc_gen4.mass*ptc_gen4.mass)
                                #print 'D*s2 daugthers:', ptc_gen4.pdgid
        
                        #print len(plist_DsK_mc_truth), len(plist_Dspi_mc_truth), len(plist_Dsmu_mc_truth)
        
                        Ds1_En = np.sqrt(plist_Ds_mctruth[0].p.absvalue()*plist_Ds_mctruth[0].p.absvalue() + plist_Ds_mctruth[0].mass*plist_Ds_mctruth[0].mass)
                        Ds1_px, Ds1_py, Ds1_pz = plist_Ds_mctruth[0].p.px, plist_Ds_mctruth[0].p.py, plist_Ds_mctruth[0].p.pz
                        Ds1_p = plist_Ds_mctruth[0].p.absvalue()

                        Ds1_ptot = np.dot([Ds1_px, Ds1_py, Ds1_pz],[Ds1_px, Ds1_py, Ds1_pz])

                        Ds1_m = np.sqrt((Ds1_En*Ds1_En) -  Ds1_ptot)

                        Ds2_En = np.sqrt(plist_Ds_mctruth[1].p.absvalue()*plist_Ds_mctruth[1].p.absvalue() + plist_Ds_mctruth[1].mass*plist_Ds_mctruth[1].mass)
                        Ds2_px, Ds2_py, Ds2_pz = plist_Ds_mctruth[1].p.px, plist_Ds_mctruth[1].p.py, plist_Ds_mctruth[1].p.pz
                        Ds2_p = plist_Ds_mctruth[1].p.absvalue()

                        Ds2_ptot = np.dot([Ds2_px, Ds2_py, Ds2_pz],[Ds2_px, Ds2_py, Ds2_pz])

                        Ds2_m = np.sqrt((Ds2_En*Ds2_En) -  Ds2_ptot)

                        #Ds1Ds2_ptot = np.dot([Ds1_px + Ds2_px + Bs_gammapi_px, Ds1_py + Ds2_py + Bs_gammapi_py, Ds1_pz + Ds2_pz + Bs_gammapi_pz],[Ds1_px + Ds2_px + Bs_gammapi_px, Ds1_py + Ds2_py + Bs_gammapi_py, Ds1_pz + Ds2_pz + Bs_gammapi_pz])

                        #BsDs1Ds2_m = np.sqrt((Ds1_En + Ds2_En + Bs_gammapi_En)*(Ds1_En + Ds2_En + Bs_gammapi_En) -  Ds1Ds2_ptot)

                        #Ds1Ds2_ptot = np.dot([Ds1_px + Ds2_px + Dsst1_gp_px, Ds1_py + Ds2_py + Dsst1_gp_py, Ds1_pz + Ds2_pz + Dsst1_gp_pz],[Ds1_px + Ds2_px + Dsst1_gp_px, Ds1_py + Ds2_py + Dsst1_gp_py, Ds1_pz + Ds2_pz + Dsst1_gp_pz])

                        #BsDs1Ds2_m = np.sqrt((Ds1_En + Ds2_En + Dsst1_gp_En)*(Ds1_En + Ds2_En + Dsst2_gp_En) -  Ds1Ds2_ptot)

                        Ds1Ds2_ptot = np.dot([Ds1_px + Ds2_px, Ds1_py + Ds2_py, Ds1_pz + Ds2_pz],[Ds1_px + Ds2_px, Ds1_py + Ds2_py, Ds1_pz + Ds2_pz])

                        BsDs1Ds2_m = np.sqrt((Ds1_En + Ds2_En)*(Ds1_En + Ds2_En) -  Ds1Ds2_ptot)

                        #print BsDs1Ds2_m
        
                        if len(plist_Ds_mctruth) == 2:
                        #if len(plist_DsK_mc_truth) == 2 and len(plist_Dspi_mc_truth) == 1 and len(plist_Dsmu_mc_truth) == 1:
                            #self.Ds_counter += 1

                            #print plist_DsK_mc_truth[0].pdgid, plist_DsK_mc_truth[1].pdgid, plist_Dspi_mc_truth[0].pdgid, plist_Dsmu_mc_truth[0].pdgid
                            DsK1_mc_truth, DsK2_mc_truth, Dspi_mc_truth = plist_DsK_mc_truth[0], plist_DsK_mc_truth[1], plist_Dspi_mc_truth[0]
                            Ds_K1, Ds_K2, Ds_pi = copy.deepcopy(DsK1_mc_truth), copy.deepcopy(DsK2_mc_truth), copy.deepcopy(Dspi_mc_truth)
                                                                
                            tv_K1_mctruth = DsK1_mc_truth.start_vertex
                            tv_K2_mctruth = DsK2_mc_truth.start_vertex
                            tv_pi_mctruth = Dspi_mc_truth.start_vertex
                            
                            tv_K1 = Ds_K1.start_vertex
                            tv_K2 = Ds_K2.start_vertex
                            tv_pi = Ds_pi.start_vertex
                            
                            Ds_En = np.sqrt(Ds_mctruth.p.absvalue()*Ds_mctruth.p.absvalue() + Ds_mctruth.mass*Ds_mctruth.mass)
                            Ds_px, Ds_py, Ds_pz = Ds_mctruth.p.px, Ds_mctruth.p.py, Ds_mctruth.p.pz
                            Ds_p = Ds_mctruth.p.absvalue()
                            Ds_pT = np.sqrt(Ds_px*Ds_px + Ds_py*Ds_py)

                            DsK1_En = np.sqrt(plist_DsK_mc_truth[0].p.absvalue()*plist_DsK_mc_truth[0].p.absvalue() + plist_DsK_mc_truth[0].mass*plist_DsK_mc_truth[0].mass)

                            DsK1_px, DsK1_py, DsK1_pz = plist_DsK_mc_truth[0].p.px, plist_DsK_mc_truth[0].p.py, plist_DsK_mc_truth[0].p.pz
                            DsK1_p = plist_DsK_mc_truth[0].p.absvalue()

                            DsK2_En = np.sqrt(plist_DsK_mc_truth[1].p.absvalue()*plist_DsK_mc_truth[1].p.absvalue() + plist_DsK_mc_truth[1].mass*plist_DsK_mc_truth[1].mass)

                            DsK2_px, DsK2_py, DsK2_pz = plist_DsK_mc_truth[1].p.px, plist_DsK_mc_truth[1].p.py, plist_DsK_mc_truth[1].p.pz
                            DsK2_p = plist_DsK_mc_truth[1].p.absvalue()

                            Dspi_En = np.sqrt(plist_Dspi_mc_truth[0].p.absvalue()*plist_Dspi_mc_truth[0].p.absvalue() + plist_Dspi_mc_truth[0].mass*plist_Dspi_mc_truth[0].mass)

                            Dspi_px, Dspi_py, Dspi_pz = plist_Dspi_mc_truth[0].p.px, plist_Dspi_mc_truth[0].p.py, plist_Dspi_mc_truth[0].p.pz
                            Dspi_p = plist_Dspi_mc_truth[0].p.absvalue()

                            Ds_KKpi_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px, DsK1_py + DsK2_py + Dspi_py, DsK1_pz + DsK2_pz + Dspi_pz],[DsK1_px + DsK2_px + Dspi_px, DsK1_py + DsK2_py + Dspi_py, DsK1_pz + DsK2_pz + Dspi_pz])

                            Ds_KKpi_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En)*(DsK1_En + DsK2_En + Dspi_En) -  Ds_KKpi_ptot)

                            Dsmu_En = np.sqrt(plist_Dsmu_mc_truth[0].p.absvalue()**2 + plist_Dsmu_mc_truth[0].mass**2)

                            Dsmu_px, Dsmu_py, Dsmu_pz = plist_Dsmu_mc_truth[0].p.px, plist_Dsmu_mc_truth[0].p.py, plist_Dsmu_mc_truth[0].p.pz
                            Dsmu_p = plist_Dsmu_mc_truth[0].p.absvalue()
                            
                            #print np.sqrt(Dsmu_En*Dsmu_En - np.dot([Dsmu_px, Dsmu_py, Dsmu_pz],[Dsmu_px, Dsmu_py, Dsmu_pz]))
                            ###Invariant mass calculation

                            Bs_KKpimu_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px + Dsmu_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz])

                            Bs_KKpimu_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En + Dsmu_En)*(DsK1_En + DsK2_En + Dspi_En + Dsmu_En) -  Bs_KKpimu_ptot)

                            #Bs_KKpimu_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px + Dsmu_px + DsX_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + DsX_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + DsX_pz],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px + DsX_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + DsX_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + DsX_pz])

                            #Bs_KKpimu_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En + Dsmu_En + DsX_En)*(DsK1_En + DsK2_En + Dspi_En + Dsmu_En + DsX_En) -  Bs_KKpimu_ptot)

                            ##with gamma/pi0==
                            #Bs_KKpimu_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px + DsX_px + Dsst1_gp_px, DsK1_py + DsK2_py + Dspi_py + DsX_py + Dsst1_gp_py, DsK1_pz + DsK2_pz + Dspi_pz + DsX_pz + Dsst1_gp_pz],[DsK1_px + DsK2_px + Dspi_px + DsX_px + Dsst1_gp_px, DsK1_py + DsK2_py + Dspi_py + DsX_py + Dsst1_gp_py, DsK1_pz + DsK2_pz + Dspi_pz + DsX_pz + Dsst1_gp_pz])

                            #Bs_KKpimu_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En + DsX_En + Dsst1_gp_En)*(DsK1_En + DsK2_En + Dspi_En + DsX_En + Dsst1_gp_En) -  Bs_KKpimu_ptot)

                            #Bs_KKpimu_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px + Dsmu_px + Dsst1_gp_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + Dsst1_gp_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + Dsst1_gp_pz],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px + Dsst1_gp_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + Dsst1_gp_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + Dsst1_gp_pz])

                            #Bs_KKpimu_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En + Dsmu_En + Dsst1_gp_En)*(DsK1_En + DsK2_En + Dspi_En + Dsmu_En + Dsst1_gp_En) -  Bs_KKpimu_ptot)

                            #Bs_KKpimu_ptot = np.dot([DsK1_px + DsK2_px + Dspi_px + Dsmu_px + Bs_gammapi_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + Bs_gammapi_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + Bs_gammapi_pz],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px + Bs_gammapi_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + Bs_gammapi_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + Bs_gammapi_pz])

                            #Bs_KKpimu_m = np.sqrt((DsK1_En + DsK2_En + Dspi_En + Dsmu_En + Bs_gammapi_En)*(DsK1_En + DsK2_En + Dspi_En + Dsmu_En + Bs_gammapi_En) -  Bs_KKpimu_ptot)
        
                            #####Mass correction scheme

                            KKpimu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz])/pvsv_distance

                            KKpimu_per = np.dot([(DsK1_px + DsK2_px + Dspi_px + Dsmu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + Dsmu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance],[(DsK1_px + DsK2_px + Dspi_px + Dsmu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + Dsmu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance])

                            #KKpimu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[DsK1_px + DsK2_px + Dspi_px + Dsmu_px + DsX_px, DsK1_py + DsK2_py + Dspi_py + Dsmu_py + DsX_pz, DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + DsX_pz])/pvsv_distance

                            #KKpimu_per = np.dot([(DsK1_px + DsK2_px + Dspi_px + Dsmu_px + DsX_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + Dsmu_py + DsX_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + DsX_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance],[(DsK1_px + DsK2_px + Dspi_px + Dsmu_px + DsX_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + Dsmu_py + DsX_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + Dsmu_pz + DsX_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance])

                            #KKpimu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[DsK1_px + DsK2_px + Dspi_px + DsX_px, DsK1_py + DsK2_py + Dspi_py + DsX_py, DsK1_pz + DsK2_pz + Dspi_pz + DsX_pz])/pvsv_distance

                            #KKpimu_per = np.dot([(DsK1_px + DsK2_px + Dspi_px + DsX_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + DsX_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + DsX_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance],[(DsK1_px + DsK2_px + Dspi_px + DsX_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_py + DsK2_py + Dspi_py + DsX_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_pz + DsK2_pz + Dspi_pz + DsX_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance])

                            Bs_KKpimu_mcorr = np.sqrt(Bs_KKpimu_m*Bs_KKpimu_m + KKpimu_per) + np.sqrt(KKpimu_per)

        
                            #==================Smearing=========================

                            DsK1_pt = np.sqrt(DsK1_px*DsK1_px + DsK1_py*DsK1_py)
                            DsK2_pt = np.sqrt(DsK2_px*DsK2_px + DsK2_py*DsK2_py)
                            Dspi_pt = np.sqrt(Dspi_px*Dspi_px + Dspi_py*Dspi_py)
                            Dsmu_pT = np.sqrt(Dsmu_px*Dsmu_px + Dsmu_py*Dsmu_py)
                            
                            Ds_KKpi_pt = np.sqrt((DsK1_px + DsK2_px + Dspi_px)**2 + (DsK1_py + DsK2_py + Dspi_py)**2)

                            DsK1_resol = momentum_res(DsK1_pt)
                            DsK2_resol = momentum_res(DsK2_pt)
                            Dspi_resol = momentum_res(Dspi_pt)
                            Dsmu_resol = momentum_res(Dsmu_pT)
                            Ds_resol = momentum_res(Ds_pT)
                            
                            KKpi_resol = momentum_res(Ds_KKpi_pt)

                            DsK1, DsK2, Dspi, Dsmu = copy.deepcopy(plist_DsK_mc_truth[0]), copy.deepcopy(plist_DsK_mc_truth[1]), copy.deepcopy(plist_Dspi_mc_truth[0]), copy.deepcopy(plist_Dsmu_mc_truth[0])
                            
                            Ds_B = copy.deepcopy(plist_DsBs_mctruth[0])

                            #print DsK1.pdgid, DsK2.pdgid, Dspi.pdgid

                            if self.cfg_ana.smear_pv:
                                pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)

                            if self.cfg_ana.smear_sv:
                                sv = smear_vertex(sv, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
                                pv_Ds = smear_vertex(pv_Ds, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
                                
                            if self.cfg_ana.smear_tv:
                                sv_Ds = smear_vertex(sv_Ds, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                pv_mu = smear_vertex(pv_mu, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                            
                            if self.cfg_ana.smear_tv:
                                tv_K1 = smear_vertex(tv_K1, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                tv_K2 = smear_vertex(tv_K2, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                tv_pi = smear_vertex(tv_pi, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)

                            if self.cfg_ana.smear_momentum:
                                Ds_B.p = smear_momentum(Ds_B.p, Ds_resol, Ds_resol, Ds_resol)
                                DsK1.p = smear_momentum(DsK1.p, DsK1_resol, DsK1_resol, DsK1_resol)
                                DsK2.p = smear_momentum(DsK2.p, DsK2_resol, DsK2_resol, DsK2_resol)
                                Dspi.p = smear_momentum(Dspi.p, Dspi_resol, Dspi_resol, Dspi_resol)
                                Dsmu.p = smear_momentum(Dsmu.p, Dsmu_resol, Dsmu_resol, Dsmu_resol)

                            pvsv_sdistance = np.sqrt((sv.x - pv.x)**2 + (sv.y - pv.y)**2 + (sv.z - pv.z)**2)
                            
                            Ds_spx, Ds_spy, Ds_spz = Ds_B.p.px, Ds_B.p.py, Ds_B.p.pz
                            DsK1_spx, DsK1_spy, DsK1_spz = DsK1.p.px, DsK1.p.py, DsK1.p.pz
                            DsK2_spx, DsK2_spy, DsK2_spz = DsK2.p.px, DsK2.p.py, DsK2.p.pz
                            Dspi_spx, Dspi_spy, Dspi_spz = Dspi.p.px, Dspi.p.py, Dspi.p.pz
                            Dsmu_spx, Dsmu_spy, Dsmu_spz = Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz
                            Dsmu_spT = np.sqrt(Dsmu_spx**2 + Dsmu_spy**2)
                            Ds_spT = np.sqrt(Ds_spx**2 + Ds_spy**2)
                            
                            Ds_sEn = np.sqrt(Ds_B.p.absvalue()*Ds_B.p.absvalue() + Ds_B.mass*Ds_B.mass)
                            DsK1_sEn = np.sqrt(DsK1.p.absvalue()*DsK1.p.absvalue() + DsK1.mass*DsK1.mass)
                            DsK2_sEn = np.sqrt(DsK2.p.absvalue()*DsK2.p.absvalue() + DsK2.mass*DsK2.mass)
                            Dspi_sEn = np.sqrt(Dspi.p.absvalue()*Dspi.p.absvalue() + Dspi.mass*Dspi.mass)
                            Dsmu_sEn = np.sqrt(Dsmu.p.absvalue()*Dsmu.p.absvalue() + Dsmu.mass*Dsmu.mass)
        
                            Ds_KKpi_sptot = np.dot([(DsK1_spx + DsK2_spx + Dspi_spx),(DsK1_spy + DsK2_spy + Dspi_spy), (DsK1_spz + DsK2_spz + Dspi_spz)],[(DsK1_spx + DsK2_spx + Dspi_spx),(DsK1_spy + DsK2_spy + Dspi_spy), (DsK1_spz + DsK2_spz + Dspi_spz)])

                            Ds_KKpi_sm = np.sqrt((DsK1_sEn + DsK2_sEn + Dspi_sEn)*(DsK1_sEn + DsK2_sEn + Dspi_sEn) - Ds_KKpi_sptot)

                            Bs_KKpimu_sptot = np.dot([(DsK1_spx + DsK2_spx + Dspi_spx + Dsmu_spx),(DsK1_spy + DsK2_spy + Dspi_spy + Dsmu_spy), (DsK1_spz + DsK2_spz + Dspi_spz + Dsmu_spz)],[(DsK1_spx + DsK2_spx + Dspi_spx + Dsmu_spx),(DsK1_spy + DsK2_spy + Dspi_spy + Dsmu_spy), (DsK1_spz + DsK2_spz + Dspi_spz + Dsmu_spz)])

                            Bs_KKpimu_sm = np.sqrt((DsK1_sEn + DsK2_sEn + Dspi_sEn + Dsmu_sEn)*(DsK1_sEn + DsK2_sEn + Dspi_sEn + Dsmu_sEn) - Bs_KKpimu_sptot)

                            KKpimu_spar = np.dot([sv.x - pv.x, sv.y - pv.y, sv.z - pv.z],[DsK1_spx + DsK2_spx + Dspi_spx + Dsmu_spx, DsK1_spy + DsK2_spy + Dspi_spy + Dsmu_spy, DsK1_spz + DsK2_spz + Dspi_spz + Dsmu_spz])/pvsv_sdistance
        
                            KKpimu_sper = np.dot([(DsK1_spx + DsK2_spx + Dspi_spx + Dsmu_spx) - KKpimu_spar*(sv.x - pv.x)/pvsv_sdistance, (DsK1_spy + DsK2_spy + Dspi_spy + Dsmu_spy) - KKpimu_spar*(sv.y - pv.y)/pvsv_sdistance, (DsK1_spz + DsK2_spz + Dspi_spz + Dsmu_spz) - KKpimu_spar*(sv.z - pv.z)/pvsv_sdistance],[(DsK1_spx + DsK2_spx + Dspi_spx + Dsmu_spx) - KKpimu_spar*(sv.x - pv.x)/pvsv_sdistance, (DsK1_spy + DsK2_spy + Dspi_spy + Dsmu_spy) - KKpimu_spar*(sv.y - pv.y)/pvsv_sdistance, (DsK1_spz + DsK2_spz + Dspi_spz + Dsmu_spz) - KKpimu_spar*(sv.z - pv.z)/pvsv_sdistance])

                            Bs_KKpimu_smcorr = np.sqrt(Bs_KKpimu_sm*Bs_KKpimu_sm + KKpimu_sper) + np.sqrt(KKpimu_sper)
                            
                            #impact parameter calculation
                            
                            sdiffx = sv.x - pv_mu.x
                            sdiffy = sv.y - pv_mu.y
                            
                            #sdiffx = pv.x - pv_mu.x
                            #sdiffy = pv.y - pv_mu.y
                                                                
                            mu_sIPdist = abs(Dsmu.p.px*sdiffy - Dsmu.p.py*sdiffx)/np.sqrt(Dsmu.p.px**2 + Dsmu.p.py**2)
                            
                            Ds_sdiffx = sv.x - pv_Ds.x
                            Ds_sdiffy = sv.y - pv_Ds.y
                            Ds_sdiffz = sv.z - pv_Ds.z
                            
                            #Ds_sdiffx = pv.x - pv_Ds.x
                            #Ds_sdiffy = pv.y - pv_Ds.y
                            #Ds_sdiffz = pv.z - pv_Ds.z
                            
                            Ds_sIPdist = abs(Ds_B.p.px*Ds_sdiffy - Ds_B.p.py*Ds_sdiffx)/np.sqrt(Ds_B.p.px**2 + Ds_B.p.py**2)
                            
                            Ds_spxdir = Ds_B.p.px/Ds_B.p.absvalue()
                            Ds_spydir = Ds_B.p.py/Ds_B.p.absvalue()
                            Ds_spzdir = Ds_B.p.pz/Ds_B.p.absvalue()
                            
                            mu_spxdir = Dsmu.p.px/Dsmu.p.absvalue()
                            mu_spydir = Dsmu.p.py/Dsmu.p.absvalue()
                            mu_spzdir = Dsmu.p.pz/Dsmu.p.absvalue()
                            
                            Dsmu_spcp = np.cross([mu_spxdir, mu_spydir, mu_spzdir],[Ds_spxdir, Ds_spydir, Ds_spzdir])
                            
                            #============Muon transverse momentum wrt B flight direction=====================================
                            
                            Fd_B = np.subtract([sv.x, sv.y, sv.z],[pv.x, pv.y, pv.z])/pvsv_sdistance
                            #mu_spT_FdB = np.cross([Dsmu_spx,Dsmu_spy,Dsmu_spz],[Fd_B[0],Fd_B[1],Fd_B[2]])
                            #mu_spT_B = np.sqrt(np.dot([mu_spT_FdB[0],mu_spT_FdB[1],mu_spT_FdB[2]],[mu_spT_FdB[0],mu_spT_FdB[1],mu_spT_FdB[2]]))
                            
                            mu_sppar_B = np.dot([Fd_B[0],Fd_B[1],Fd_B[2]],[Dsmu.p.px,Dsmu.p.py,Dsmu.p.pz])*Fd_B
                            
                            mu_spT_FdB = np.sqrt(np.dot([Dsmu.p.px-mu_sppar_B[0],Dsmu.p.py-mu_sppar_B[1],Dsmu.p.pz-mu_sppar_B[2]],[Dsmu.p.px-mu_sppar_B[0],Dsmu.p.py-mu_sppar_B[1],Dsmu.p.pz-mu_sppar_B[2]]))
                            
                            #================================================================================================
                            
                            #Vertex fitting
                            
                            covmat_K1 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,DsK1_resol*DsK1_resol,0.0,0.0], [0.0,0.0,0.0,0.0,DsK1_resol*DsK1_resol,0.0],[0.0,0.0,0.0,0.0,0.0,DsK1_resol*DsK1_resol]])
                                
                            covmat_K2 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,DsK2_resol*DsK2_resol,0.0,0.0], [0.0,0.0,0.0,0.0,DsK2_resol*DsK2_resol,0.0],[0.0,0.0,0.0,0.0,0.0,DsK2_resol*DsK2_resol]])
                                                                
                            covmat_pi = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Dspi_resol*Dspi_resol,0.0,0.0], [0.0,0.0,0.0,0.0,Dspi_resol*Dspi_resol,0.0],[0.0,0.0,0.0,0.0,0.0,Dspi_resol*Dspi_resol]])
                            
                            Ds_vtx.setDaughter(0,  DsK1.charge, np.array([DsK1.p.px, DsK1.p.py, DsK1.p.pz]), 0.1*np.array([tv_K1.x, tv_K1.y, tv_K1.z]), covmat_K1)
                            Ds_vtx.setDaughter(1,  DsK2.charge, np.array([DsK2.p.px, DsK2.p.py, DsK2.p.pz]), 0.1*np.array([tv_K2.x, tv_K2.y, tv_K2.z]), covmat_K2)
                            Ds_vtx.setDaughter(2,  Dspi.charge, np.array([Dspi.p.px, Dspi.p.py, Dspi.p.pz]), 0.1*np.array([tv_pi.x, tv_pi.y, tv_pi.z]), covmat_pi)
    
                            Ds_vtx.fit(100)                            
                            Ds_fitvtx = 10*Ds_vtx.getVertex()
                            
                            Ds_fitvtx_diffx = sv_Ds.x - Ds_fitvtx[0]
                            Ds_fitvtx_diffy = sv_Ds.y - Ds_fitvtx[1]
                            Ds_fitvtx_diffz = sv_Ds.z - Ds_fitvtx[2]
                                
                            Ds_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Ds_vtx.getChi2(), Ds_vtx.getNDF())
                                
                            Ds_K1_fitp  = Ds_vtx.getDaughterMomentum(0)                                
                            Ds_K2_fitp  = Ds_vtx.getDaughterMomentum(1)                                
                            Ds_pi_fitp  = Ds_vtx.getDaughterMomentum(2)
                            
                            Ds_K1_fitptot = np.sqrt(Ds_K1_fitp[0]*Ds_K1_fitp[0] + Ds_K1_fitp[1]*Ds_K1_fitp[1] + Ds_K1_fitp[2]*Ds_K1_fitp[2])
                            Ds_K2_fitptot = np.sqrt(Ds_K2_fitp[0]*Ds_K2_fitp[0] + Ds_K2_fitp[1]*Ds_K2_fitp[1] + Ds_K2_fitp[2]*Ds_K2_fitp[2])
                            Ds_pi_fitptot = np.sqrt(Ds_pi_fitp[0]*Ds_pi_fitp[0] + Ds_pi_fitp[1]*Ds_pi_fitp[1] + Ds_pi_fitp[2]*Ds_pi_fitp[2])
                                
                            Ds_K1_fitpT = np.sqrt(Ds_K1_fitp[0]*Ds_K1_fitp[0] + Ds_K1_fitp[1]*Ds_K1_fitp[1])
                            Ds_K2_fitpT = np.sqrt(Ds_K2_fitp[0]*Ds_K2_fitp[0] + Ds_K2_fitp[1]*Ds_K2_fitp[1])
                            Ds_pi_fitpT = np.sqrt(Ds_pi_fitp[0]*Ds_pi_fitp[0] + Ds_pi_fitp[1]*Ds_pi_fitp[1])
                                
                            Ds_KKpi_fitpx = Ds_K1_fitp[0] + Ds_K2_fitp[0] + Ds_pi_fitp[0]
                            Ds_KKpi_fitpy = Ds_K1_fitp[1] + Ds_K2_fitp[1] + Ds_pi_fitp[1]
                            Ds_KKpi_fitpz = Ds_K1_fitp[2] + Ds_K2_fitp[2] + Ds_pi_fitp[2]
                            
                            Ds_KKpi_fitp = np.array([Ds_KKpi_fitpx,Ds_KKpi_fitpy,Ds_KKpi_fitpz])
                            Ds_KKpi_fitpmag = np.sqrt(Ds_KKpi_fitpx**2 + Ds_KKpi_fitpy**2 + Ds_KKpi_fitpz**2)
                            
                            mu_ptot = np.array([Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz])
                            mu_pv = np.array([pv_mu.x, pv_mu.y, pv_mu.z])
                            
                            Ds_vtxfit_charge = DsK1.charge + DsK2.charge + Dspi.charge
                            
                            #============================================================================================================
                            #Calculate point of closest approach (DCA) along between Ds and mu
                                
                            N_Dsmu = (np.cross([Ds_KKpi_fitp[0], Ds_KKpi_fitp[1], Ds_KKpi_fitp[2]],[Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz]))/(Ds_KKpi_fitpmag*Dsmu.p.absvalue())
                            N_Dsmu_mag = np.sqrt(np.dot([N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]]))
                                
                            pv_Ds_PCA = Ds_fitvtx + (Ds_KKpi_fitp/np.sqrt(Ds_KKpi_fitp[0]**2 + Ds_KKpi_fitp[1]**2 + Ds_KKpi_fitp[2]**2))*Ds_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,Dsmu)
                                
                            pv_mu_PCA = mu_pv + (mu_ptot/np.sqrt(Dsmu.p.px**2 + Dsmu.p.py**2 + Dsmu.p.pz**2))*mu_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,Dsmu)
                                
                            #Calculate distance of closest approach (PCA) of Ds and mu
                                
                            Dsmu_DCA = np.dot([(pv_Ds_PCA[0]-pv_mu_PCA[0]),(pv_Ds_PCA[1]-pv_mu_PCA[1]),(pv_Ds_PCA[2]-pv_mu_PCA[2])],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]])/N_Dsmu_mag
                                
                            #print Dsmu_DCA
                            #print pv_Ds_PCA[0], pv_Ds.x, pv_Ds_PCA[1], pv_Ds.y, pv_Ds_PCA[2], pv_Ds.z
                            #print pv_mu_PCA[0], pv_mu.x, plist_Dsmu_mc_truth[0].start_vertex.x
                            #print sv_Dsfit[0], sv_Dsfit[1], sv_Dsfit[2], pv_Ds.x, pv_Ds.y, pv_Ds.z
                            #==========================================================================================================
                            covmat_Ds = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0, 4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,KKpi_resol*KKpi_resol,0.0,0.0], [0.0,0.0,0.0,0.0,KKpi_resol*KKpi_resol,0.0],[0.0,0.0,0.0,0.0,0.0,KKpi_resol*KKpi_resol]])
                            
                            covmat_mu = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0,4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Dsmu_resol*Dsmu_resol,0.0,0.0], [0.0,0.0,0.0,0.0,Dsmu_resol*Dsmu_resol,0.0],[0.0,0.0,0.0,0.0,0.0,Dsmu_resol*Dsmu_resol]])
                            
                            Bs_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([pv_Ds_PCA[0], pv_Ds_PCA[1],pv_Ds_PCA[2]]),covmat_Ds)
                            
                            Bs_Dsmu_vtx.setDaughter(1, Dsmu.charge, np.array([Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz]), 0.1*np.array([pv_mu_PCA[0], pv_mu_PCA[1], pv_mu_PCA[2]]), covmat_mu)
                            
                            Bs_Dsmu_vtx.fit(100)
                            Bs_Dsmu_fitvtx = 10*Bs_Dsmu_vtx.getVertex()
                            Bs_Dsmu_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Bs_Dsmu_vtx.getChi2(), Bs_Dsmu_vtx.getNDF())
                            
                            #=============================================================================================================
                            
                            Dsmu_vtx.setDaughter(0,  Ds_B.charge, np.array([Ds_B.p.px, Ds_B.p.py, Ds_B.p.pz]), np.array([pv_Ds.x, pv_Ds.y, pv_Ds.z]), np.diag([0.01] * 6))
                            Dsmu_vtx.setDaughter(1, Dsmu.charge, np.array([Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz]), np.array([pv_mu.x, pv_mu.y, pv_mu.z]), np.diag([0.01] * 6))
    
                            Dsmu_vtx.fit(3)
                            
                            Dsmu_fitvtx = Dsmu_vtx.getVertex()
                            
                            Bs_vtx.setDaughter(0,  DsK1.charge, np.array([DsK1.p.px, DsK1.p.py, DsK1.p.pz]), np.array([tv_K1.x, tv_K1.y, tv_K1.z]), np.diag([0.01] * 6))
                            Bs_vtx.setDaughter(1,  DsK2.charge, np.array([DsK2.p.px, DsK2.p.py, DsK2.p.pz]), np.array([tv_K2.x, tv_K2.y, tv_K2.z]), np.diag([0.01] * 6))
                            Bs_vtx.setDaughter(2,  Dspi.charge, np.array([Dspi.p.px, Dspi.p.py, Dspi.p.pz]), np.array([tv_pi.x, tv_pi.y, tv_pi.z]), np.diag([0.01] * 6))
                                
                            Bs_vtx.setDaughter(3, Dsmu.charge, np.array([Dsmu.p.px, Dsmu.p.py, Dsmu.p.pz]), np.array([pv_mu.x, pv_mu.y, pv_mu.z]), np.diag([0.01] * 6))
                                
                            Bs_vtx.fit(3)                                
                            Bs_fitvtx = Bs_vtx.getVertex()
                            Bs_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Bs_vtx.getChi2(), Bs_vtx.getNDF())
                            
                            
                            #Dsmu_spdp = np.abs(np.dot([Ds_B.p.px - mu_B.p.px, Ds_B.p.py - mu_B.p.py, Ds_B.p.pz - mu_B.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            #Dsmu_spdp = np.abs(np.dot([mu_B.p.px - Ds_B.p.px, mu_B.p.py - Ds_B.p.py, mu_B.p.pz - Ds_B.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            Dsmu_spdp = np.abs(np.dot([pv_mu.x - pv_Ds.x, pv_mu.y - pv_Ds.y, pv_mu.z - pv_Ds.z],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                                           
                            Dsmu_sdist = np.abs(np.dot([Ds_B.p.px - Dsmu.p.px, Ds_B.p.py - Dsmu.p.py, Ds_B.p.pz - Dsmu.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))/np.sqrt(np.dot([Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            Dsmu_sDOCA = Dsmu_spdp/np.sqrt(np.dot([Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))

                            #===================================================

                            self.mc_truth_tree.fill('event_number',     event_number)
                            self.mc_truth_tree.fill('n_particles',      n_particles)
                            self.mc_truth_tree.fill('pv_x',             pv_mctruth.x)
                            self.mc_truth_tree.fill('pv_y',             pv_mctruth.y)
                            self.mc_truth_tree.fill('pv_z',             pv_mctruth.z)
                            self.mc_truth_tree.fill('sv_x',             sv_mctruth.x)
                            self.mc_truth_tree.fill('sv_y',             sv_mctruth.y)
                            self.mc_truth_tree.fill('sv_z',             sv_mctruth.z)
                            self.mc_truth_tree.fill('pvDs_x',           pv_Ds_mctruth.x)
                            self.mc_truth_tree.fill('pvDs_y',           pv_Ds_mctruth.y)
                            self.mc_truth_tree.fill('pvDs_z',           pv_Ds_mctruth.z)
                            self.mc_truth_tree.fill('svDs_x',           sv_Ds_mctruth.x)
                            self.mc_truth_tree.fill('svDs_y',           sv_Ds_mctruth.y)
                            self.mc_truth_tree.fill('svDs_z',           sv_Ds_mctruth.z)
                            self.mc_truth_tree.fill('pvsv_distance',    pvsv_distance)
                            self.mc_truth_tree.fill('bquark_px',        bquark_px)
                            self.mc_truth_tree.fill('bquark_py',        bquark_py)
                            self.mc_truth_tree.fill('bquark_pz',        bquark_pz)
                            self.mc_truth_tree.fill('bquark_npx',       bquark_npx)
                            self.mc_truth_tree.fill('bquark_npy',       bquark_npy)
                            self.mc_truth_tree.fill('bquark_npz',       bquark_npz)
                            self.mc_truth_tree.fill('bquark_p',         bquark_p)
                            self.mc_truth_tree.fill('bquark_E',         bquark_E)
                            self.mc_truth_tree.fill('bquarkSS_px',      bquarkSS_px)
                            self.mc_truth_tree.fill('bquarkSS_py',      bquarkSS_py)
                            self.mc_truth_tree.fill('bquarkSS_pz',      bquarkSS_pz)
                            self.mc_truth_tree.fill('bquarkSS_npx',     bquarkSS_npx)
                            self.mc_truth_tree.fill('bquarkSS_npy',     bquarkSS_npy)
                            self.mc_truth_tree.fill('bquarkSS_npz',     bquarkSS_npz)
                            self.mc_truth_tree.fill('bquarkSS_p',       bquarkSS_p)
                            self.mc_truth_tree.fill('bquarkSS_E',       bquarkSS_E)
                            self.mc_truth_tree.fill('bquarkOS_px',      bquarkOS_px)
                            self.mc_truth_tree.fill('bquarkOS_py',      bquarkOS_py)
                            self.mc_truth_tree.fill('bquarkOS_pz',      bquarkOS_pz)
                            self.mc_truth_tree.fill('bquarkOS_npx',     bquarkOS_npx)
                            self.mc_truth_tree.fill('bquarkOS_npy',     bquarkOS_npy)
                            self.mc_truth_tree.fill('bquarkOS_npz',     bquarkOS_npz)
                            self.mc_truth_tree.fill('bquarkOS_p',       bquarkOS_p)
                            self.mc_truth_tree.fill('bquarkOS_E',       bquarkOS_E)
                            self.mc_truth_tree.fill('B_px',             Bs_mc_truth.p.px)
                            self.mc_truth_tree.fill('B_py',             Bs_mc_truth.p.py)
                            self.mc_truth_tree.fill('B_pz',             Bs_mc_truth.p.pz)
                            self.mc_truth_tree.fill('B_p',              pb)
                            self.mc_truth_tree.fill('B_m',              Bs_mc_truth.mass)
                            self.mc_truth_tree.fill('B_ID',             Bs_mc_truth.pdgid)
                            self.mc_truth_tree.fill('B_q',              Bs_mc_truth.charge)
                            self.mc_truth_tree.fill('Ds_E',             Ds_En)
                            self.mc_truth_tree.fill('Ds_px',            Ds_px)
                            self.mc_truth_tree.fill('Ds_py',            Ds_py)
                            self.mc_truth_tree.fill('Ds_pz',            Ds_pz)
                            self.mc_truth_tree.fill('Ds_p',             Ds_p)
                            self.mc_truth_tree.fill('Ds_pT',            Ds_pT)
                            self.mc_truth_tree.fill('Ds1_E',            Ds1_En)
                            self.mc_truth_tree.fill('Ds1_px',           Ds1_px)
                            self.mc_truth_tree.fill('Ds1_py',           Ds1_py)
                            self.mc_truth_tree.fill('Ds1_pz',           Ds1_pz)
                            self.mc_truth_tree.fill('Ds1_p',            Ds1_p)
                            self.mc_truth_tree.fill('Ds1_q',            plist_Ds_mctruth[0].charge)
                            self.mc_truth_tree.fill('Ds1_m',            Ds1_m)
                            self.mc_truth_tree.fill('Ds1_ID',           plist_Ds_mctruth[0].pdgid)
                            self.mc_truth_tree.fill('Ds2_E',            Ds2_En)
                            self.mc_truth_tree.fill('Ds2_px',           Ds2_px)
                            self.mc_truth_tree.fill('Ds2_py',           Ds2_py)
                            self.mc_truth_tree.fill('Ds2_pz',           Ds2_pz)
                            self.mc_truth_tree.fill('Ds2_p',            Ds2_p)
                            self.mc_truth_tree.fill('Ds2_q',            plist_Ds_mctruth[1].charge)
                            self.mc_truth_tree.fill('Ds2_m',            Ds2_m)
                            self.mc_truth_tree.fill('Ds2_ID',           plist_Ds_mctruth[1].pdgid)
                            self.mc_truth_tree.fill('mu_px',            Dsmu_px)
                            self.mc_truth_tree.fill('mu_py',            Dsmu_py)
                            self.mc_truth_tree.fill('mu_pz',            Dsmu_pz)
                            self.mc_truth_tree.fill('mu_E',             Dsmu_En)
                            self.mc_truth_tree.fill('mu_p',             Dsmu_p)
                            self.mc_truth_tree.fill('mu_pT',            Dsmu_pT)                            
                            self.mc_truth_tree.fill('mu_m',             plist_Dsmu_mc_truth[0].mass)
                            self.mc_truth_tree.fill('mu_ID',            plist_Dsmu_mc_truth[0].pdgid)
                            self.mc_truth_tree.fill('mu_q',             plist_Dsmu_mc_truth[0].charge)
                            self.mc_truth_tree.fill('BsDs1Ds2_m',       BsDs1Ds2_m)
                            self.mc_truth_tree.fill('Ds_KKpi_m',        Ds_KKpi_m)
                            self.mc_truth_tree.fill('Bs_KKpimu_m',      Bs_KKpimu_m)
                            self.mc_truth_tree.fill('Bs_KKpimu_mcorr',  Bs_KKpimu_mcorr)
                            self.mc_truth_tree.fill('Pvis_SS_p',        Pvis_SS_p)
                            self.mc_truth_tree.fill('Pvis_SS_px',       Pvis_SS_px)
                            self.mc_truth_tree.fill('Pvis_SS_py',       Pvis_SS_py)
                            self.mc_truth_tree.fill('Pvis_SS_pz',       Pvis_SS_pz)
                            self.mc_truth_tree.fill('Pvis_SS_E',        Pvis_SS_E)
                            self.mc_truth_tree.fill('Pvis_SS_ID',       Pvis_SS_ID)
                            self.mc_truth_tree.fill('Pvis_OS_p',        Pvis_OS_p)
                            self.mc_truth_tree.fill('Pvis_OS_px',       Pvis_OS_px)
                            self.mc_truth_tree.fill('Pvis_OS_py',       Pvis_OS_py)
                            self.mc_truth_tree.fill('Pvis_OS_pz',       Pvis_OS_pz)
                            self.mc_truth_tree.fill('Pvis_OS_E',        Pvis_OS_E)
                            self.mc_truth_tree.fill('Pvis_OS_ID',       Pvis_OS_ID)
                            self.mc_truth_tree.fill('Pnu_SS_p',         Pnu_SS_p)
                            self.mc_truth_tree.fill('Pnu_OS_p',         Pnu_OS_p)
                            self.mc_truth_tree.fill('Pnu_px',           Pnu_px)
                            self.mc_truth_tree.fill('Pnu_py',           Pnu_py)
                            self.mc_truth_tree.fill('Pnu_pz',           Pnu_pz)
                            self.mc_truth_tree.fill('Pnu_p',            Pnu_p)
                            self.mc_truth_tree.fill('Thr1',             Thr1)
                            self.mc_truth_tree.fill('Thr1_px',          Thr1_px)
                            self.mc_truth_tree.fill('Thr1_py',          Thr1_py)
                            self.mc_truth_tree.fill('Thr1_pz',          Thr1_pz)
                            self.mc_truth_tree.fill('Thr2',             Thr2)
                            self.mc_truth_tree.fill('Thr2_px',          Thr2_px)
                            self.mc_truth_tree.fill('Thr2_py',          Thr2_py)
                            self.mc_truth_tree.fill('Thr2_pz',          Thr2_pz)
                            self.mc_truth_tree.fill('Thr3',             Thr3)
                            self.mc_truth_tree.fill('Thr3_px',          Thr3_px)
                            self.mc_truth_tree.fill('Thr3_py',          Thr3_py)
                            self.mc_truth_tree.fill('Thr3_pz',          Thr3_pz)
                            self.mc_truth_tree.fill('OblThr1',          OblThr1)
                            self.mc_truth_tree.fill('OblThr2',          OblThr2)
                            self.mc_truth_tree.fill('OblThr3',          OblThr3)
                            self.mc_truth_tree.fill('Sph1',             Sph1)
                            self.mc_truth_tree.fill('Sph1_px',          Sph1_px)
                            self.mc_truth_tree.fill('Sph1_py',          Sph1_py)
                            self.mc_truth_tree.fill('Sph1_pz',          Sph1_pz)
                            self.mc_truth_tree.fill('Sph2',             Sph2)
                            self.mc_truth_tree.fill('Sph2_px',          Sph2_px)
                            self.mc_truth_tree.fill('Sph2_py',          Sph2_py)
                            self.mc_truth_tree.fill('Sph2_pz',          Sph2_pz)
                            self.mc_truth_tree.fill('Sph3',             Sph3)
                            self.mc_truth_tree.fill('Sph3_px',          Sph3_px)
                            self.mc_truth_tree.fill('Sph3_py',          Sph3_py)
                            self.mc_truth_tree.fill('Sph3_pz',          Sph3_py)
                            self.mc_truth_tree.fill('LinSph1',          LinSph1)
                            self.mc_truth_tree.fill('LinSph1_px',       LinSph1_px)
                            self.mc_truth_tree.fill('LinSph1_py',       LinSph1_py)
                            self.mc_truth_tree.fill('LinSph1_pz',       LinSph1_pz)
                            self.mc_truth_tree.fill('LinSph2',          LinSph2)
                            self.mc_truth_tree.fill('LinSph2_px',       LinSph2_px)
                            self.mc_truth_tree.fill('LinSph2_py',       LinSph2_py)
                            self.mc_truth_tree.fill('LinSph2_pz',       LinSph2_pz)
                            self.mc_truth_tree.fill('LinSph3',          LinSph3)
                            self.mc_truth_tree.fill('LinSph3_px',       LinSph3_px)
                            self.mc_truth_tree.fill('LinSph3_py',       LinSph3_py)
                            self.mc_truth_tree.fill('LinSph3_pz',       LinSph3_pz)
                            self.mc_truth_tree.fill('mu_IPdist',        mu_IPdist)
                            self.mc_truth_tree.fill('Ds_IPdist',        Ds_IPdist)
                            
                            self.mc_truth_tree.tree.Fill()


                            #Fill smeared events

                              # filling event information
                            self.smeared_tree.fill('event_number',      event_number)
                            self.smeared_tree.fill('n_particles',       n_particles)
                            self.smeared_tree.fill('pv_sx',             pv.x)
                            self.smeared_tree.fill('pv_sy',             pv.y)
                            self.smeared_tree.fill('pv_sz',             pv.z)
                            self.smeared_tree.fill('sv_sx',             sv.x)
                            self.smeared_tree.fill('sv_sy',             sv.y)
                            self.smeared_tree.fill('sv_sz',             sv.z)
                            self.smeared_tree.fill('pvDs_sx',           pv_Ds.x)
                            self.smeared_tree.fill('pvDs_sy',           pv_Ds.y)
                            self.smeared_tree.fill('pvDs_sz',           pv_Ds.z)
                            self.smeared_tree.fill('svDs_sx',           sv_Ds.x)
                            self.smeared_tree.fill('svDs_sy',           sv_Ds.y)
                            self.smeared_tree.fill('svDs_sz',           sv_Ds.z)
                            self.smeared_tree.fill('pvsv_sdistance',    pvsv_sdistance)
                            self.smeared_tree.fill('Bs_Ds_sEn',         Ds_sEn)
                            self.smeared_tree.fill('Bs_Ds_spx',         Ds_spx)                            
                            self.smeared_tree.fill('Bs_Ds_spy',         Ds_spy)
                            self.smeared_tree.fill('Bs_Ds_spz',         Ds_spz)
                            self.smeared_tree.fill('Bs_Ds_sp',          Ds_B.p.absvalue())
                            self.smeared_tree.fill('Bs_Ds_spT',         Ds_spT)
                            self.smeared_tree.fill('DsK1_sE',           DsK1_sEn)
                            self.smeared_tree.fill('DsK1_spx',          DsK1_spx)
                            self.smeared_tree.fill('DsK1_spy',          DsK1_spy)
                            self.smeared_tree.fill('DsK1_spz',          DsK1_spz)
                            self.smeared_tree.fill('DsK1_sp',           DsK1.p.absvalue())
                            self.smeared_tree.fill('DsK2_sE',           DsK2_sEn)
                            self.smeared_tree.fill('DsK2_spx',          DsK2_spx)
                            self.smeared_tree.fill('DsK2_spy',          DsK2_spy)
                            self.smeared_tree.fill('DsK2_spz',          DsK2_spz)
                            self.smeared_tree.fill('DsK2_sp',           DsK2.p.absvalue())
                            self.smeared_tree.fill('Dspi_sE',           Dspi_sEn)
                            self.smeared_tree.fill('Dspi_spx',          Dspi_spx)
                            self.smeared_tree.fill('Dspi_spy',          Dspi_spy)
                            self.smeared_tree.fill('Dspi_spz',          Dspi_spz)
                            self.smeared_tree.fill('Dspi_sp',           Dspi.p.absvalue())
                            self.smeared_tree.fill('Dsmu_sE',           Dsmu_sEn)
                            self.smeared_tree.fill('Dsmu_spx',          Dsmu_spx)
                            self.smeared_tree.fill('Dsmu_spy',          Dsmu_spy)
                            self.smeared_tree.fill('Dsmu_spz',          Dsmu_spz)                            
                            self.smeared_tree.fill('Dsmu_sp',           Dsmu.p.absvalue())
                            self.smeared_tree.fill('Dsmu_spT',          Dsmu_spT)
                            self.smeared_tree.fill('Ds_KKpi_sm',        Ds_KKpi_sm)
                            self.smeared_tree.fill('Bs_KKpimu_sm',      Bs_KKpimu_sm)
                            self.smeared_tree.fill('Bs_KKpimu_smcorr',  Bs_KKpimu_smcorr)
                            self.smeared_tree.fill('Pvis_SS_sp',        Pvis_SS_sp)
                            self.smeared_tree.fill('Pvis_SS_sE',        Pvis_SS_sE)
                            self.smeared_tree.fill('Pvis_OS_sp',        Pvis_OS_sp)
                            self.smeared_tree.fill('Pvis_OS_sE',        Pvis_OS_sE)
                            self.smeared_tree.fill('mu_sIPdist',        mu_sIPdist)
                            self.smeared_tree.fill('Ds_sIPdist',        Ds_sIPdist)
                            self.smeared_tree.fill('Dsmu_sDOCA',        Dsmu_sDOCA)
                            self.smeared_tree.fill('Dsmu_fitvtx_x',     Dsmu_fitvtx[0])
                            self.smeared_tree.fill('Dsmu_fitvtx_y',     Dsmu_fitvtx[1])
                            self.smeared_tree.fill('Dsmu_fitvtx_z',     Dsmu_fitvtx[2])
                            self.smeared_tree.fill('Dsmu_Chi2',         Dsmu_vtx.getChi2())
                            self.smeared_tree.fill('Dsmu_NDF',          Dsmu_vtx.getNDF())
                            self.smeared_tree.fill('Dsmu_FitTest',      Dsmu_vtx.getChi2()/Dsmu_vtx.getNDF())
                            self.smeared_tree.fill('Dsmu_fitvtx_diffx', Dsmu_fitvtx[0]-sv.x)
                            self.smeared_tree.fill('Dsmu_fitvtx_diffy', Dsmu_fitvtx[1]-sv.y)
                            self.smeared_tree.fill('Dsmu_fitvtx_diffz', Dsmu_fitvtx[2]-sv.z)

                            self.smeared_tree.tree.Fill()
                            
                            #Fill smeared vtx
                            self.smeared_vtx.fill('pv_sx',              pv.x)
                            self.smeared_vtx.fill('pv_sy',              pv.y)
                            self.smeared_vtx.fill('pv_sz',              pv.z)
                            self.smeared_vtx.fill('sv_sx',              sv.x)
                            self.smeared_vtx.fill('sv_sy',              sv.y)
                            self.smeared_vtx.fill('sv_sz',              sv.z)
                            self.smeared_vtx.fill('pv_sdiffx',          pv.x-pv_mctruth.x)
                            self.smeared_vtx.fill('pv_sdiffy',          pv.y-pv_mctruth.y)
                            self.smeared_vtx.fill('pv_sdiffz',          pv.z-pv_mctruth.z)
                            self.smeared_vtx.fill('sv_sdiffx',          sv.x-sv_mctruth.x)
                            self.smeared_vtx.fill('sv_sdiffy',          sv.y-sv_mctruth.y)
                            self.smeared_vtx.fill('sv_sdiffz',          sv.z-sv_mctruth.z)
                            self.smeared_vtx.fill('pvsv_sdistance',     pvsv_sdistance)
                            self.smeared_vtx.fill('B_KKpimu_sm',        Bs_KKpimu_sm)
                            self.smeared_vtx.fill('B_KKpimu_smcorr',    Bs_KKpimu_smcorr)                            
                            self.smeared_vtx.fill('mu_spx',             Dsmu_spx)
                            self.smeared_vtx.fill('mu_spy',             Dsmu_spy)
                            self.smeared_vtx.fill('mu_spz',             Dsmu_spz)
                            self.smeared_vtx.fill('mu_sp',              Dsmu.p.absvalue())
                            self.smeared_vtx.fill('mu_spT',             Dsmu_spT)
                            self.smeared_vtx.fill('mu_spT_B',           mu_spT_FdB)
                            self.smeared_vtx.fill('mu_sIPdist',         mu_sIPdist)
                            self.smeared_vtx.fill('Ds_Chi2',            Ds_vtx.getChi2()/Ds_vtx.getNDF())
                            self.smeared_vtx.fill('Ds_NDF',             Ds_vtx.getNDF())
                            self.smeared_vtx.fill('Ds_CDF',             Ds_fitvtx_CDF)
                            self.smeared_vtx.fill('Ds_fitvtx_x',        Ds_fitvtx[0])
                            self.smeared_vtx.fill('Ds_fitvtx_y',        Ds_fitvtx[1])
                            self.smeared_vtx.fill('Ds_fitvtx_z',        Ds_fitvtx[2])
                            self.smeared_vtx.fill('Ds_fitvtx_diffx',    Ds_fitvtx_diffx)
                            self.smeared_vtx.fill('Ds_fitvtx_diffy',    Ds_fitvtx_diffy)
                            self.smeared_vtx.fill('Ds_fitvtx_diffz',    Ds_fitvtx_diffz) 
                            self.smeared_vtx.fill('B_fitvtx_Chi2',      Bs_vtx.getChi2())
                            self.smeared_vtx.fill('B_fitvtx_NDF',       Bs_vtx.getNDF())
                            self.smeared_vtx.fill('B_fitvtx_CDF',       Bs_fitvtx_CDF)
                            self.smeared_vtx.fill('B_fitvtx_x',         Bs_fitvtx[0])
                            self.smeared_vtx.fill('B_fitvtx_y',         Bs_fitvtx[1])
                            self.smeared_vtx.fill('B_fitvtx_z',         Bs_fitvtx[2])
                            self.smeared_vtx.fill('B_fitvtx_diffx',     Bs_fitvtx[0]-sv.x)
                            self.smeared_vtx.fill('B_fitvtx_diffy',     Bs_fitvtx[1]-sv.y)
                            self.smeared_vtx.fill('B_fitvtx_diffz',     Bs_fitvtx[2]-sv.z)
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_Chi2', Bs_Dsmu_vtx.getChi2())
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_NDF',  Bs_Dsmu_vtx.getNDF())
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_CDF',  Bs_Dsmu_fitvtx_CDF)
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_x',    Bs_Dsmu_fitvtx[0])
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_y',    Bs_Dsmu_fitvtx[1])
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_z',    Bs_Dsmu_fitvtx[2])
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_diffx',Bs_Dsmu_fitvtx[0]-sv.x)
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_diffy',Bs_Dsmu_fitvtx[1]-sv.y)
                            self.smeared_vtx.fill('B_Dsmu_fitvtx_diffz',Bs_Dsmu_fitvtx[2]-sv.z)
                            self.smeared_vtx.fill('pv_DsPCA_x',         pv_Ds_PCA[0])
                            self.smeared_vtx.fill('pv_DsPCA_y',         pv_Ds_PCA[1])
                            self.smeared_vtx.fill('pv_DsPCA_z',         pv_Ds_PCA[2])
                            self.smeared_vtx.fill('pv_muPCA_x',         pv_mu_PCA[0])
                            self.smeared_vtx.fill('pv_muPCA_y',         pv_mu_PCA[1])
                            self.smeared_vtx.fill('pv_muPCA_z',         pv_mu_PCA[2])                                
                            self.smeared_vtx.fill('Dsmu_DCA',           np.abs(Dsmu_DCA))
                            self.smeared_vtx.fill('Thr1',               Thr1)
                            self.smeared_vtx.fill('Thr1_px',            Thr1_px)
                            self.smeared_vtx.fill('Thr1_py',            Thr1_py)
                            self.smeared_vtx.fill('Thr1_pz',            Thr1_pz)
                            self.smeared_vtx.fill('Thr2',               Thr2)
                            self.smeared_vtx.fill('Thr2_px',            Thr2_px)
                            self.smeared_vtx.fill('Thr2_py',            Thr2_py)
                            self.smeared_vtx.fill('Thr2_pz',            Thr2_pz)
                            self.smeared_vtx.fill('Thr3',               Thr3)
                            self.smeared_vtx.fill('Thr3_px',            Thr3_px)
                            self.smeared_vtx.fill('Thr3_py',            Thr3_py)
                            self.smeared_vtx.fill('Thr3_pz',            Thr3_pz)
                            self.smeared_vtx.fill('ThrMaj1'            ,ThrMaj1)
                            self.smeared_vtx.fill('ThrMaj1_x'          ,ThrMaj1_x)
                            self.smeared_vtx.fill('ThrMaj1_y'          ,ThrMaj1_y)
                            self.smeared_vtx.fill('ThrMaj1_z'          ,ThrMaj1_z)
                            self.smeared_vtx.fill('ThrMaj2'            ,ThrMaj2)
                            self.smeared_vtx.fill('ThrMaj2_x'          ,ThrMaj2_x)
                            self.smeared_vtx.fill('ThrMaj2_y'          ,ThrMaj2_y)
                            self.smeared_vtx.fill('ThrMaj2_z'          ,ThrMaj2_z)
                            self.smeared_vtx.fill('ThrMaj3'            ,ThrMaj3)
                            self.smeared_vtx.fill('ThrMaj3_x'          ,ThrMaj3_x)
                            self.smeared_vtx.fill('ThrMaj3_y'          ,ThrMaj3_y)
                            self.smeared_vtx.fill('ThrMaj3_z'          ,ThrMaj3_z)                                
                            self.smeared_vtx.fill('ThrMaj2_OS_spT'     ,ThrMaj2_OS_spT)
                            self.smeared_vtx.fill('ThrMaj2_SS_spT'     ,ThrMaj2_SS_spT)
                            self.smeared_vtx.fill('Pvis_SS_sp'         ,Pvis_SS_sp)
                            self.smeared_vtx.fill('Pvis_SS_sE'         ,Pvis_SS_sE)
                            self.smeared_vtx.fill('Pvis_OS_sp'         ,Pvis_OS_sp)
                            self.smeared_vtx.fill('Pvis_OS_sE'         ,Pvis_OS_sE)
                            
                            self.smeared_vtx.tree.Fill()
                        

            # fill here only if successfully found all particles
            #if (Bs_mc_truth != None and
            #    D_mc_truth != None and
            #	K_Dplus_mc_truth != None and
            #	pi1_Dplus_mc_truth != None and
            # 	pi2_Dplus_mc_truth != None and
            # 	mu_mc_truth != None and
            #	munu_mc_truth != None) :
                # fill the tree

    def write(self, unusefulVar):
        self.rootfile.Write()
        self.rootfile.Close()

        pb_canvas = TCanvas('pb_canvas', 'B momentum', 600, 400)
        pb_canvas.cd()
        self.pb_hist.Draw()
        pb_canvas.Update()


        print('Total decays processed: {}'.format(self.counter))
        print('B0s events with momentum cut: {}'.format(self.pb_counter))
        #print('Ds(+/-) meson events: {}'.format(self.Ds_counter))
        print('Two Ds(+/-) meson events: {}'.format(self.Ds_counter))        
        print('Muons from Ds: {}'.format(self.mu_counter))
        print('Tau from B0s: {}'.format(self.tau_counter))

        print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
        print('Efficiency:\n\tMomentum of B cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter)))
        raw_input('Press ENTER when finished')
