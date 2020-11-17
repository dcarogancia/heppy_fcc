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

def momentum_res(pT):
    return (2.0e-5/pT + 1.0e-3)

def getEt(pEn):
    return np.sqrt(pEn.p.absvalue*pEn.p.absvalue + pEn.mass*pEn.mass)


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


class BsDsMuNuXKKPiAnalyzer_notau(Analyzer):
    def beginLoop(self, setup):
        self.start_time = time.time()
        self.last_timestamp = time.time()

        self.counter = 0 # Total number of processed decays
        self.pb_counter = 0 # Number of events with B momentum > 25 GeV
        self.Ds_counter = 0
        self.munu_counter = 0

        self.Bmu_counter = 0
        self.taumu_counter = 0

        self.tau_counter = 0


        gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

        # histograms to visualize cuts
        self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)

        super(BsDsMuNuXKKPiAnalyzer_notau, self).beginLoop(setup)
        #self.rootfile = TFile('/'.join([self.dirName, 'Bd2DmunuKKpi-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DmunuXAnl-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsmunuXAnl_vtxfit-100k.root']), 'recreate')
        #self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsmunuXAnl_vtxfit_signal-100k.root']), 'recreate')
        self.rootfile = TFile('/'.join([self.dirName, 'Bs_DsmunuXAnl_vtxfit_signal_evtpdl2019-1M.root']), 'recreate')
        
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
        self.mc_truth_tree.var('tv_K1_x')
        self.mc_truth_tree.var('tv_K1_y')
        self.mc_truth_tree.var('tv_K1_z')
        self.mc_truth_tree.var('tv_K2_x')
        self.mc_truth_tree.var('tv_K2_y')
        self.mc_truth_tree.var('tv_K2_z')
        self.mc_truth_tree.var('tv_pi_x')
        self.mc_truth_tree.var('tv_pi_y')
        self.mc_truth_tree.var('tv_pi_z')
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
    	self.mc_truth_tree.var('Ds_px')
        self.mc_truth_tree.var('Ds_py')
        self.mc_truth_tree.var('Ds_pz')
    	self.mc_truth_tree.var('Ds_E')
    	self.mc_truth_tree.var('D_m')
    	self.mc_truth_tree.var('Ds_p')
    	self.mc_truth_tree.var('Ds_pT')
    	self.mc_truth_tree.var('Ds_ID')
    	self.mc_truth_tree.var('Ds_q')
    	self.mc_truth_tree.var('mu_px')
        self.mc_truth_tree.var('mu_py')
        self.mc_truth_tree.var('mu_pz')
        self.mc_truth_tree.var('mu_E')
        self.mc_truth_tree.var('mu_ET')
    	self.mc_truth_tree.var('mu_m')
    	self.mc_truth_tree.var('mu_p')
    	self.mc_truth_tree.var('mu_pT')
    	self.mc_truth_tree.var('mu_ID')
        self.mc_truth_tree.var('mu_q')
        self.mc_truth_tree.var('Dsmu_pTmis')
        self.mc_truth_tree.var('Dsmu_par')
    	self.mc_truth_tree.var('Bs_Dsmu_m')
    	self.mc_truth_tree.var('Bs_Dsmu_mcorr')
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
    	self.mc_truth_tree.var('mu_IPdist2')
    	self.mc_truth_tree.var('mu_IP3dist')
    	self.mc_truth_tree.var('Ds_IPdist')
    	
    	self.mc_truth_KKpi = Tree(self.cfg_ana.mc_truth_KKpi_name, self.cfg_ana.mc_truth_KKpi_title)
    	self.mc_truth_KKpi.var('Ds_K1_px')
        self.mc_truth_KKpi.var('Ds_K1_py')
        self.mc_truth_KKpi.var('Ds_K1_pz')
    	self.mc_truth_KKpi.var('Ds_K1_E')
    	self.mc_truth_KKpi.var('Ds_K1_p')
        self.mc_truth_KKpi.var('Ds_K1_m')
    	self.mc_truth_KKpi.var('Ds_K1_ID')
    	self.mc_truth_KKpi.var('Ds_K1_q')
    	self.mc_truth_KKpi.var('Ds_K2_px')
        self.mc_truth_KKpi.var('Ds_K2_py')
        self.mc_truth_KKpi.var('Ds_K2_pz')
    	self.mc_truth_KKpi.var('Ds_K2_E')
    	self.mc_truth_KKpi.var('Ds_K2_p')
     	self.mc_truth_KKpi.var('Ds_K2_m')
    	self.mc_truth_KKpi.var('Ds_K2_ID')
    	self.mc_truth_KKpi.var('Ds_K2_q')
    	self.mc_truth_KKpi.var('Ds_pi_px')
        self.mc_truth_KKpi.var('Ds_pi_py')
        self.mc_truth_KKpi.var('Ds_pi_pz')
    	self.mc_truth_KKpi.var('Ds_pi_E')
    	self.mc_truth_KKpi.var('Ds_pi_p')
     	self.mc_truth_KKpi.var('Ds_pi_m')
    	self.mc_truth_KKpi.var('Ds_pi_ID')
    	self.mc_truth_KKpi.var('Ds_pi_q')
    	self.mc_truth_KKpi.var('Ds_KKpi_m')
    	self.mc_truth_KKpi.var('KKpimu_pTmis')
    	self.mc_truth_KKpi.var('KKpimu_par')
    	self.mc_truth_KKpi.var('Bs_KKpimu_m')
    	self.mc_truth_KKpi.var('Bs_KKpimu_mcorr')

    	self.smeared_tree = Tree(self.cfg_ana.smeared_tree_name, self.cfg_ana.smeared_tree_title)
    	self.smeared_tree.var('n_particles')
        self.smeared_tree.var('event_number')                                
        self.smeared_tree.var('Bs_Ds_spx')
        self.smeared_tree.var('Bs_Ds_spy')
        self.smeared_tree.var('Bs_Ds_spz')
        self.smeared_tree.var('Bs_Ds_sp')
        self.smeared_tree.var('Bs_Ds_spT')
        self.smeared_tree.var('Bs_mu_spx')
        self.smeared_tree.var('Bs_mu_spy')
        self.smeared_tree.var('Bs_mu_spz')
        self.smeared_tree.var('Bs_mu_sp')
        self.smeared_tree.var('Bs_mu_spT')
        self.smeared_tree.var('Ds_K1_spx')
        self.smeared_tree.var('Ds_K1_spy')
        self.smeared_tree.var('Ds_K1_spz')
        self.smeared_tree.var('Ds_K1_sp')
        self.smeared_tree.var('Ds_K2_spx')
        self.smeared_tree.var('Ds_K2_spy')
        self.smeared_tree.var('Ds_K2_spz')
        self.smeared_tree.var('Ds_K2_sp')
        self.smeared_tree.var('Ds_pi_spx')
        self.smeared_tree.var('Ds_pi_spy')
        self.smeared_tree.var('Ds_pi_spz')
        self.smeared_tree.var('Ds_pi_sp')
        self.smeared_tree.var('Dsmu_spTmis')        
        self.smeared_tree.var('Dsmu_spar')        
        self.smeared_tree.var('Bs_Dsmu_sm')
        self.smeared_tree.var('Bs_Dsmu_smcorr')
        self.smeared_tree.var('KKpi_sM')
        self.smeared_tree.var('KKpimu_sM')
        self.smeared_tree.var('KKpimu_spar')
        self.smeared_tree.var('KKpimu_spTmis')
        self.smeared_tree.var('Bs_KKpimu_smcorr')
        self.smeared_tree.var('Pvis_SS_sp')
        self.smeared_tree.var('Pvis_SS_sE')
        self.smeared_tree.var('Pvis_OS_sp')
        self.smeared_tree.var('Pvis_OS_sE')
        self.smeared_tree.var('mu_sIPdist')
        self.smeared_tree.var('mu_sIPdist2')
        self.smeared_tree.var('mu_sIP3dist')
        self.smeared_tree.var('Ds_sIPdist')
        self.smeared_tree.var('Dsmu_sDOCA')
        self.smeared_tree.var('Dsmu_Chi2')
        self.smeared_tree.var('Dsmu_NDF')
        self.smeared_tree.var('Dsmu_FitTest')        
        self.smeared_tree.var('Dsmu_fitvtx_x')
        self.smeared_tree.var('Dsmu_fitvtx_y')
        self.smeared_tree.var('Dsmu_fitvtx_z')
        self.smeared_tree.var('Bs_fitvtx_Chi2')
        self.smeared_tree.var('Bs_fitvtx_NDF')
        self.smeared_tree.var('Bs_fitvtx_CDF')
        self.smeared_tree.var('Bs_fitvtx_x')
        self.smeared_tree.var('Bs_fitvtx_y')
        self.smeared_tree.var('Bs_fitvtx_z')
        self.smeared_tree.var('Bs_fitvtx_diffx')
        self.smeared_tree.var('Bs_fitvtx_diffy')
        self.smeared_tree.var('Bs_fitvtx_diffz')
        self.smeared_tree.var('Ds_fitvtx_Chi2')
        self.smeared_tree.var('Ds_fitvtx_NDF')        
        self.smeared_tree.var('Ds_fitvtx_x')
        self.smeared_tree.var('Ds_fitvtx_y')
        self.smeared_tree.var('Ds_fitvtx_z')
        self.smeared_tree.var('Ds_K1_fitpx')
        self.smeared_tree.var('Ds_K1_fitpy')
        self.smeared_tree.var('Ds_K1_fitpz')
        self.smeared_tree.var('Ds_K1_fitptot')
        self.smeared_tree.var('Ds_K1_fitpT')
        self.smeared_tree.var('Ds_K2_fitpx')
        self.smeared_tree.var('Ds_K2_fitpy')
        self.smeared_tree.var('Ds_K2_fitpz')
        self.smeared_tree.var('Ds_K2_fitptot')
        self.smeared_tree.var('Ds_K2_fitpT')
        self.smeared_tree.var('Ds_pi_fitpx')
        self.smeared_tree.var('Ds_pi_fitpy')
        self.smeared_tree.var('Ds_pi_fitpz')
        self.smeared_tree.var('Ds_pi_fitptot')
        self.smeared_tree.var('Ds_pi_fitpT')
        
        
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
        self.smeared_vtx.var('tv_K1_sx')
        self.smeared_vtx.var('tv_K1_sy')
        self.smeared_vtx.var('tv_K1_sz')
        self.smeared_vtx.var('tv_K1_diffx')
        self.smeared_vtx.var('tv_K1_diffy')
        self.smeared_vtx.var('tv_K1_diffz')
        self.smeared_vtx.var('tv_K2_sx')
        self.smeared_vtx.var('tv_K2_sy')
        self.smeared_vtx.var('tv_K2_sz')
        self.smeared_vtx.var('tv_pi_sx')
        self.smeared_vtx.var('tv_pi_sy')
        self.smeared_vtx.var('tv_pi_sz')
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
        self.smeared_vtx.var('mu_sIPdist2')
        self.smeared_vtx.var('pvDs_sx')
        self.smeared_vtx.var('pvDs_sy')
        self.smeared_vtx.var('pvDs_sz')
        self.smeared_vtx.var('svDs_sx')
        self.smeared_vtx.var('svDs_sy')
        self.smeared_vtx.var('svDs_sz')
        self.smeared_vtx.var('Ds_Chi2')
        self.smeared_vtx.var('Ds_NDF')
        self.smeared_vtx.var('Ds_CDF')
        self.smeared_vtx.var('Ds_fitvtx_x')
        self.smeared_vtx.var('Ds_fitvtx_y')
        self.smeared_vtx.var('Ds_fitvtx_z')
        self.smeared_vtx.var('Ds_fitvtx_diffx')
        self.smeared_vtx.var('Ds_fitvtx_diffy')
        self.smeared_vtx.var('Ds_fitvtx_diffz')        
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

        B_mc_truth = None # B0s particle (MC truth)
        b_quarkSS_mc_truth = None
        b_quarkOS_mc_truth = None
        os_b_quark_mc_truth = None # b quark opposite side to B0s (MC truth)
        ss_b_quark_mc_truth = None # b quark same side to B0s (MC truth)
        
        pv_mctruth = None
        pv = None # primary vertex
        pv_sm = None
        sv_mctruth = None
        sv = None
        tv_mctruth = None
        tv = None

        pv_Ds_mctruth = None
        pv_Ds = None
        sv_Ds_mctruth = None
        sv_Ds = None
        
        pv_mu_mctruth = None
        pv_mu = None


        Bmu_mc_truth = None
        taumu_mc_truth = None
        Dmu_mc_truth = None
        K_mc_truth = None
        Ds_mctruth = None
        tau_mc_truth = None
        mu_mc_truth = None        
        Pvis_SS_mc_truth = None
        Pvis_SS = None
        Pvis_OS_mc_truth = None
        Pvis_OS = None

        Ds_B_mc_truth = None
        mu_B_mc_truth = None
        munu_B_mc_truth = None
        Ds_B = None
        mu_B = None
        
        Bs_gammapi_mc_truth = None

        DsK1_mc_truth = None
        DsK2_mc_truth = None
        Dspi_mc_truth = None
        Ds_K1 = None
        Ds_K2 = None
        Ds_pi = None

        K_Dminus_mc_truth = None
        pi1_Dminus_mc_truth = None
        pi2_Dminus_mc_truth = None
        K_Dminus = None
        pi1_Dminus = None
        pi2_Dminus = None

        pb = 0. # B momentum
        pvsv_distance = 0.0
        
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
        

        Ds_En = 0. #D energy
        Ds_px = 0.
        Ds_py = 0.
        Ds_px = 0.
        Ds_pT = 0.
        mu_En = 0. #muon energy
        mu_px = 0. #muon momentum
        mu_py = 0. #muon momentum
        mu_pz = 0. #muon momentum
        mu_p = 0. #muon momentum magnitude
        mu_pT = 0.0
        mu_ET = 0.0
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
        ThrMaj2 = 0.
        ThrMaj3 = 0.
        ThrMaj2_SS_spT = 0.
        ThrMaj2_OS_spT = 0.
        Pvis_SS_spx = 0.0
        Pvis_SS_spy = 0.0
        Pvis_SS_spz = 0.0
        Pvis_OS_spx = 0.0
        Pvis_OS_spy = 0.0
        Pvis_OS_spz = 0.0
        
        Dmu_Ptot = 0.  #Total momentum
        Dmu_sPtot = 0.  #Total momentum

        Dmunu_Mass = 0.
        Bs_Dsmu_m = 0.
        Bs_Dsmu_sm = 0.        
        Ds_B_sEn = 0. #D energy1
        mu_B_sEn = 0. #muon energy
        

        Ds_K1_En = 0. #Kaon energy
        Ds_K2_En = 0. #Pion energy
        Ds_pi_En = 0. #Pion energy
        Ds_K1_sEn = 0. #Kaon energy
        Ds_K2_sEn = 0. #Pion energy
        Ds_pi_sEn = 0. #Pion energy

        Ds_KKpi_m = 0. #Invariant mass of D meson
        KKpi_Ptot = 0  #Total
        KKpi_sm = 0. #Invariant mass of D meson
        KKpi_sPtot = 0.  #Total

        Bs_KKpimu_m = 0. #Invariant mass of D meson
        KKpimu_Ptot = 0.  #Total
        Bs_KKpimu_sm = 0. #Invariant mass of D meson
        KKpimu_sPtot = 0.  #Total
        
        KKpimu_par = 0.
        KKpimu_spar = 0.
        KKpimu_pTmis = 0.
        KKpimu_spTmis = 0.
        Bs_KKpimu_mcorr = 0.0
        Bs_KKpimu_smcorr = 0.0
        
        Bs_mu_spT = 0.
        Bs_mu_sp = 0.
        Bs_Ds_spT = 0.

        #resolutions
        Pvis_SS_resol = 0.0
        Pvis_OS_resol = 0.0
        Ds_resol = 0.0
        mu_resol = 0.0
        K1_resol = 0.0
        K2_resol = 0.0
        pi_resol = 0.0
        Ds_K1_pt = 0. #Kaon energy
        Ds_K2_pt = 0. #Pion energy
        Ds_pi_pt = 0. #Pion energy

        Dsmu_pTmis = 0.
        Dsmu_spTmis = 0.
        
        Bs_Dsmu_mcorr = 0.
        Bs_Dsmu_smcorr = 0.
        Dsmu_par = 0.
        Dsmu_spar = 0.
        pvsv_sdistance = 0.
        BD_vdistance = 0.0
        Bdec_vdistance = 0.0

        Bs_gammapi_px = 0.0
        Bs_gammapi_py = 0.0
        Bs_gammapi_pz = 0.0
        Bs_gammapi_p = 0.0
        Bs_gammapi_En = 0.0
        
        mu_IPdist = 0.0
        mu_sIPdist = 0.0
        mu_IPdist2 = 0.0
        mu_sIPdist2 = 0.0
        mu_IP3dist = 0.0
        mu_sIP3dist = 0.0
        
        mu_diffx = 0.0
        mu_diffy = 0.0
        mu_diffz = 0.0
        mu_sdiffx = 0.0
        mu_sdiffy = 0.0
        mu_sdiffz = 0.0
        
        mu_diffx2 = 0.0
        mu_diffy2 = 0.0
        mu_diffz2 = 0.0
        mu_sdiffx2 = 0.0
        mu_sdiffy2 = 0.0
        mu_sdiffz2 = 0.0
        
        Ds_IPdist = 0.0
        Ds_sIPdist = 0.0
        
        Ds_diffx = 0.0
        Ds_diffy = 0.0
        Ds_diffz = 0.0
        Ds_sdiffx = 0.0
        Ds_sdiffy = 0.0
        Ds_sdiffz = 0.0
        
        Ds_sptrk = 0.0
        mu_sptrk = 0.0
        
        Ds_spxdir = 0.0
        Ds_spydir = 0.0
        Ds_spzdir = 0.0
        
        mu_spxdir = 0.0
        mu_spydir = 0.0
        mu_spzdir = 0.0
        
        fitvtx = 0.0
        Ds_fitvtx_diffx = 0.0
        Ds_fitvtx_diffy = 0.0
        Ds_fitvtx_diffz = 0.0
        
        mu_spT_B = 0.0
        mu_sppar_B = 0.0
        
        store = event.input # This is just a shortcut
        event_info = store.get("EventInfo")
        particles_info = store.get("GenParticle")
        vertices_info = store.get("GenVertex")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        # looking for B
        #print event_number
        plist_bquark_mc_truth = list([])
        plist_bquarkSS_mc_truth = list([])
        plist_bquarkOS_mc_truth = list([])
        plist_Ds_mctruth = list([])
        plist_Dsst_mc_truth = list([])
        
        plist_Ds0st_mc_truth = list([])
        plist_Ds1_mc_truth = list([])
        plist_Ds1prime_mc_truth = list([])

        plist_taumu_mc_truth = list([])
        plist_Bmu_mc_truth = list([])
        plist_BD1_mc_truth = list([])
        plist_Bmu1_mc_truth = list([])
        plist_Bmunu_mc_truth = list([])
        plist_Btau_mc_truth = list([])
        plist_Bpi_mc_truth = list([])
        plist_Bpi0_mc_truth = list([])
        plist_KKpi_mc_truth = list([])

        plist_Bdaughters = list([])
        plist_B0s = list([])
        
        
        for ptc_gen1 in ptcs:
            
            #print ptc_gen1.pdgid, ptc_gen1.status
            
            #if ptc_gen1.status == 1 and ptc_gen1.charge == 0:
            #    p_vis = ptc_gen1.p.absvalue()
            
            if abs(ptc_gen1.pdgid) == 531 and (ptc_gen1.start_vertex != ptc_gen1.end_vertex): # if B found and it's not an oscillation

                self.counter += 1
                if self.counter %1000 == 0:
                    print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
                    self.last_timestamp = time.time()

                B_mc_truth = ptc_gen1
                pb = B_mc_truth.p.absvalue()

                plist_B0s.append(ptc_gen1)
                
                fitter = FastFit.FastFit(2, 0)
                Ds_vtx = FastFit.FastFit(3, 0)
                Bs_Dsmu_vtx = FastFit.FastFit(2, 0)

                #print 'B0s id: ', B_mc_truth.pdgid

                if pb > 0.0: # select only events with large momentum of the B
                    self.pb_counter += 1                    
                    
                    # looking for opposite b quark. This is a dirty hack. Works only because both PYTHIA/HepMC and PODIO store particles ordered. But IT'S NOT GUARANTEED
		            # need to find better algorithm to look for the opposite b-quark
                    index = 0
                    while os_b_quark_mc_truth == None and index < len(ptcs):
                        if (abs(ptcs[index].pdgid) == 5 and ptcs[index].status == 23) and np.dot([B_mc_truth.p.px, B_mc_truth.p.py, B_mc_truth.p.pz], [ptcs[index].p.px, ptcs[index].p.py, ptcs[index].p.pz]) < 0:
                            os_b_quark_mc_truth = ptcs[index]
                            #print os_b_quark_mc_truth.start_vertex
                        index += 1
                    
                    #print os_b_quark_mc_truth.status
                    
                    pv_mctruth = B_mc_truth.start_vertex
                    pv = copy.deepcopy(pv_mctruth)

                    sv_mctruth = B_mc_truth.end_vertex
                    sv = copy.deepcopy(sv_mctruth)

                    pvsv_distance = np.sqrt((sv_mctruth.x - pv_mctruth.x)**2 + (sv_mctruth.y - pv_mctruth.y)**2 + (sv_mctruth.z - pv_mctruth.z)**2)
                    
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
                            
                            #    print ptc_gen2.pdgid
                            #if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23):
                            #    print np.dot([ptc_gen2.p.px/ptc_gen2.p.absvalue(),ptc_gen2.p.py/ptc_gen2.p.absvalue(), ptc_gen2.p.pz/ptc_gen2.p.absvalue()],[Thr2_px, Thr2_py,Thr2_pz])
                            #if (abs(ptc_gen2.pdgid) == 5 and ptc_gen2.status == 23) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0):
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
                                
                                Pvis_SS_spx = Pvis_SS.p.px
                                Pvis_SS_spy = Pvis_SS.p.py
                                Pvis_SS_spz = Pvis_SS.p.pz
                                
                                Pvis_SS_sp += Pvis_SS.p.absvalue()
                                Pvis_SS_sE += np.sqrt(Pvis_SS.p.absvalue()*Pvis_SS.p.absvalue() + Pvis_SS.mass*Pvis_SS.mass)
                                #ThrMaj2_SS_spT = np.sqrt(np.dot([Pvis_SS_spx,Pvis_SS_spy,Pvis_SS_spz],[ThrMaj2_x,ThrMaj2_y,ThrMaj2_z]))
                                #ThrMaj2_SS_spT = ThrMaj2_x
                                #print np.dot([Pvis_SS_spx,Pvis_SS_spy,Pvis_SS_spz],[ThrMaj2_x,ThrMaj2_y,ThrMaj2_z])
                                
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
                                #ThrMaj2_OS_spT = np.sqrt(np.dot([Pvis_OS.p.px, Pvis_OS.p.py, Pvis_OS.pz],[ThrMaj2_x, ThrMaj2_y, ThrMaj2_z]))
                                

                            #if (ptc_gen2.start_vertex == B_mc_truth.end_vertex) and (np.dot([ptc_gen2.p.px,ptc_gen2.p.py, ptc_gen2.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0):
                            
                            Pnu_SS_p = np.sqrt(np.dot([Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz],[Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz]))
                            
                            Pnu_OS_p = np.sqrt(np.dot([Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz],[Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz]))
                            
                            if (ptc_gen2.start_vertex == B_mc_truth.end_vertex):
                                
                                plist_Bdaughters.append(ptc_gen2)

                                BD_vdistance = np.sqrt((B_mc_truth.end_vertex.x - ptc_gen2.start_vertex.x)**2 + (B_mc_truth.end_vertex.y - ptc_gen2.start_vertex.y)**2 + (B_mc_truth.end_vertex.z - ptc_gen2.start_vertex.z)**2)
                                
                                #if ptc_gen2.status == 1 and (ptc_gen2.pdgid not in [14,16,18]):
                                #    p_vis += ptc_gen2.p.absvalue()
                                #    E_vis += ptc_gen2.energy
                                #if abs(ptc_gen2.charge) > 0:                                
                                #    print ptc_gen2.pdgid, ptc_gen2.charge
                                
                                #if abs(ptc_gen2.pdgid) == 13:
                                #    diffx = pv_mctruth.x - sv_mctruth.x
                                #    diffy = pv_mctruth.y - sv_mctruth.y                                    
                                #    mu_IPdist = abs(ptc_gen2.p.px*diffx - ptc_gen2.p.py*diffy)/np.sqrt(ptc_gen2.p.px**2 + ptc_gen2.p.py**2)
                                    
                                    #print mu_IPdist
                                
                                #print pb, pvsv_distance, ptc_gen2.pdgid, ptc_gen2.p.px, ptc_gen2.p.py, ptc_gen2.p.pz, ptc_gen2.p.absvalue(), ptc_gen2.mass
        
                                #looking for muon neutrino
                                if abs(ptc_gen2.pdgid) == 14:
                                    munu_mc_truth = ptc_gen2
                                    plist_Bmunu_mc_truth.append(ptc_gen2)
                                    self.munu_counter += 1

                                #lookging for tau
                                if abs(ptc_gen2.pdgid) == 15:
                                    plist_Btau_mc_truth.append(ptc_gen2)
                                    tau_mc_truth = ptc_gen2
                                    self.tau_counter += 1

                                #if abs(ptc_gen2.pdgid) == 431 and BD_vdistance < 0.25:
                                if abs(ptc_gen2.pdgid) == 431:
                                    plist_Ds_mctruth.append(ptc_gen2)
                                    Ds_mctruth = ptc_gen2
                                    pv_Ds_mctruth = Ds_mctruth.start_vertex
                                    pv_Ds = copy.deepcopy(pv_Ds_mctruth)
                                    sv_Ds_mctruth = Ds_mctruth.end_vertex
                                    sv_Ds = copy.deepcopy(sv_Ds_mctruth)
                                    
                                    Ds_diffx = sv_mctruth.x - sv_Ds_mctruth.x
                                    Ds_diffy = sv_mctruth.y - sv_Ds_mctruth.y
                                    Ds_diffz = sv_mctruth.z - sv_Ds_mctruth.z
                                    
                                    #print Ds_diffx, Ds_diffy, Ds_diffz, ptc_gen2.p.px, ptc_gen2.p.py, ptc_gen2.p.px*Ds_diffy, ptc_gen2.p.py*Ds_diffx
                                                                        
                                    Ds_IPdist = abs(ptc_gen2.p.px*Ds_diffy - ptc_gen2.p.py*Ds_diffx)/np.sqrt(ptc_gen2.p.px**2 + ptc_gen2.p.py**2)                                 
                                    
                                    self.Ds_counter += 1

                                #if abs(ptc_gen3.pdgid) == 13 and BD_vdistance < 0.25 or (tau_mc_truth != None and ptc_gen3.start_vertex == tau_mc_truth.end_vertex):
                                if abs(ptc_gen2.pdgid) == 13 and BD_vdistance < 0.25:                                    
                                    #plist_Bmu_mc_truth.append(ptc_gen2)
                                    Bmu_mc_truth = ptc_gen2                                    
                                    self.Bmu_counter += 1                                   
                                   
                                    #pv_mu_mctruth = ptc_gen2.start_vertex
                                    #pv_mu = copy.deepcopy(pv_mu_mctruth)
                                    
                                    #mu_diffx = pv_mctruth.x - pv_mu_mctruth.x
                                    #mu_diffy = pv_mctruth.y - pv_mu_mctruth.y
                                    #mu_diffz = pv_mctruth.z - pv_mu_mctruth.z
                                    
                                    #mu_diffx2 = sv_mctruth.x - pv_mu_mctruth.x 
                                    #mu_diffy2 = sv_mctruth.y - pv_mu_mctruth.y 
                                    #mu_diffz2 = sv_mctruth.z - pv_mu_mctruth.z 
                                    
                                    #mu_IP3dist = np.sqrt((ptc_gen3.p.py*diffz - ptc_gen3.p.pz*diffy)**2 - (ptc_gen3.p.px*diffz - ptc_gen3.p.pz*diffx)**2 + (ptc_gen3.p.px*diffy - ptc_gen3.p.py*diffx)**2)/np.sqrt(ptc_gen3.p.px**2 + ptc_gen3.p.py**2 + ptc_gen3.p.pz**2)
                                    
                                    #mu_IPdist = abs(ptc_gen2.p.px*mu_diffy - ptc_gen2.p.py*mu_diffx)/np.sqrt(ptc_gen2.p.px**2 + ptc_gen2.p.py**2)                                
                                    #mu_IPdist2 = abs(ptc_gen2.p.px*mu_diffy2 - ptc_gen2.p.py*mu_diffx2)/np.sqrt(ptc_gen2.p.px**2 + ptc_gen2.p.py**2)
        
                                #looking for photons
                                #if abs(ptc_gen2.pdgid) == 22 or abs(ptc_gen2.pdgid) == 111 or abs(ptc_gen2.pdgid) == 211 and BD_vdistance < 0.25:
                                if abs(ptc_gen2.pdgid) == 111 or abs(ptc_gen2.pdgid) == 211 and BD_vdistance < 0.25:
                                #if abs(ptc_gen2.pdgid) == 22 and BD_vdistance < 0.25:
                                    Bs_gammapi_mc_truth = ptc_gen2
                                    if Bs_gammapi_mc_truth != None:
                                        Bs_gammapi_px = Bs_gammapi_px + ptc_gen2.p.px
                                        Bs_gammapi_py = Bs_gammapi_py + ptc_gen2.p.py
                                        Bs_gammapi_pz = Bs_gammapi_pz + ptc_gen2.p.pz
                                        Bs_gammapi_p = Bs_gammapi_p + ptc_gen2.p.absvalue()
                                        Bs_gammapi_En = Bs_gammapi_p + ptc_gen2.p.absvalue()
                                    else:
                                        Bs_gammapi_px = 0.0
                                        Bs_gammapi_py = 0.0
                                        Bs_gammapi_pz = 0.0
                                        Bs_gammapi_En = 0.0
                                    #print ptc_gen2.pdgid
                        
                        #print bquarkSS_p, np.sqrt(ptc_gen2.p.px**2 + ptc_gen2.p.py**2 + ptc_gen2.p.pz**2)

                        for ptc_gen3 in ptcs:
                            #looking for Ds meson
                            #print 'ptcgen3: ', ptc_gen3.pdgid
                            
                            if abs(ptc_gen3.pdgid) == 13 and (tau_mc_truth != None and ptc_gen3.start_vertex == tau_mc_truth.end_vertex):
                                self.taumu_counter += 1
                                

                            if abs(ptc_gen3.pdgid) == 13 and ((tau_mc_truth != None and ptc_gen3.start_vertex == tau_mc_truth.end_vertex) or (Bmu_mc_truth != None and ptc_gen3.start_vertex == Bmu_mc_truth.start_vertex)):
                                #print 'Muon from tau', ptc_gen3.pdgid
                                plist_Bmu_mc_truth.append(ptc_gen3)
                                
                                pv_mu_mctruth = ptc_gen3.start_vertex
                                pv_mu = copy.deepcopy(pv_mu_mctruth)
                                
                                mu_diffx = pv_mctruth.x - pv_mu_mctruth.x
                                mu_diffy = pv_mctruth.y - pv_mu_mctruth.y
                                mu_diffz = pv_mctruth.z - pv_mu_mctruth.z
                                
                                mu_diffx2 = sv_mctruth.x - pv_mu_mctruth.x 
                                mu_diffy2 = sv_mctruth.y - pv_mu_mctruth.y 
                                mu_diffz2 = sv_mctruth.z - pv_mu_mctruth.z 
                                
                                #mu_IP3dist = np.sqrt((ptc_gen3.p.py*diffz - ptc_gen3.p.pz*diffy)**2 - (ptc_gen3.p.px*diffz - ptc_gen3.p.pz*diffx)**2 + (ptc_gen3.p.px*diffy - ptc_gen3.p.py*diffx)**2)/np.sqrt(ptc_gen3.p.px**2 + ptc_gen3.p.py**2 + ptc_gen3.p.pz**2)
                                
                                mu_IPdist = abs(ptc_gen3.p.px*mu_diffy - ptc_gen3.p.py*mu_diffx)/np.sqrt(ptc_gen3.p.px**2 + ptc_gen3.p.py**2)                                
                                mu_IPdist2 = abs(ptc_gen3.p.px*mu_diffy2 - ptc_gen3.p.py*mu_diffx2)/np.sqrt(ptc_gen3.p.px**2 + ptc_gen3.p.py**2)
                                
                            #if abs(ptc_gen3.pdgid) == 22 and Bs_gammapi_mc_truth != None:
                            #if Bs_gammapi_mc_truth != None:
                            #    Bs_gammapi_px = Bs_gammapi_px + ptc_gen3.p.px
                            #    Bs_gammapi_py = Bs_gammapi_py + ptc_gen3.p.py
                            #    Bs_gammapi_pz = Bs_gammapi_pz + ptc_gen3.p.pz
                            #    Bs_gammapi_p = Bs_gammapi_p + ptc_gen3.p.absvalue()
                            #    Bs_gammapi_En = Bs_gammapi_p + ptc_gen3.p.absvalue()
                            #else:
                            #    Bs_gammapi_px = 0.0
                            #    Bs_gammapi_py = 0.0
                            #    Bs_gammapi_pz = 0.0
                            #    Bs_gammapi_En = 0.0
                        
                        if len(plist_Ds_mctruth) == 1 and len(plist_Bmu_mc_truth) == 1:
                        #if len(plist_Bmu_mc_truth) == 1:
                                                    
                            Ds_px, Ds_py, Ds_pz = Ds_mctruth.p.px, Ds_mctruth.p.py, Ds_mctruth.p.pz
                            Ds_En = np.sqrt(Ds_mctruth.p.absvalue()*Ds_mctruth.p.absvalue() + Ds_mctruth.mass*Ds_mctruth.mass)
                            Ds_pT = np.sqrt(Ds_mctruth.p.px*Ds_mctruth.p.px + Ds_mctruth.p.py*Ds_mctruth.p.py)
                            
                            #print Ds_En, Ds_mctruth.energy

                            mu_En = np.sqrt(plist_Bmu_mc_truth[0].p.absvalue()*plist_Bmu_mc_truth[0].p.absvalue() + plist_Bmu_mc_truth[0].mass*plist_Bmu_mc_truth[0].mass)

                            mu_px, mu_py, mu_pz = plist_Bmu_mc_truth[0].p.px, plist_Bmu_mc_truth[0].p.py, plist_Bmu_mc_truth[0].p.pz
                            mu_p = plist_Bmu_mc_truth[0].p.absvalue()
                            mu_pT = np.sqrt(plist_Bmu_mc_truth[0].p.px**2 + plist_Bmu_mc_truth[0].p.py**2)
                            mu_ET = np.sqrt(plist_Bmu_mc_truth[0].mass*plist_Bmu_mc_truth[0].mass + mu_pT*mu_pT)
                            
                            #if Ds_mctruth.p.absvalue() < 45 or mu_p < 45:
                            #    print "Muon mass: ", plist_Bmu_mc_truth[0].mass

                            Dmu_Ptot = np.dot([Ds_px + mu_px, Ds_py + mu_py, Ds_pz + mu_pz],[Ds_px + mu_px, Ds_py + mu_py, Ds_pz + mu_pz])
                            Bs_Dsmu_m = np.sqrt((Ds_En + mu_En)*(Ds_En + mu_En) - Dmu_Ptot)

                            #Dmu_Ptot = np.dot([Ds_px + mu_px + Bs_gammapi_px, Ds_py + mu_py + Bs_gammapi_py, Ds_pz + mu_pz + Bs_gammapi_pz],[Ds_px + mu_px + Bs_gammapi_px, Ds_py + mu_py + Bs_gammapi_py, Ds_pz + mu_pz + Bs_gammapi_pz])

                            #Bs_Dsmu_m = np.sqrt((Ds_En + mu_En + Bs_gammapi_En)*(Ds_En + mu_En + Bs_gammapi_En) - Dmu_Ptot)

                            Ds_B_mc_truth, mu_B_mc_truth = plist_Ds_mctruth[0], plist_Bmu_mc_truth[0]
                            Ds_B, mu_B = copy.deepcopy(Ds_B_mc_truth), copy.deepcopy(mu_B_mc_truth)

                            Dsmu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[Ds_px + mu_px, Ds_py + mu_py, Ds_pz + mu_pz])/pvsv_distance

                            Dsmu_pTmis = np.dot([(Ds_px + mu_px) - Dsmu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (Ds_py + mu_py) - Dsmu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (Ds_pz + mu_pz) - Dsmu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance], [(Ds_px + mu_px) - Dsmu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (Ds_py + mu_py) - Dsmu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (Ds_pz + mu_pz) - Dsmu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance])

                            Bs_Dsmu_mcorr = np.sqrt(Bs_Dsmu_m*Bs_Dsmu_m + Dsmu_pTmis) + np.sqrt(Dsmu_pTmis)

                            Ds_resol = momentum_res(Ds_pT)
                            mu_resol = momentum_res(mu_pT)

                            if self.cfg_ana.smear_pv:
                                pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)
                                #pv_sm = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)                               

                            if self.cfg_ana.smear_sv:
                                sv = smear_vertex(sv, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
                                pv_Ds = smear_vertex(pv_Ds, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
 
                            if self.cfg_ana.smear_pv:
                                pv_mu = smear_vertex(pv_mu, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)

                            if self.cfg_ana.smear_momentum:                                
                                Ds_B.p = smear_momentum(Ds_B.p, Ds_resol, Ds_resol, Ds_resol)
                                mu_B.p = smear_momentum(mu_B.p, mu_resol, mu_resol, mu_resol)
                                
                                #Ds_B.p = smear_momentum(Ds_B.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
                                #mu_B.p = smear_momentum(mu_B.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
                            
                                                            
                            Ds_B_sEn = np.sqrt(Ds_B.p.absvalue()*Ds_B.p.absvalue() + Ds_B.mass*Ds_B.mass)
                            mu_B_sEn = np.sqrt(mu_B.p.absvalue()*mu_B.p.absvalue() + mu_B.mass*mu_B.mass)
                            
                            Bs_mu_sp = mu_B.p.absvalue()
                            Bs_mu_spT = np.sqrt(mu_B.p.px*mu_B.p.px + mu_B.p.py*mu_B.p.py)
                            
                            Bs_Ds_spT = np.sqrt(Ds_B.p.px*Ds_B.p.px + Ds_B.p.py*Ds_B.p.py)

                            Dmu_sPtot = np.dot([Ds_B.p.px + mu_B.p.px, Ds_B.p.py + mu_B.p.py, Ds_B.p.pz + mu_B.p.pz], [Ds_B.p.px + mu_B.p.px, Ds_B.p.py + mu_B.p.py, Ds_B.p.pz + mu_B.p.pz])

                            Bs_Dsmu_sm = np.sqrt((Ds_B_sEn + mu_B_sEn)*(Ds_B_sEn + mu_B_sEn) - Dmu_sPtot)

                            pvsv_sdistance = np.sqrt((sv.x-pv.x)**2 + (sv.y-pv.y)**2 + (sv.z-pv.z)**2)

                            Dsmu_spar = np.dot([(sv.x-pv.x), (sv.y-pv.y), (sv.z-pv.z)],[Ds_B.p.px + mu_B.p.px, Ds_B.p.py + mu_B.p.py, Ds_B.p.pz + mu_B.p.pz])/pvsv_sdistance

                            Dsmu_spTmis = np.dot([(Ds_B.p.px + mu_B.p.px) - Dsmu_spar*(sv.x-pv.x)/pvsv_sdistance, (Ds_B.p.py + mu_B.p.py) - Dsmu_spar*(sv.y-pv.y)/pvsv_sdistance, (Ds_B.p.pz + mu_B.p.pz) - Dsmu_spar*(sv.z-pv.z)/pvsv_sdistance], [(Ds_B.p.px + mu_B.p.px) - Dsmu_spar*(sv.x-pv.x)/pvsv_sdistance, (Ds_B.p.py + mu_B.p.py) - Dsmu_spar*(sv.y-pv.y)/pvsv_sdistance, (Ds_B.p.pz + mu_B.p.pz) - Dsmu_spar*(sv.z-pv.z)/pvsv_sdistance])

                            Bs_Dsmu_smcorr = np.sqrt(Bs_Dsmu_sm*Bs_Dsmu_sm + Dsmu_spTmis) + np.sqrt(Dsmu_spTmis)
                            
                            
                            
                            #impact parameter calculation
                            mu_sdiffx = pv.x - pv_mu.x
                            mu_sdiffy = pv.y - pv_mu.y
                            mu_sdiffz = pv.z - pv_mu.z
                            
                            mu_sdiffx2 = sv.x - pv_mu.x
                            mu_sdiffy2 = sv.y - pv_mu.y
                            mu_sdiffz2 = sv.z - pv_mu.z
                            
                            #print mu_sdiffx2
                            
                            #Ds_sdiffx = sv.x - pv_Ds.x
                            #Ds_sdiffy = sv.y - pv_Ds.y
                            #Ds_sdiffz = sv.z - pv_Ds.z
                            
                            Ds_sdiffx = pv.x - pv_Ds.x
                            Ds_sdiffy = pv.y - pv_Ds.y
                            Ds_sdiffz = pv.z - pv_Ds.z
                            
                            Ds_spxdir = Ds_B.p.px/Ds_B.p.absvalue()
                            Ds_spydir = Ds_B.p.py/Ds_B.p.absvalue()
                            Ds_spzdir = Ds_B.p.pz/Ds_B.p.absvalue()
                            
                            mu_spxdir = mu_B.p.px/mu_B.p.absvalue()
                            mu_spydir = mu_B.p.py/mu_B.p.absvalue()
                            mu_spzdir = mu_B.p.pz/mu_B.p.absvalue()                            
                                                        
                            Dsmu_spcp = np.cross([mu_spxdir, mu_spydir, mu_spzdir],[Ds_spxdir, Ds_spydir, Ds_spzdir])
                            
                            #Dsmu_spdp = np.abs(np.dot([Ds_B.p.px - mu_B.p.px, Ds_B.p.py - mu_B.p.py, Ds_B.p.pz - mu_B.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            #Dsmu_spdp = np.abs(np.dot([mu_B.p.px - Ds_B.p.px, mu_B.p.py - Ds_B.p.py, mu_B.p.pz - Ds_B.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            Dsmu_spdp = np.abs(np.dot([pv_mu.x - pv_Ds.x, pv_mu.y - pv_Ds.y, pv_mu.z - pv_Ds.z],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                                           
                            Dsmu_sdist = np.abs(np.dot([Ds_B.p.px - mu_B.p.px, Ds_B.p.py - mu_B.p.py, Ds_B.p.pz - mu_B.p.pz],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))/np.sqrt(np.dot([Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            Dsmu_sDOCA = Dsmu_spdp/np.sqrt(np.dot([Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            
                            #print Dsmu_spdp/np.sqrt(np.dot([Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]],[Dsmu_spcp[0], Dsmu_spcp[1], Dsmu_spcp[2]]))
                            #print mu_B.p.px, mu_B.p.py, mu_B.p.pz, mu_spxdir, mu_spydir, mu_spzdir
                            
                            
                            #mu_sIP3dist = np.sqrt((mu_B.p.py*mu_sdiffz - mu_B.p.pz*mu_sdiffy)**2 - (mu_B.p.px*mu_sdiffz - mu_B.p.pz*mu_sdiffx)**2 + (mu_B.p.px*mu_sdiffy - mu_B.p.py*mu_sdiffx)**2)/np.sqrt(mu_B.p.px**2 + mu_B.p.py**2 + mu_B.p.pz**2)
                            
                            mu_sIPdist = abs(mu_B.p.px*mu_sdiffy - mu_B.p.py*mu_sdiffx)/np.sqrt(mu_B.p.px**2 + mu_B.p.py**2)
                            mu_sIPdist2 = abs(mu_B.p.px*mu_sdiffy2 - mu_B.p.py*mu_sdiffx2)/np.sqrt(mu_B.p.px**2 + mu_B.p.py**2)
                            
                            Ds_sIPdist = abs(Ds_B.p.px*Ds_sdiffy - Ds_B.p.py*Ds_sdiffx)/np.sqrt(Ds_B.p.px**2 + Ds_B.p.py**2)
                            
                            #print sv_mctruth.x, sv_mctruth.y, sv_mctruth.z
                            #print Ds_B.charge, Ds_B.p.px, Ds_B.p.py, Ds_B.p.pz, pv_Ds.x, pv_Ds.y, pv_Ds.z
                            #print mu_B.charge, mu_B.p.px, mu_B.p.py, mu_B.p.pz, pv_mu.x, pv_mu.y, pv_mu.z
                            
                            #============Muon transverse momentum wrt B flight direction=====================================
                            
                            Fd_B = np.subtract([sv.x, sv.y, sv.z],[pv.x, pv.y, pv.z])/pvsv_sdistance
                            mu_sppar_B = np.dot([Fd_B[0],Fd_B[1],Fd_B[2]],[mu_B.p.px,mu_B.p.py,mu_B.p.pz])*Fd_B
                            
                            mu_spT_FB = np.sqrt(np.dot([mu_B.p.px-mu_sppar_B[0],mu_B.p.py-mu_sppar_B[1],mu_B.p.pz-mu_sppar_B[2]],[mu_B.p.px-mu_sppar_B[0],mu_B.p.py-mu_sppar_B[1],mu_B.p.pz-mu_sppar_B[2]]))
                            
                            #print mu_spT_FB
                            
                            #================================================================================================

                            plist_D2K_mc_truth = list([])
                            plist_Dpi_mc_truth = list([])

                            for ptc_gen4 in ptcs:
                                if ptc_gen4.start_vertex == Ds_mctruth.end_vertex:
                                    #Kaons from Ds meson
                                    if abs(ptc_gen4.pdgid) == 321:
                                        plist_D2K_mc_truth.append(ptc_gen4)
                                    #pions from Ds meson
                                    if abs(ptc_gen4.pdgid) == 211:
                                        plist_Dpi_mc_truth.append(ptc_gen4)

                            #print len(plist_D2K_mc_truth), len(plist_Dpi_mc_truth)


                            if len(plist_D2K_mc_truth) == 2 and len(plist_Dpi_mc_truth) == 1:
                            #if (K_mc_truth.start_vertex == Ds_mctruth.end_vertex) and (pi_mc_truth.start_vertex == Ds_mctruth.end_vertex):
                                #self.KKpi_counter += 1

                                #print plist_D2K_mc_truth[0].pdgid, plist_D2K_mc_truth[1].pdgid, plist_Dpi_mc_truth[0].pdgid

                                DsK1_mc_truth, DsK2_mc_truth, Dspi_mc_truth = plist_D2K_mc_truth[0], plist_D2K_mc_truth[1], plist_Dpi_mc_truth[0]
                                Ds_K1, Ds_K2, Ds_pi = copy.deepcopy(DsK1_mc_truth), copy.deepcopy(DsK2_mc_truth), copy.deepcopy(Dspi_mc_truth)
                                                                
                                tv_K1_mctruth = DsK1_mc_truth.start_vertex
                                tv_K2_mctruth = DsK2_mc_truth.start_vertex
                                tv_pi_mctruth = Dspi_mc_truth.start_vertex
                                
                                tv_K1 = Ds_K1.start_vertex
                                tv_K2 = Ds_K2.start_vertex
                                tv_pi = Ds_pi.start_vertex
                                
                                #misidentify kaons as pions
                                Ds_K1_En = np.sqrt(DsK1_mc_truth.p.absvalue()*DsK1_mc_truth.p.absvalue() + DsK1_mc_truth.mass*DsK1_mc_truth.mass)
                                Ds_K2_En = np.sqrt(DsK2_mc_truth.p.absvalue()*DsK2_mc_truth.p.absvalue() + DsK2_mc_truth.mass*DsK2_mc_truth.mass)
                                Ds_pi_En = np.sqrt(Dspi_mc_truth.p.absvalue()*Dspi_mc_truth.p.absvalue() + Dspi_mc_truth.mass*Dspi_mc_truth.mass)

                                KKpi_Ptot = np.dot([DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px, DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py, DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz], [DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px, DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py, DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz])

                                KKpimu_Ptot = np.dot([DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px + mu_px, DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py + mu_py, DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz + mu_pz], [DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px + mu_px, DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py + mu_py, DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz + mu_pz])

                                Ds_KKpi_m = np.sqrt((Ds_K1_En + Ds_K2_En + Ds_pi_En)*(Ds_K1_En + Ds_K2_En + Ds_pi_En)- KKpi_Ptot)

                                Bs_KKpimu_m = np.sqrt((Ds_K1_En + Ds_K2_En + Ds_pi_En + mu_En)*(Ds_K1_En + Ds_K2_En + Ds_pi_En + mu_En)- KKpimu_Ptot)

                                KKpimu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px + mu_px, DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py + mu_py, DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz + mu_pz])/pvsv_distance
                                
                                KKpimu_pTmis = np.dot([(DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px + mu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py + mu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz + mu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance],[(DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px + mu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/pvsv_distance, (DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py + mu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/pvsv_distance, (DsK1_mc_truth.p.pz + DsK2_mc_truth.p.pz + Dspi_mc_truth.p.pz + mu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/pvsv_distance])
                                
                                Bs_KKpimu_mcorr = np.sqrt(Bs_KKpimu_m*Bs_KKpimu_m + KKpimu_pTmis) + np.sqrt(KKpimu_pTmis)

                                Ds_K1_pt = np.sqrt(DsK1_mc_truth.p.px*DsK1_mc_truth.p.px + DsK1_mc_truth.p.py*DsK1_mc_truth.p.py)
                                Ds_K2_pt = np.sqrt(DsK2_mc_truth.p.px*DsK2_mc_truth.p.px + DsK2_mc_truth.p.py*DsK2_mc_truth.p.py)
                                Ds_pi_pt = np.sqrt(Dspi_mc_truth.p.px*Dspi_mc_truth.p.px + Dspi_mc_truth.p.py*Dspi_mc_truth.p.py)
                                
                                Ds_KKpi_pt = np.sqrt((DsK1_mc_truth.p.px + DsK2_mc_truth.p.px + Dspi_mc_truth.p.px)**2 + (DsK1_mc_truth.p.py + DsK2_mc_truth.p.py + Dspi_mc_truth.p.py)**2)

                                K1_resol = momentum_res(Ds_K1_pt)
                                K2_resol = momentum_res(Ds_K2_pt)
                                pi_resol = momentum_res(Ds_pi_pt)
                                
                                KKpi_resol = momentum_res(Ds_KKpi_pt)
                                
                                #print Ds_resol, KKpi_resol
                                #print K_resol, pi1_resol, pi2_resol
                                
                                
                                if self.cfg_ana.smear_tv:
                                    sv_Ds = smear_vertex(sv_Ds, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                    tv_K1 = smear_vertex(tv_K1, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                    tv_K2 = smear_vertex(tv_K2, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                    tv_pi = smear_vertex(tv_pi, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                                
                                if self.cfg_ana.smear_momentum:                
                                    Ds_K1.p = smear_momentum(Ds_K1.p, K1_resol, K1_resol, K1_resol)
                                    Ds_K2.p = smear_momentum(Ds_K2.p, K2_resol, K2_resol, K2_resol)
                                    Ds_pi.p = smear_momentum(Ds_pi.p, pi_resol, pi_resol, pi_resol)
                                    
                                    #Ds_K1.p = smear_momentum(Ds_K1.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
                                    #Ds_K2.p = smear_momentum(Ds_K2.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
                                    #Ds_pi.p = smear_momentum(Ds_pi.p, self.cfg_ana.momentum_x_resolution, self.cfg_ana.momentum_y_resolution, self.cfg_ana.momentum_z_resolution)
                                
                                #print sv_Ds.x, tv_K1.x, tv_K2.x, tv_pi.x
                                
                                #=================================================
                                covmat_K1 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,K1_resol*K1_resol,0.0,0.0], [0.0,0.0,0.0,0.0,K1_resol*K1_resol,0.0],[0.0,0.0,0.0,0.0,0.0,K1_resol*K1_resol]])
                                
                                covmat_K2 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,K2_resol*K2_resol,0.0,0.0], [0.0,0.0,0.0,0.0,K2_resol*K2_resol,0.0],[0.0,0.0,0.0,0.0,0.0,K2_resol*K2_resol]])
                                                                
                                covmat_pi = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,pi_resol*pi_resol,0.0,0.0], [0.0,0.0,0.0,0.0,pi_resol*pi_resol,0.0],[0.0,0.0,0.0,0.0,0.0,pi_resol*pi_resol]])
                                
                                                               
                                #covmat_K1 = np.array([[2.5e-5,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-5,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-5,0.0,0.0,0.0],[0.0,0.0,0.0,K1_resol*K1_resol*0.1,0.0,0.0], [0.0,0.0,0.0,0.0,K1_resol*K1_resol*0.1,0.0],[0.0,0.0,0.0,0.0,0.0,K1_resol*K1_resol*0.1]])
                                
                                #covmat_K2 = np.array([[2.5e-5,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-5,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-5,0.0,0.0,0.0],[0.0,0.0,0.0,K2_resol*K2_resol*0.1,0.0,0.0], [0.0,0.0,0.0,0.0,K2_resol*K2_resol*0.1,0.0],[0.0,0.0,0.0,0.0,0.0,K2_resol*K2_resol*0.1]])
                                                                
                                #covmat_pi = np.array([[2.5e-5,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-5,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-5,0.0,0.0,0.0],[0.0,0.0,0.0,pi_resol*pi_resol*0.1,0.0,0.0], [0.0,0.0,0.0,0.0,pi_resol*pi_resol*0.1,0.0],[0.0,0.0,0.0,0.0,0.0,pi_resol*pi_resol*0.1]])
                                
                                #Ds_vtx.setDaughter(0,  Ds_K1.charge, np.array([Ds_K1.p.px, Ds_K1.p.py, Ds_K1.p.pz]), np.array([tv_K1.x, tv_K1.y, tv_K1.z]), covmat_K1)
                                #Ds_vtx.setDaughter(1,  Ds_K2.charge, np.array([Ds_K2.p.px, Ds_K2.p.py, Ds_K2.p.pz]), np.array([tv_K2.x, tv_K2.y, tv_K2.z]), covmat_K2)
                                #Ds_vtx.setDaughter(2,  Ds_pi.charge, np.array([Ds_pi.p.px, Ds_pi.p.py, Ds_pi.p.pz]), np.array([tv_pi.x, tv_pi.y, tv_pi.z]), covmat_pi)
                                
                                Ds_vtx.setDaughter(0,  Ds_K1.charge, np.array([Ds_K1.p.px, Ds_K1.p.py, Ds_K1.p.pz]), 0.1*np.array([tv_K1.x, tv_K1.y, tv_K1.z]), covmat_K1)
                                Ds_vtx.setDaughter(1,  Ds_K2.charge, np.array([Ds_K2.p.px, Ds_K2.p.py, Ds_K2.p.pz]), 0.1*np.array([tv_K2.x, tv_K2.y, tv_K2.z]), covmat_K2)
                                Ds_vtx.setDaughter(2,  Ds_pi.charge, np.array([Ds_pi.p.px, Ds_pi.p.py, Ds_pi.p.pz]), 0.1*np.array([tv_pi.x, tv_pi.y, tv_pi.z]), covmat_pi)
                                #Ds_vtx.setDaughter(0,  Ds_K1.charge, np.array([Ds_K1.p.px, Ds_K1.p.py, Ds_K1.p.pz]), np.array([sv_Ds.x, sv_Ds.y, sv_Ds.z]), covmat_K1)
                                #Ds_vtx.setDaughter(1,  Ds_K2.charge, np.array([Ds_K2.p.px, Ds_K2.p.py, Ds_K2.p.pz]), np.array([sv_Ds.x, sv_Ds.y, sv_Ds.z]), covmat_K2)
                                #Ds_vtx.setDaughter(2,  Ds_pi.charge, np.array([Ds_pi.p.px, Ds_pi.p.py, Ds_pi.p.pz]), np.array([sv_Ds.x, sv_Ds.y, sv_Ds.z]), covmat_pi)
    
                                Ds_KKpi_fit = Ds_vtx.fit(100)
                                
                                #Ds_fitvtx = Ds_vtx.getVertex()
                                Ds_fitvtx = 10.0*Ds_vtx.getVertex() #convert back to mm
                                
                                Ds_fitvtx_diffx = sv_Ds.x - Ds_fitvtx[0]
                                Ds_fitvtx_diffy = sv_Ds.y - Ds_fitvtx[1]
                                Ds_fitvtx_diffz = sv_Ds.z - Ds_fitvtx[2]
                                
                                Ds_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Ds_vtx.getChi2(), Ds_vtx.getNDF())
                                
                                #print pi_resol*pi_resol;
                                
                                #print sv_Ds_mctruth, sv_Ds, Ds_fitvtx*10, np.subtract([sv_Ds_mctruth.x, sv_Ds_mctruth.y, sv_Ds_mctruth.z],[sv_Ds.x, sv_Ds.y, sv_Ds.z]),np.subtract([Ds_fitvtx[0]*10, Ds_fitvtx[1]*10, Ds_fitvtx[2]*10],[sv_Ds.x, sv_Ds.y, sv_Ds.z]), Ds_vtx.getChi2() 
                                
                                #print (1 - scipy.stats.chi2.cdf(Ds_vtx.getChi2(), Ds_vtx.getNDF()))
                                
                                #print Ds_vtx.getChi2()
                                #print Ds_vtx.getVariance()
                                #print "Smeared Ds vertex", sv_Ds.x, sv_Ds.y, sv_Ds.z 
                                #print "Fitted Ds vertex", Ds_fitvtx[0], Ds_fitvtx[1], Ds_fitvtx[2]
                                #print covmat_pi
                                                            
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
                                
                                mu_ptot = np.array([mu_B.p.px, mu_B.p.py, mu_B.p.pz])
                                mu_pv = np.array([pv_mu.x, pv_mu.y, pv_mu.z])
                                
                                Ds_vtxfit_charge = Ds_K1.charge + Ds_K2.charge + Ds_pi.charge
                                
                                #============================================================================================================
                                #Calculate point of closest approach (DCA) along between Ds and mu
                                
                                N_Dsmu = (np.cross([Ds_KKpi_fitp[0], Ds_KKpi_fitp[1], Ds_KKpi_fitp[2]],[mu_B.p.px, mu_B.p.py, mu_B.p.pz]))/(Ds_KKpi_fitpmag*mu_B.p.absvalue())
                                N_Dsmu_mag = np.sqrt(np.dot([N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]]))
                                
                                pv_Ds_PCA = Ds_fitvtx + (Ds_KKpi_fitp/np.sqrt(Ds_KKpi_fitp[0]**2 + Ds_KKpi_fitp[1]**2 + Ds_KKpi_fitp[2]**2))*Ds_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,mu_B)
                                
                                pv_mu_PCA = mu_pv + (mu_ptot/np.sqrt(mu_B.p.px**2 + mu_B.p.py**2 + mu_B.p.pz**2))*mu_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,mu_B)
                                
                                #Calculate distance of closest approach (PCA) of Ds and mu
                                
                                Dsmu_DCA = np.dot([(pv_Ds_PCA[0]-pv_mu_PCA[0]),(pv_Ds_PCA[1]-pv_mu_PCA[1]),(pv_Ds_PCA[2]-pv_mu_PCA[2])],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]])/N_Dsmu_mag
                                
                                #print Dsmu_DCA
                                #print pv_Ds_PCA[0], pv_Ds.x, pv_Ds_PCA[1], pv_Ds.y, pv_Ds_PCA[2], pv_Ds.z
                                #print pv_mu_PCA[0], pv_mu.x
                                #print sv_Dsfit[0], sv_Dsfit[1], sv_Dsfit[2], pv_Ds.x, pv_Ds.y, pv_Ds.z
                                #==========================================================================================================
                                
                                covmat_mu = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0,4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,mu_resol*mu_resol,0.0,0.0], [0.0,0.0,0.0,0.0,mu_resol*mu_resol,0.0],[0.0,0.0,0.0,0.0,0.0,mu_resol*mu_resol]])
                                
                                #covmat_Ds = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0, 4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Ds_resol*Ds_resol,0.0,0.0], [0.0,0.0,0.0,0.0,Ds_resol*Ds_resol,0.0],[0.0,0.0,0.0,0.0,0.0,Ds_resol*Ds_resol]])
                                
                                #covmat_mu = np.array([[4.9e-5,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-5,0.0,0.0,0.0,0.0],[0.0,0.0,4.9e-5,0.0,0.0,0.0],[0.0,0.0,0.0,mu_resol*mu_resol,0.0,0.0], [0.0,0.0,0.0,0.0,mu_resol*mu_resol,0.0],[0.0,0.0,0.0,0.0,0.0,mu_resol*mu_resol]])
                                
                                covmat_Ds = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0, 4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,KKpi_resol*KKpi_resol,0.0,0.0], [0.0,0.0,0.0,0.0,KKpi_resol*KKpi_resol,0.0],[0.0,0.0,0.0,0.0,0.0,KKpi_resol*KKpi_resol]])
                                
                                #Bs_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([pv_Ds_PCA[0], pv_Ds_PCA[1],pv_Ds_PCA[2]]),Ds_vtx.getVariance())
                                
                                Bs_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([pv_Ds_PCA[0], pv_Ds_PCA[1],pv_Ds_PCA[2]]),covmat_Ds)
                                
                                #Bs_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([pv_Dsfit[0], pv_Dsfit[1],pv_Dsfit[2]]),Ds_vtx.getVariance())
                                
                                #Bs_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([Ds_fitvtx[0], Ds_fitvtx[1],Ds_fitvtx[2]]),Ds_vtx.getVariance())
                                
                                #Bs_Dsmu_vtx.setDaughter(0, Ds_B.charge, np.array([Ds_B.p.px, Ds_B.p.py, Ds_B.p.pz]), 0.1*np.array([pv_Ds.x, pv_Ds.y,pv_Ds.z]),covmat_Ds)
                                                                                                
                                Bs_Dsmu_vtx.setDaughter(1, mu_B.charge, np.array([mu_B.p.px, mu_B.p.py, mu_B.p.pz]), 0.1*np.array([pv_mu.x, pv_mu.y, pv_mu.z]), covmat_mu)
                                
                                #Bs_Dsmu_vtx.setDaughter(1, mu_B.charge, np.array([mu_B.p.px, mu_B.p.py, mu_B.p.pz]), 0.1*np.array([pv_mu_PCA[0], pv_mu_PCA[1],pv_mu_PCA[2]]), covmat_mu)
                                
                                Bs_Dsmu_vtx.fit(100)
                                Bs_Dsmu_fitvtx = 10*Bs_Dsmu_vtx.getVertex()
                                Bs_Dsmu_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Bs_Dsmu_vtx.getChi2(), Bs_Dsmu_vtx.getNDF())
                                
                                Bs_Dsmu_fitvtx_diff = np.subtract([Bs_Dsmu_fitvtx[0], Bs_Dsmu_fitvtx[1], Bs_Dsmu_fitvtx[2]], [sv.x, sv.y,sv.z])
                                
                                
                                #print Bs_Dsmu_vtx.getVariance()
                                #print "Smeared Bs vertex", sv.x, sv.y, sv.z 
                                #print "Fitted Bs vertex", Bs_Dsmu_fitvtx[0], Bs_Dsmu_fitvtx[1], Bs_Dsmu_fitvtx[2]
                                
                                #Bs_Dsmu_fitvtx_diffmc = np.subtract([Bs_Dsmu_fitvtx[0], Bs_Dsmu_fitvtx[1], Bs_Dsmu_fitvtx[2]], [sv_mctruth.x, sv_mctruth.y,sv_mctruth.z])
                                
                                #print Bs_Dsmu_fitvtx_diff, Bs_Dsmu_fitvtx[0], sv_mctruth.x, Bs_Dsmu_fitvtx_CDF
                                
                                #if Bs_Dsmu_vtx.getChi2() > 10:                                
                                #    print Bs_Dsmu_fitvtx_diff, Bs_Dsmu_fitvtx[0], sv_mctruth.x, Bs_Dsmu_fitvtx_CDF
                                
                                
                                #smeared momentum and energy
                                Ds_K1_sEn = np.sqrt(Ds_K1.p.absvalue()*Ds_K1.p.absvalue() + Ds_K1.mass*Ds_K1.mass)
                                #Ds_K1_sEn = np.sqrt(Ds_K1.p.absvalue()*Ds_K1.p.absvalue() + Ds_pi.mass*Ds_pi.mass)

                                #misidentify kaons as pions
                                Ds_K2_sEn = np.sqrt(Ds_K2.p.absvalue()*Ds_K2.p.absvalue() + Ds_K2.mass*Ds_K2.mass)
                                Ds_pi_sEn = np.sqrt(Ds_pi.p.absvalue()*Ds_pi.p.absvalue() + Ds_pi.mass*Ds_pi.mass)

                                KKpi_sPtot = np.dot([Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz], [Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz])

                                KKpi_sm = np.sqrt((Ds_K1_sEn + Ds_K2_sEn + Ds_pi_sEn)*(Ds_K1_sEn + Ds_K2_sEn + Ds_pi_sEn)- KKpi_sPtot)

                                KKpimu_sPtot = np.dot([Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz], [Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz])

                                Bs_KKpimu_sm = np.sqrt((Ds_K1_sEn + Ds_K2_sEn + Ds_pi_sEn +  mu_B_sEn)*(Ds_K1_sEn + Ds_K2_sEn + Ds_pi_sEn +  mu_B_sEn)- KKpimu_sPtot)
                                
                                KKpimu_spar = np.dot([(sv.x-pv.x), (sv.y-pv.y), (sv.z-pv.z)],[Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz])/pvsv_sdistance
                                
                                KKpimu_spTmis = np.dot([(Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px) - KKpimu_spar*(sv.x-pv.x)/pvsv_sdistance, (Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py) - KKpimu_spar*(sv.y-pv.y)/pvsv_sdistance, (Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz) - KKpimu_spar*(sv.z-pv.z)/pvsv_sdistance],[(Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px) - KKpimu_spar*(sv.x-pv.x)/pvsv_sdistance, (Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py) - KKpimu_spar*(sv.y-pv.y)/pvsv_sdistance, (Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz) - KKpimu_spar*(sv.z-pv.z)/pvsv_sdistance])
                                
                                Bs_KKpimu_smcorr = np.sqrt(Bs_KKpimu_sm*Bs_KKpimu_sm + KKpimu_spTmis) + np.sqrt(KKpimu_spTmis)
                                
                                #Using fitted vertex
                                #pvsv_fitdist = np.sqrt((Bs_fitvtx[0] - pv.x)**2 + (Bs_fitvtx[1] - pv.y)**2 + (Bs_fitvtx[2] - pv.z)**2)
                                
                                #KKpimu_fitpar = np.dot([(Bs_fitvtx[0]-pv.x), (Bs_fitvtx[1]-pv.y), (Bs_fitvtx[2]-pv.z)],[Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px, Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py, Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz])/pvsv_fitdist
                                
                                #KKpimu_fitpTmis = np.dot([(Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px) - KKpimu_fitpar*(Bs_fitvtx[0]-pv.x)/pvsv_fitdist, (Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py) - KKpimu_fitpar*(Bs_fitvtx[1]-pv.y)/pvsv_fitdist, (Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz) - KKpimu_fitpar*(Bs_fitvtx[2]-pv.z)/pvsv_fitdist],[(Ds_K1.p.px + Ds_K2.p.px + Ds_pi.p.px + mu_B.p.px) - KKpimu_fitpar*(Bs_fitvtx[0]-pv.x)/pvsv_fitdist, (Ds_K1.p.py + Ds_K2.p.py + Ds_pi.p.py + mu_B.p.py) - KKpimu_fitpar*(Bs_fitvtx[1]-pv.y)/pvsv_fitdist, (Ds_K1.p.pz + Ds_K2.p.pz + Ds_pi.p.pz + mu_B.p.pz) - KKpimu_fitpar*(Bs_fitvtx[2]-pv.z)/pvsv_fitdist])
                                
                                #Bs_KKpimu_fitmcorr = np.sqrt(Bs_KKpimu_sm*Bs_KKpimu_sm + KKpimu_fitpTmis) + np.sqrt(KKpimu_fitpTmis)
                                

                                #Thrust_p = np.sqrt(event_info.at(0).Px()*event_info.at(0).Px() + event_info.at(0).Py()*event_info.at(0).Py() + event_info.at(0).Pz()*event_info.at(0).Pz())
                                
                                self.mc_truth_tree.fill('event_number',     event_number)
                                self.mc_truth_tree.fill('n_particles',      n_particles)
                                self.mc_truth_tree.fill('pv_x',             pv_mctruth.x)
                                self.mc_truth_tree.fill('pv_y',             pv_mctruth.y)
                                self.mc_truth_tree.fill('pv_z',             pv_mctruth.z)
                                self.mc_truth_tree.fill('sv_x',             sv_mctruth.x)
                                self.mc_truth_tree.fill('sv_y',             sv_mctruth.y)
                                self.mc_truth_tree.fill('sv_z',             sv_mctruth.z)
                                self.mc_truth_tree.fill('tv_K1_x',          tv_K1_mctruth.x)
                                self.mc_truth_tree.fill('tv_K1_y',          tv_K1_mctruth.y)
                                self.mc_truth_tree.fill('tv_K1_z',          tv_K1_mctruth.z)
                                self.mc_truth_tree.fill('tv_K2_x',          tv_K2_mctruth.x)
                                self.mc_truth_tree.fill('tv_K2_y',          tv_K2_mctruth.y)
                                self.mc_truth_tree.fill('tv_K2_z',          tv_K2_mctruth.z)
                                self.mc_truth_tree.fill('tv_pi_x',          tv_pi_mctruth.x)
                                self.mc_truth_tree.fill('tv_pi_y',          tv_pi_mctruth.y)
                                self.mc_truth_tree.fill('tv_pi_z',          tv_pi_mctruth.z)
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
                                self.mc_truth_tree.fill('B_px',             B_mc_truth.p.px)
                                self.mc_truth_tree.fill('B_py',             B_mc_truth.p.py)
                                self.mc_truth_tree.fill('B_pz',             B_mc_truth.p.pz)
                                self.mc_truth_tree.fill('B_p',              pb)
                                self.mc_truth_tree.fill('B_m',              B_mc_truth.mass)
                                self.mc_truth_tree.fill('B_ID',             B_mc_truth.pdgid)
                                self.mc_truth_tree.fill('B_q',              B_mc_truth.charge)
                                self.mc_truth_tree.fill('Ds_px',            Ds_mctruth.p.px)
                                self.mc_truth_tree.fill('Ds_py',            Ds_mctruth.p.py)
                                self.mc_truth_tree.fill('Ds_pz',            Ds_mctruth.p.pz)
                                self.mc_truth_tree.fill('Ds_p',             Ds_mctruth.p.absvalue())
                                self.mc_truth_tree.fill('Ds_pT',            Ds_pT)
                                self.mc_truth_tree.fill('Ds_E',             Ds_En)
                                self.mc_truth_tree.fill('D_m',              Ds_mctruth.mass)
                                self.mc_truth_tree.fill('Ds_ID',            Ds_mctruth.pdgid)
                                self.mc_truth_tree.fill('Ds_q',             Ds_mctruth.charge)
                                self.mc_truth_tree.fill('mu_px',            mu_px)
                                self.mc_truth_tree.fill('mu_py',            mu_py)
                                self.mc_truth_tree.fill('mu_pz',            mu_pz)
                                self.mc_truth_tree.fill('mu_E',             mu_En)
                                self.mc_truth_tree.fill('mu_p',             mu_p)
                                self.mc_truth_tree.fill('mu_pT',            mu_pT)
                                self.mc_truth_tree.fill('mu_ET',            mu_ET)
                                self.mc_truth_tree.fill('mu_m',             plist_Bmu_mc_truth[0].mass)
                                self.mc_truth_tree.fill('mu_ID',            plist_Bmu_mc_truth[0].pdgid)
                                self.mc_truth_tree.fill('mu_q',             plist_Bmu_mc_truth[0].charge)
                                self.mc_truth_tree.fill('Dsmu_pTmis',       np.sqrt(Dsmu_pTmis))
                                self.mc_truth_tree.fill('Dsmu_par',         Dsmu_par)
                                self.mc_truth_tree.fill('Bs_Dsmu_m',        Bs_Dsmu_m)
                                self.mc_truth_tree.fill('Bs_Dsmu_mcorr',    Bs_Dsmu_mcorr)
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
                                self.mc_truth_tree.fill('mu_IPdist2',       mu_IPdist2)
                                #self.mc_truth_tree.fill('mu_IP3dist',      mu_IP3dist)
                                self.mc_truth_tree.fill('Ds_IPdist',        Ds_IPdist)
                                

                                #filling MC truth information
                                self.mc_truth_tree.tree.Fill()


                                self.mc_truth_KKpi.fill('Ds_KKpi_m',        Ds_KKpi_m)
                                self.mc_truth_KKpi.fill('Bs_KKpimu_m',      Bs_KKpimu_m)
                                self.mc_truth_KKpi.fill('Bs_KKpimu_mcorr',  Bs_KKpimu_mcorr)
                                self.mc_truth_KKpi.fill('KKpimu_pTmis',     np.sqrt(KKpimu_pTmis))
                                self.mc_truth_KKpi.fill('KKpimu_par',       KKpimu_par)
                                self.mc_truth_KKpi.fill('Ds_K1_px',         DsK1_mc_truth.p.px)
                                self.mc_truth_KKpi.fill('Ds_K1_py',         DsK1_mc_truth.p.py)
                                self.mc_truth_KKpi.fill('Ds_K1_pz',         DsK1_mc_truth.p.pz)
                                self.mc_truth_KKpi.fill('Ds_K1_E',          Ds_K1_En)
                                self.mc_truth_KKpi.fill('Ds_K1_p',          DsK1_mc_truth.p.absvalue())
                                self.mc_truth_KKpi.fill('Ds_K1_m',          DsK1_mc_truth.mass)
                                self.mc_truth_KKpi.fill('Ds_K1_ID',         DsK1_mc_truth.pdgid)
                                self.mc_truth_KKpi.fill('Ds_K1_q',          DsK1_mc_truth.charge)
                                self.mc_truth_KKpi.fill('Ds_K2_px',         DsK2_mc_truth.p.px)
                                self.mc_truth_KKpi.fill('Ds_K2_py',         DsK2_mc_truth.p.py)
                                self.mc_truth_KKpi.fill('Ds_K2_pz',         DsK2_mc_truth.p.pz)
                                self.mc_truth_KKpi.fill('Ds_K2_E',          Ds_K2_En)
                                self.mc_truth_KKpi.fill('Ds_K2_p',          DsK2_mc_truth.p.absvalue())
                                self.mc_truth_KKpi.fill('Ds_K2_m',          DsK2_mc_truth.mass)
                                self.mc_truth_KKpi.fill('Ds_K2_ID',         DsK2_mc_truth.pdgid)
                                self.mc_truth_KKpi.fill('Ds_K2_q',          DsK2_mc_truth.charge)
                                self.mc_truth_KKpi.fill('Ds_pi_px',         Dspi_mc_truth.p.px)
                                self.mc_truth_KKpi.fill('Ds_pi_py',         Dspi_mc_truth.p.py)
                                self.mc_truth_KKpi.fill('Ds_pi_pz',         Dspi_mc_truth.p.pz)
                                self.mc_truth_KKpi.fill('Ds_pi_E',          Ds_pi_En)
                                self.mc_truth_KKpi.fill('Ds_pi_p',          Dspi_mc_truth.p.absvalue())
                                self.mc_truth_KKpi.fill('Ds_pi_m',          Dspi_mc_truth.mass)
                                self.mc_truth_KKpi.fill('Ds_pi_ID',         Dspi_mc_truth.pdgid)
                                self.mc_truth_KKpi.fill('Ds_pi_q',          Dspi_mc_truth.charge)                               

                                #filling MC truth information
                                self.mc_truth_KKpi.tree.Fill()
                                
                                self.smeared_tree.fill('event_number',      event_number)
                                self.smeared_tree.fill('n_particles',       n_particles)
                                self.smeared_tree.fill('Bs_mu_sp',          Bs_mu_sp)                                
                                self.smeared_tree.fill('Bs_Ds_spx',         Ds_B.p.px)
                                self.smeared_tree.fill('Bs_Ds_spy',         Ds_B.p.py)
                                self.smeared_tree.fill('Bs_Ds_spz',         Ds_B.p.pz)
                                self.smeared_tree.fill('Bs_Ds_sp',          Ds_B.p.absvalue())
                                self.smeared_tree.fill('Bs_Ds_spT',         Bs_Ds_spT)
                                self.smeared_tree.fill('Bs_mu_spx',         mu_B.p.px)
                                self.smeared_tree.fill('Bs_mu_spy',         mu_B.p.py)
                                self.smeared_tree.fill('Bs_mu_spz',         mu_B.p.pz)
                                self.smeared_tree.fill('Bs_mu_spT',         Bs_mu_spT)
                                self.smeared_tree.fill('Ds_K1_spx',         Ds_K1.p.px)
                                self.smeared_tree.fill('Ds_K1_spy',         Ds_K1.p.py)
                                self.smeared_tree.fill('Ds_K1_spz',         Ds_K1.p.pz)
                                self.smeared_tree.fill('Ds_K1_sp',          Ds_K1.p.absvalue())
                                self.smeared_tree.fill('Ds_K2_spx',         Ds_K2.p.px)
                                self.smeared_tree.fill('Ds_K2_spy',         Ds_K2.p.py)
                                self.smeared_tree.fill('Ds_K2_spz',         Ds_K2.p.pz)
                                self.smeared_tree.fill('Ds_K2_sp',          Ds_K2.p.absvalue())
                                self.smeared_tree.fill('Ds_pi_spx',         Ds_pi.p.px)
                                self.smeared_tree.fill('Ds_pi_spy',         Ds_pi.p.py)
                                self.smeared_tree.fill('Ds_pi_spz',         Ds_pi.p.pz)
                                self.smeared_tree.fill('Ds_pi_sp',          Ds_pi.p.absvalue())
                                self.smeared_tree.fill('Bs_Dsmu_sm',        Bs_Dsmu_sm)
                                self.smeared_tree.fill('Bs_Dsmu_smcorr',    Bs_Dsmu_smcorr)
                                self.smeared_tree.fill('Dsmu_spTmis',       np.sqrt(Dsmu_spTmis))
                                self.smeared_tree.fill('Dsmu_spar',         Dsmu_spar)
                                self.smeared_tree.fill('KKpi_sM',           KKpi_sm)
                                self.smeared_tree.fill('KKpimu_sM',         Bs_KKpimu_sm)
                                self.smeared_tree.fill('KKpimu_spTmis',     np.sqrt(KKpimu_spTmis))
                                self.smeared_tree.fill('KKpimu_spar',       KKpimu_spar)
                                self.smeared_tree.fill('Bs_KKpimu_smcorr',  Bs_KKpimu_smcorr)
                                self.smeared_tree.fill('Pvis_SS_sp',        Pvis_SS_sp)
                                self.smeared_tree.fill('Pvis_SS_sE',        Pvis_SS_sE)
                                self.smeared_tree.fill('Pvis_OS_sp',        Pvis_OS_sp)
                                self.smeared_tree.fill('Pvis_OS_sE',        Pvis_OS_sE)
                                self.smeared_tree.fill('mu_sIPdist',        mu_sIPdist)
                                self.smeared_tree.fill('mu_sIPdist2',       mu_sIPdist2)
                                #self.smeared_tree.fill('mu_sIP3dist',      mu_sIP3dist)
                                self.smeared_tree.fill('Ds_sIPdist',        Ds_sIPdist)
                                self.smeared_tree.fill('Dsmu_sDOCA',        Dsmu_sDOCA)                                
                                self.smeared_tree.fill('Ds_fitvtx_Chi2',    Ds_vtx.getChi2())
                                self.smeared_tree.fill('Ds_fitvtx_NDF',     Ds_vtx.getNDF())        
                                self.smeared_tree.fill('Ds_fitvtx_x',       Ds_fitvtx[0])
                                self.smeared_tree.fill('Ds_fitvtx_y',       Ds_fitvtx[1])
                                self.smeared_tree.fill('Ds_fitvtx_z',       Ds_fitvtx[2])
                                self.smeared_tree.fill('Ds_K1_fitpx',       Ds_K1_fitp[0])
                                self.smeared_tree.fill('Ds_K1_fitpy',       Ds_K1_fitp[1])
                                self.smeared_tree.fill('Ds_K1_fitpz',       Ds_K1_fitp[2])
                                self.smeared_tree.fill('Ds_K1_fitptot',     Ds_K1_fitptot)
                                self.smeared_tree.fill('Ds_K1_fitpT',       Ds_K1_fitpT)
                                self.smeared_tree.fill('Ds_K2_fitpx',       Ds_K2_fitp[0])
                                self.smeared_tree.fill('Ds_K2_fitpy',       Ds_K2_fitp[1])
                                self.smeared_tree.fill('Ds_K2_fitpz',       Ds_K2_fitp[2])
                                self.smeared_tree.fill('Ds_K2_fitptot',     Ds_K2_fitptot)
                                self.smeared_tree.fill('Ds_K2_fitpT',       Ds_K2_fitpT)
                                self.smeared_tree.fill('Ds_pi_fitpx',       Ds_pi_fitp[0])
                                self.smeared_tree.fill('Ds_pi_fitpy',       Ds_pi_fitp[1])
                                self.smeared_tree.fill('Ds_pi_fitpz',       Ds_pi_fitp[2])
                                self.smeared_tree.fill('Ds_pi_fitptot',     Ds_pi_fitptot)
                                self.smeared_tree.fill('Ds_pi_fitpT',       Ds_pi_fitpT)
                                #self.smeared_tree.fill('pvsv_fitdist',      pvsv_fitdist)
                                #self.smeared_tree.fill('Bs_KKpimu_fitmcorr',Bs_KKpimu_fitmcorr)
                                
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
                                self.smeared_vtx.fill('tv_K1_sx',           tv_K1.x)
                                self.smeared_vtx.fill('tv_K1_sy',           tv_K1.y)
                                self.smeared_vtx.fill('tv_K1_sz',           tv_K1.z)
                                self.smeared_vtx.fill('tv_K1_diffx',        tv_K1.x-tv_K1_mctruth.x)
                                self.smeared_vtx.fill('tv_K1_diffy',        tv_K1.y-tv_K1_mctruth.y)
                                self.smeared_vtx.fill('tv_K1_diffz',        tv_K1.z-tv_K1_mctruth.z)
                                self.smeared_vtx.fill('tv_K2_sx',           tv_K2.x)
                                self.smeared_vtx.fill('tv_K2_sy',           tv_K2.y)
                                self.smeared_vtx.fill('tv_K2_sz',           tv_K2.z)
                                self.smeared_vtx.fill('tv_pi_sx',           tv_pi.x)
                                self.smeared_vtx.fill('tv_pi_sy',           tv_pi.y)
                                self.smeared_vtx.fill('tv_pi_sz',           tv_pi.z)
                                self.smeared_vtx.fill('pvsv_sdistance',     pvsv_sdistance)
                                self.smeared_vtx.fill('B_KKpimu_sm',        Bs_KKpimu_sm)
                                self.smeared_vtx.fill('B_KKpimu_smcorr',    Bs_KKpimu_smcorr)                                
                                self.smeared_vtx.fill('mu_spx',             mu_B.p.px)
                                self.smeared_vtx.fill('mu_spy',             mu_B.p.py)
                                self.smeared_vtx.fill('mu_spz',             mu_B.p.pz)
                                self.smeared_vtx.fill('mu_sp',              mu_B.p.absvalue())
                                self.smeared_vtx.fill('mu_spT',             Bs_mu_spT)
                                self.smeared_vtx.fill('mu_spT_B',           mu_spT_FB)
                                self.smeared_vtx.fill('mu_sIPdist',         mu_sIPdist)
                                self.smeared_vtx.fill('mu_sIPdist2',        mu_sIPdist2)
                                self.smeared_vtx.fill('pvDs_sx',            pv_Ds.x)
                                self.smeared_vtx.fill('pvDs_sy',            pv_Ds.y)
                                self.smeared_vtx.fill('pvDs_sz',            pv_Ds.z)
                                self.smeared_vtx.fill('svDs_sx',            sv_Ds.x)
                                self.smeared_vtx.fill('svDs_sy',            sv_Ds.y)
                                self.smeared_vtx.fill('svDs_sz',            sv_Ds.z)
                                self.smeared_vtx.fill('Ds_Chi2',            Ds_vtx.getChi2()/Ds_vtx.getNDF())
                                self.smeared_vtx.fill('Ds_NDF',             Ds_vtx.getNDF())
                                self.smeared_vtx.fill('Ds_CDF',             Ds_fitvtx_CDF)
                                self.smeared_vtx.fill('Ds_fitvtx_x',        Ds_fitvtx[0])
                                self.smeared_vtx.fill('Ds_fitvtx_y',        Ds_fitvtx[1])
                                self.smeared_vtx.fill('Ds_fitvtx_z',        Ds_fitvtx[2])
                                self.smeared_vtx.fill('Ds_fitvtx_diffx',    Ds_fitvtx_diffx)
                                self.smeared_vtx.fill('Ds_fitvtx_diffy',    Ds_fitvtx_diffy)
                                self.smeared_vtx.fill('Ds_fitvtx_diffz',    Ds_fitvtx_diffz)                                
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_Chi2', Bs_Dsmu_vtx.getChi2()/Bs_Dsmu_vtx.getNDF())
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_NDF',  Bs_Dsmu_vtx.getNDF())
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_CDF',  Bs_Dsmu_fitvtx_CDF)
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_x',    Bs_Dsmu_fitvtx[0])
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_y',    Bs_Dsmu_fitvtx[1])
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_z',    Bs_Dsmu_fitvtx[2])
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_diffx',Bs_Dsmu_fitvtx_diff[0])
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_diffy',Bs_Dsmu_fitvtx_diff[1])
                                self.smeared_vtx.fill('B_Dsmu_fitvtx_diffz',Bs_Dsmu_fitvtx_diff[2])
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


                                # filling event information
                                #self.tree.fill('event_number',              event_number)
                                #self.tree.fill('n_particles',               n_particles)
                                #self.tree.fill('pv_x',                      pv.x)
                                #self.tree.fill('pv_y',                      pv.y)
                                #self.tree.fill('pv_z',                      pv.z)
                                
                                #self.tree.tree.Fill()

        
            # fill here only if successfully found all particles
            #if (B_mc_truth != None and
            #    Ds_mctruth != None and
            #	DsK1_mc_truth != None and
            #	DsK2_mc_truth != None and
            # 	Dspi_mc_truth != None and
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
        print('Ds(+/-) meson events: {}'.format(self.Ds_counter))
        print('Muons from B0s: {}'.format(self.Bmu_counter))
        print('Muons from tau: {}'.format(self.taumu_counter))

        print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
        print('Efficiency:\n\tMomentum of B cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter)))
        raw_input('Press ENTER when finished')
