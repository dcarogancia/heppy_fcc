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

class LbLcXpKPiAnalyzer(Analyzer):
    def beginLoop(self, setup):
        self.start_time = time.time()
        self.last_timestamp = time.time()

        self.counter = 0 # Total number of processed decays
        self.pb_counter = 0 # Number of events with B momentum > 25 GeV
        self.BdDs_counter = 0
        self.L0b_mu_counter = 0
        self.Lc_counter = 0
        self.Lcmu_counter = 0                   
        self.mu_counter = 0                
        
        gROOT.ProcessLine('.x ' + self.cfg_ana.stylepath) # nice looking plots

        # histograms to visualize cuts
        self.pb_hist = TH1F('pb_hist', 'P_{B}', 500, 0, 50)

        super(LbLcXpKPiAnalyzer, self).beginLoop(setup)
        
        self.rootfile = TFile('/'.join([self.dirName, 'Lb_Lcmu_pKpi_bkg_evtpdl2019-1M.root']), 'recreate')

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
        self.mc_truth_tree.var('pv_Ds_x')
        self.mc_truth_tree.var('pv_Ds_y')
        self.mc_truth_tree.var('pv_Ds_z')
        self.mc_truth_tree.var('sv_Ds_x')
        self.mc_truth_tree.var('sv_Ds_y')
        self.mc_truth_tree.var('sv_Ds_z')
        self.mc_truth_tree.var('pvsv_distance')
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
        self.mc_truth_tree.var('L0b_Lc_En')
        self.mc_truth_tree.var('L0b_Lc_px')                            
        self.mc_truth_tree.var('L0b_Lc_py')
        self.mc_truth_tree.var('L0b_Lc_pz')
        self.mc_truth_tree.var('L0b_Lc_p')
        self.mc_truth_tree.var('Lc_proton_En')
        self.mc_truth_tree.var('Lc_proton_px')
        self.mc_truth_tree.var('Lc_proton_py')
        self.mc_truth_tree.var('Lc_proton_pz')
        self.mc_truth_tree.var('Lc_proton_p')
        self.mc_truth_tree.var('Lc_K_En')
        self.mc_truth_tree.var('Lc_K_px')
        self.mc_truth_tree.var('Lc_K_py')
        self.mc_truth_tree.var('Lc_K_pz')
        self.mc_truth_tree.var('Lc_K_p')
        self.mc_truth_tree.var('Lc_pi_En')
        self.mc_truth_tree.var('Lc_pi_px')
        self.mc_truth_tree.var('Lc_pi_py')
        self.mc_truth_tree.var('Lc_pi_pz')
        self.mc_truth_tree.var('Lc_pi_p')
        self.mc_truth_tree.var('L0b_mu_En')
        self.mc_truth_tree.var('L0b_mu_px')
        self.mc_truth_tree.var('L0b_mu_py')
        self.mc_truth_tree.var('L0b_mu_pz')
        self.mc_truth_tree.var('L0b_mu_p')    	
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
    	self.mc_truth_tree.var('Ds_KKpi_mass')    	
        self.mc_truth_tree.var('KKpimu_pTmis')
        self.mc_truth_tree.var('KKpimu_par')
    	self.mc_truth_tree.var('B_KKpimu_mass')
    	self.mc_truth_tree.var('B_KKpimu_mcorr')
    	self.mc_truth_tree.var('mu_IPdist')
    	
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
        self.smeared_tree.var('pv_Ds_sx')
        self.smeared_tree.var('pv_Ds_sy')
        self.smeared_tree.var('pv_Ds_sz')
        self.smeared_tree.var('sv_Ds_sx')
        self.smeared_tree.var('sv_Ds_sy')
        self.smeared_tree.var('sv_Ds_sz')
        self.smeared_tree.var('pv_sdiffx')
        self.smeared_tree.var('pv_sdiffy')
        self.smeared_tree.var('pv_sdiffz')
        self.smeared_tree.var('sv_sdiffx')
        self.smeared_tree.var('sv_sdiffy')
        self.smeared_tree.var('sv_sdiffz')
        self.smeared_tree.var('pvsv_sdistance')
        self.smeared_tree.var('L0b_Lc_sEn')
        self.smeared_tree.var('L0b_Lc_spx')
        self.smeared_tree.var('L0b_Lc_spy')
        self.smeared_tree.var('L0b_Lc_spz')
        self.smeared_tree.var('L0b_Lc_sp')           
        self.smeared_tree.var('Lc_proton_sEn')
    	self.smeared_tree.var('Lc_proton_spx')
    	self.smeared_tree.var('Lc_proton_spy')
    	self.smeared_tree.var('Lc_proton_spz')
    	self.smeared_tree.var('Lc_proton_sp')
    	self.smeared_tree.var('Lc_K_sEn')
    	self.smeared_tree.var('Lc_K_spx')
    	self.smeared_tree.var('Lc_K_spy')
    	self.smeared_tree.var('Lc_K_spz')
    	self.smeared_tree.var('Lc_K_sp')
    	self.smeared_tree.var('Lc_pi_sEn')
    	self.smeared_tree.var('Lc_pi_spx')
    	self.smeared_tree.var('Lc_pi_spy')
    	self.smeared_tree.var('Lc_pi_spz')
    	self.smeared_tree.var('Lc_pi_sp')
    	self.smeared_tree.var('L0b_mu_sEn')
    	self.smeared_tree.var('L0b_mu_spx')
    	self.smeared_tree.var('L0b_mu_spy')
    	self.smeared_tree.var('L0b_mu_spz')
    	self.smeared_tree.var('L0b_mu_sp')
    	self.smeared_tree.var('L0b_mu_spT')
    	self.smeared_tree.var('L0b_mu_spT_B')
        self.smeared_tree.var('mu_sIPdist')
        self.smeared_tree.var('Ds_Chi2')
        self.smeared_tree.var('Ds_NDF')
        self.smeared_tree.var('Ds_CDF')
        self.smeared_tree.var('Ds_fitvtx_x')
        self.smeared_tree.var('Ds_fitvtx_y')
        self.smeared_tree.var('Ds_fitvtx_z')
        self.smeared_tree.var('Ds_fitvtx_diffx')
        self.smeared_tree.var('Ds_fitvtx_diffy')
        self.smeared_tree.var('Ds_fitvtx_diffz')         
    	self.smeared_tree.var('Pvis_SS_sp')
        self.smeared_tree.var('Pvis_SS_sE')
        self.smeared_tree.var('Pvis_OS_sp')
        self.smeared_tree.var('Pvis_OS_sE')               
        self.smeared_tree.var('B_Dsmu_fitvtx_Chi2')
        self.smeared_tree.var('B_Dsmu_fitvtx_NDF')
        self.smeared_tree.var('B_Dsmu_fitvtx_CDF')
        self.smeared_tree.var('B_Dsmu_fitvtx_x')
        self.smeared_tree.var('B_Dsmu_fitvtx_y')
        self.smeared_tree.var('B_Dsmu_fitvtx_z')
        self.smeared_tree.var('B_Dsmu_fitvtx_diffx')
        self.smeared_tree.var('B_Dsmu_fitvtx_diffy')
        self.smeared_tree.var('B_Dsmu_fitvtx_diffz')
        self.smeared_tree.var('pv_DsPCA_x')
        self.smeared_tree.var('pv_DsPCA_y')
        self.smeared_tree.var('pv_DsPCA_z')
        self.smeared_tree.var('pv_muPCA_x')
        self.smeared_tree.var('pv_muPCA_y')
        self.smeared_tree.var('pv_muPCA_z')                                
        self.smeared_tree.var('Dsmu_DCA')
        self.smeared_tree.var('Lc_Kpipi_smass')
        self.smeared_tree.var('Lc_Kpipi_spT')
        self.smeared_tree.var('Lc_Kpipi_spT_FdB')
        self.smeared_tree.var('Kpipimu_spar')
        self.smeared_tree.var('Kpipimu_spTmis')
        self.smeared_tree.var('B_KKpimuX_mass')
        self.smeared_tree.var('B_KKpimuX_smass')
        self.smeared_tree.var('B_KKpimu_mcorr')
        self.smeared_tree.var('B_KKpimu_smcorr')
        
        # same for smeared values
        self.tree = Tree(self.cfg_ana.tree_name, self.cfg_ana.tree_title)
        self.tree.var('n_particles')
        self.tree.var('event_number')
        self.tree.var('pv_x')
        self.tree.var('pv_y')
        self.tree.var('pv_z')

    def process(self, event):
                
        store = event.input # This is just a shortcut
        event_info = store.get("EventInfo")
        particles_info = store.get("GenParticle")
        vertices_info = store.get("GenVertex")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        B_mc_truth = None # B0s particle (MC truth)
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
        
        plist_bquarkSS_mc_truth = list([])
        plist_bquarkOS_mc_truth = list([])
        plist_D_mc_truth = list([])
        plist_Ds0st_mc_truth = list([])
        plist_Dsst_mc_truth = list([])
        plist_Ds_mctruth = list([])
        plist_D1s_mc_truth = list([])
        plist_D1sprime_mc_truth = list([])
        plist_Dst_mc_truth = list([])        
        plist_DsK_mc_truth = list([])
        plist_Dspi_mc_truth = list([])
        plist_Dmu_mctruth = list([])        
        plist_D0mu_mctruth = list([])
        plist_mu_mctruth = list([])
        plist_DX_mctruth = list([])

        ##b quark kinematic variables
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
        
        bquarkSSth_px = 0.
        bquarkSSth_py = 0.
        bquarkSSth_pz = 0.
        #==============================
        
        #==Visible particles=======
        
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
        #==============================
        
        #=====================sanity check =====================
        pv_Ds_mctruth = None
        sv_Ds_mctruth = None
                    
        pv_K1_mctruth = None                    
        pv_K2_mctruth = None
        pv_pi_mctruth = None
        pv_mu_mctruth = None
                    
        pv_Ds = None
        sv_Ds = None
                    
        pv_K1 = None
        pv_K2 = None
        pv_pi = None
        pv_mu = None
        
        
        plist_Bd = list([])
        
        Bd_X_Et = 0.0
        Bd_X_px = 0.0
        Bd_X_py = 0.0
        Bd_X_pz = 0.0
        Bd_X_pmag = 0.0
        Bd_X_sEt = 0.0
        Bd_X_spx = 0.0
        Bd_X_spy = 0.0
        Bd_X_spz = 0.0
        Bd_X_spmag = 0.0
        D_X_Et = 0.0
        D_X_px = 0.0
        D_X_py = 0.0
        D_X_pz = 0.0
        D_X_pmag = 0.0
        D_X_spt = 0.0
        D_X_sEt = 0.0
        D_X_spx = 0.0
        D_X_spy = 0.0
        D_X_spz = 0.0
        D_X_spmag = 0.0
        D_X_pT = 0.0
        
        L0b_Lc_px = 0.0
        L0b_Lc_py = 0.0
        L0b_Lc_pz = 0.0
        L0b_Lc_p = 0.0
        L0b_Lc_En = 0.0
        Lc_proton_px = 0.0
        Lc_proton_py = 0.0
        Lc_proton_pz = 0.0
        Lc_proton_p = 0.0
        Lc_proton_En = 0.0
        Lc_K_px = 0.0
        Lc_K_py = 0.0
        Lc_K_pz = 0.0
        Lc_K_p = 0.0
        Lc_K_En = 0.0
        Lc_pi_px = 0.0
        Lc_pi_py = 0.0
        Lc_pi_pz = 0.0
        Lc_pi_p = 0.0
        Lc_pi_En = 0.0
        L0b_mu_px = 0.0
        L0b_mu_py = 0.0
        L0b_mu_pz = 0.0
        L0b_mu_p = 0.0
        L0b_mu_En = 0.0
        
        L0b_Lc_spx = 0.0
        L0b_Lc_spy = 0.0
        L0b_Lc_spz = 0.0
        Lc_proton_spx = 0.0
        Lc_proton_spy = 0.0
        Lc_proton_spz = 0.0
        Lc_K_spx = 0.0
        Lc_K_spy = 0.0
        Lc_K_spz = 0.0
        Lc_pi_spx = 0.0
        Lc_pi_spy = 0.0
        Lc_pi_spz = 0.0
        L0b_mu_spx = 0.0
        L0b_mu_spy = 0.0
        L0b_mu_spz = 0.0
        
        L0b_Lc_sEn = 0.0
        Lc_proton_sEn = 0.0
        Lc_K_sEn = 0.0
        Lc_pi_sEn = 0.0
        L0b_mu_sEn = 0.0
        Lc_Kpipi_spT = 0.0
        Lc_Kpipi_spT_FdB = 0.0
        
        Bd_Ds_En = 0.0
        Bd_Ds_sEn = 0.0
        L0b_Lc_sEn = 0.0
        Bd_pT = 0.0
        Bd_Ds_pT = 0.0
        Bd_D_pT = 0.0
        Bd_Ds_presol = 0.0
        Bd_D_presol = 0.0
        Bd_presol = 0.0
        
        Bd_Ds = None
        L0b_Lc = None
        Bd_mctruth = None
        Bd_pmctruth = None
        D_X_mctruth = None
        D_X_sp = None
        Bd_X_sp = None
        
        plist_L0b_Lc = list([])
        plist_Bd_D2 = list([])
        plist_L0b_mu = list([])        
        plist_Lc_Kpipi = list([])
        plist_L0b_mu = list([])
        
        L0b_pt = 0.0
        L0b_spt = 0.0
        L0b_mass = 0.0
        L0b_smass = 0.0
        
        Lc_Kpipimu_En = 0.0
        Ds_KKpimu_sEn = 0.0
        
        L0b_Kpipimu_spt = 0.0
        L0b_KpipimuX_spt = 0.0
        L0b_Kpipimu_mass = 0.0
        L0b_KpipimuX_mass = 0.0
        L0b_Kpipimu_smass = 0.0
        L0b_KpipimuX_smass = 0.0
        
        KKpimu_par = 0.0
        KKpimu_pTmis = 0.0
        L0b_Kpipimu_mcorr = 0.0
        
        Kpipimu_spar = 0.0
        Kpipimu_spTmis = 0.0
        L0b_Kpipimu_smcorr = 0.0
        
        L0b_fd = 0.0
        L0b_sfd = 0.0
        
        Ds_vtx = FastFit.FastFit(3, 0)
        Bd_Dsmu_vtx = FastFit.FastFit(2, 0)
        
        #===========================Mass check for B meson decay products===============================#
        for ptc_gen in ptcs:
            
            if (abs(ptc_gen.pdgid) == 5122 and ptc_gen.start_vertex != ptc_gen.end_vertex):
                
                Ds_vtx = FastFit.FastFit(3, 0)
                Bd_Dsmu_vtx = FastFit.FastFit(2, 0)
                
                self.counter += 1
                if self.counter %1000 == 0:
                #if self.counter %1 == 0:
                    print('Processing decay #{} ({:.1f} decays / s)'.format(self.counter, 100. / (time.time() - self.last_timestamp)))
                    self.last_timestamp = time.time()
                    
                Bd_mctruth = ptc_gen
                
                plist_Bd.append(ptc_gen)
        
                pv_mctruth = ptc_gen.start_vertex
                pv = copy.deepcopy(pv_mctruth)
                
                sv_mctruth = ptc_gen.end_vertex
                sv = copy.deepcopy(sv_mctruth)
                
                L0b_fd = np.sqrt((sv_mctruth.x - pv_mctruth.x)**2 + (sv_mctruth.y - pv_mctruth.y)**2 + (sv_mctruth.z - pv_mctruth.z)**2)
                
                for ptc_genL0b in ptcs:
                    
                    if (ptc_genL0b.start_vertex == sv_mctruth):
                        if abs(ptc_genL0b.pdgid) == 5122:
                            plist_Bd_D2.append(ptc_genL0b)                
                
                #print len(plist_Bd_D2)
                
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
                
                for ptc_gen1 in ptcs:   
                    
                    #==================================================================================================================
                    # looking for opposite b quark. This is a dirty hack. Works only because both PYTHIA/HepMC and PODIO store particles ordered. But IT'S NOT GUARANTEED
		            # need to find better algorithm to look for the opposite b-quark
                    index = 0
                    while os_b_quark_mc_truth == None and index < len(ptcs):
                        if (abs(ptcs[index].pdgid) == 5 and ptcs[index].status == 23) and np.dot([Bd_mctruth.p.px, Bd_mctruth.p.py, Bd_mctruth.p.pz], [ptcs[index].p.px, ptcs[index].p.py, ptcs[index].p.pz]) < 0:
                            os_b_quark_mc_truth = ptcs[index]
                            #print os_b_quark_mc_truth.start_vertex
                        index += 1
                    
                    if (abs(ptc_gen1.pdgid) == 5 and ptc_gen1.status == 23) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                        plist_bquarkSS_mc_truth.append(ptc_gen1)
                        b_quarkSS_mc_truth = ptc_gen1                                
                        bquarkSS_px = b_quarkSS_mc_truth.p.px
                        bquarkSS_py = b_quarkSS_mc_truth.p.py
                        bquarkSS_pz = b_quarkSS_mc_truth.p.pz
                        bquarkSS_p = b_quarkSS_mc_truth.p.absvalue()
                        bquarkSS_npx = b_quarkSS_mc_truth.p.px/b_quarkSS_mc_truth.p.absvalue()
                        bquarkSS_npy = b_quarkSS_mc_truth.p.py/b_quarkSS_mc_truth.p.absvalue()
                        bquarkSS_npz = b_quarkSS_mc_truth.p.pz/b_quarkSS_mc_truth.p.absvalue()
                        bquarkSS_E = b_quarkSS_mc_truth.energy
                                
                            #if (abs(ptc_gen1.pdgid) == 5 and ptc_gen1.status == 23) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) > 0):
                    if (abs(ptc_gen1.pdgid) == 5 and ptc_gen1.status == 23) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[Thr1_px, Thr2_py,Thr2_pz]) < 0):
                                
                        plist_bquarkOS_mc_truth.append(ptc_gen1)
                        b_quarkOS_mc_truth = ptc_gen1                         
                        bquarkOS_px = b_quarkOS_mc_truth.p.px
                        bquarkOS_py = b_quarkOS_mc_truth.p.py
                        bquarkOS_pz = b_quarkOS_mc_truth.p.pz
                        bquarkOS_p = b_quarkOS_mc_truth.p.absvalue()
                        bquarkOS_npx = b_quarkOS_mc_truth.p.px/b_quarkOS_mc_truth.p.absvalue()
                        bquarkOS_npy = b_quarkOS_mc_truth.p.py/b_quarkOS_mc_truth.p.absvalue()
                        bquarkOS_npz = b_quarkOS_mc_truth.p.pz/b_quarkOS_mc_truth.p.absvalue()
                        bquarkOS_E = b_quarkOS_mc_truth.energy
        
                            #if (ptc_gen1.status == 1 and abs(ptc_gen1.pdgid) not in [12,14,16]) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0):                            
                    if (ptc_gen1.status == 1 and abs(ptc_gen1.pdgid) not in [12,14,16]) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) > 0):
                            #if (ptc_gen1.status == 1 and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) < 0) and os_b_quark_mc_truth.status == 23):
                                                               
                        Pvis_SS_ID = ptc_gen1.pdgid 
                        Pvis_SS_mc_truth = ptc_gen1
                        Pvis_SS = copy.deepcopy(Pvis_SS_mc_truth)
                        Pvis_SS_pT = np.sqrt(ptc_gen1.p.px*ptc_gen1.p.px + ptc_gen1.p.py*ptc_gen1.p.py)
                        Pvis_SS_resol = momentum_res(Pvis_SS_pT)
                                
                        Pvis_SS_px += ptc_gen1.p.px
                        Pvis_SS_py += ptc_gen1.p.py
                        Pvis_SS_pz += ptc_gen1.p.pz
                                
                        Pvis_SS_p += ptc_gen1.p.absvalue()
                        Pvis_SS_E += ptc_gen1.energy
                                
                        if self.cfg_ana.smear_momentum:
                            Pvis_SS.p = smear_momentum(Pvis_SS.p, Pvis_SS_resol, Pvis_SS_resol, Pvis_SS_resol)
                                
                        Pvis_SS_sp += Pvis_SS.p.absvalue()
                        Pvis_SS_sE += np.sqrt(Pvis_SS.p.absvalue()*Pvis_SS.p.absvalue() + Pvis_SS.mass*Pvis_SS.mass)
                            
                            #if (ptc_gen1.status == 1 and abs(ptc_gen1.pdgid) not in [12,14,16]) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[os_b_quark_mc_truth.p.px, os_b_quark_mc_truth.p.py,os_b_quark_mc_truth.p.pz]) > 0):                            
                    if (ptc_gen1.status == 1 and abs(ptc_gen1.pdgid) not in [12,14,16]) and (np.dot([ptc_gen1.p.px,ptc_gen1.p.py, ptc_gen1.p.pz],[Thr2_px, Thr2_py,Thr2_pz]) < 0):
                                                                
                        Pvis_OS_ID = ptc_gen1.pdgid 
                        Pvis_OS_mc_truth = ptc_gen1
                        Pvis_OS = copy.deepcopy(Pvis_OS_mc_truth)
                        Pvis_OS_pT = np.sqrt(ptc_gen1.p.px*ptc_gen1.p.px + ptc_gen1.p.py*ptc_gen1.p.py)
                        Pvis_OS_resol = momentum_res(Pvis_OS_pT)
                                
                        Pvis_OS_px += ptc_gen1.p.px
                        Pvis_OS_py += ptc_gen1.p.py
                        Pvis_OS_pz += ptc_gen1.p.pz
                                
                        Pvis_OS_p += ptc_gen1.p.absvalue()
                        Pvis_OS_E += ptc_gen1.energy
                                
                        if self.cfg_ana.smear_momentum:
                            Pvis_OS.p = smear_momentum(Pvis_OS.p, Pvis_OS_resol, Pvis_OS_resol, Pvis_OS_resol)
                                
                        Pvis_OS_sp += Pvis_OS.p.absvalue()
                        Pvis_OS_sE += np.sqrt(Pvis_OS.p.absvalue()*Pvis_OS.p.absvalue() + Pvis_OS.mass*Pvis_OS.mass)
                            
                    Pnu_SS_p = np.sqrt(np.dot([Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz],[Pvis_SS_px-bquarkSS_px, Pvis_SS_py-bquarkSS_py, Pvis_SS_pz-bquarkSS_pz]))
                            
                    Pnu_OS_p = np.sqrt(np.dot([Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz],[Pvis_OS_px-bquarkOS_px, Pvis_OS_py-bquarkOS_py, Pvis_OS_pz-bquarkOS_pz]))
                    
                    #==================================================================================================================
                   
                    #if (ptc_gen1.start_vertex == sv_mctruth and ptc_gen1.start_vertex != ptc_gen1.end_vertex) and len(plist_Bd_D2) == 0:
                    if (ptc_gen1.start_vertex == sv_mctruth and ptc_gen1.start_vertex != ptc_gen1.end_vertex):
                    #if (ptc_gen1.start_vertex == Bd_dvtx):                        
                    
                        #print ptc_gen1.pdgid
                        
                        if abs(ptc_gen1.pdgid) == 13:
                            plist_L0b_mu.append(ptc_gen1)   
                            self.L0b_mu_counter += 1
                                                
                        if abs(ptc_gen1.pdgid) == 4122:
                            plist_L0b_Lc.append(ptc_gen1)
                            self.Lc_counter += 1
                
                        Bd_pmctruth = ptc_gen1
                        Bd_X_sp = copy.deepcopy(Bd_pmctruth)
                                        
                        Bd_pT = np.sqrt(Bd_pmctruth.p.px**2 + Bd_pmctruth.p.py**2)
                
                        Bd_presol = momentum_res(Bd_pT)
                
                        if self.cfg_ana.smear_momentum:
                             Bd_X_sp.p = smear_momentum(Bd_X_sp.p, Bd_presol, Bd_presol, Bd_presol)
                        
                        if (abs(ptc_gen1.pdgid) not in [12,13,14,16,4122,5122] and (ptc_gen1.status == 1)):
                        #if (abs(ptc_gen1.pdgid) not in [411,431]) and ptc_gen1.charge != 0:
                        
                            Bd_X_Et += Bd_pmctruth.energy
                            Bd_X_px += Bd_pmctruth.p.px
                            Bd_X_py += Bd_pmctruth.p.py
                            Bd_X_pz += Bd_pmctruth.p.pz
                            Bd_X_pmag += Bd_pmctruth.p.absvalue()
                            
                            Bd_X_sEt += np.sqrt(Bd_X_sp.p.absvalue()**2 + Bd_X_sp.mass**2)
                            Bd_X_spx += Bd_X_sp.p.px
                            Bd_X_spy += Bd_X_sp.p.py
                            Bd_X_spz += Bd_X_sp.p.pz
                            Bd_X_spmag += Bd_X_sp.p.absvalue()
                            
                        #print ptc_gen1.pdgid
                
                #print Bd_X_sEt
                #if len(plist_L0b_mu) == 1 and len(plist_Bd_D) == 1 and len(plist_Bd_D2) == 0:
                if len(plist_L0b_mu) == 1 and len(plist_L0b_Lc) == 1 and self.Lcmu_counter < 1000000:
                    
                    self.Lcmu_counter += 1
                    
                    #Bd_Ds_En = plist_L0b_mu[0].energy
                    #Bd_Ds_px = plist_L0b_mu[0].p.px
                    #Bd_Ds_py = plist_L0b_mu[0].p.py
                    #Bd_Ds_pz = plist_L0b_mu[0].p.pz
                    #Bd_Ds_p = plist_L0b_mu[0].p.absvalue()
                    
                    L0b_Lc = copy.deepcopy(plist_L0b_Lc[0])
                    L0b_mu = copy.deepcopy(plist_L0b_mu[0])
                    
                    L0b_Lc_pT = np.sqrt(plist_L0b_Lc[0].p.px**2 + plist_L0b_Lc[0].p.py**2)
                    L0b_mu_pT = np.sqrt(plist_L0b_mu[0].p.px**2 + plist_L0b_mu[0].p.py**2)
                    
                    L0b_Lc_presol = momentum_res(L0b_Lc_pT)
                    L0b_mu_presol = momentum_res(L0b_mu_pT)                   
                    
                    if self.cfg_ana.smear_momentum:
                        L0b_Lc.p = smear_momentum(L0b_Lc.p, L0b_Lc_presol, L0b_Lc_presol, L0b_Lc_presol)  
                        L0b_mu.p = smear_momentum(L0b_mu.p, L0b_mu_presol, L0b_mu_presol, L0b_mu_presol)                        
                    
                    L0b_Lc_px = plist_L0b_Lc[0].p.px
                    L0b_Lc_py = plist_L0b_Lc[0].p.py
                    L0b_Lc_pz = plist_L0b_Lc[0].p.pz
                    L0b_Lc_p = plist_L0b_Lc[0].p.absvalue()
                    L0b_Lc_En = plist_L0b_Lc[0].energy
                    
                    L0b_Lc_spx = L0b_Lc.p.px
                    L0b_Lc_spy = L0b_Lc.p.py
                    L0b_Lc_spz = L0b_Lc.p.pz
                                                           
                    L0b_pt = np.dot([plist_L0b_mu[0].p.px + plist_L0b_Lc[0].p.px + Bd_X_px, plist_L0b_mu[0].p.py + plist_L0b_Lc[0].p.py + Bd_X_py, plist_L0b_mu[0].p.pz + plist_L0b_Lc[0].p.pz + Bd_X_pz],[plist_L0b_mu[0].p.px + plist_L0b_Lc[0].p.px + Bd_X_px, plist_L0b_mu[0].p.py + plist_L0b_Lc[0].p.py + Bd_X_py, plist_L0b_mu[0].p.pz + plist_L0b_Lc[0].p.pz + Bd_X_pz])
                                        
                    L0b_mass = np.sqrt((plist_L0b_Lc[0].energy + plist_L0b_mu[0].energy + Bd_X_Et)**2 - L0b_pt)                                     
                    
                    L0b_spt = np.dot([L0b_mu.p.px + L0b_Lc.p.px + Bd_X_spx, L0b_mu.p.py + L0b_Lc.p.py + Bd_X_spy, L0b_mu.p.pz + L0b_Lc.p.pz + Bd_X_spz],[L0b_mu.p.px + L0b_Lc.p.px + Bd_X_spx, L0b_mu.p.py + L0b_Lc.p.py + Bd_X_spy, L0b_mu.p.pz + L0b_Lc.p.pz + Bd_X_spz])
                    
                    L0b_mu_sEn = np.sqrt(L0b_mu.p.absvalue()**2 + L0b_mu.mass**2)
                    
                    L0b_Lc_sEn = np.sqrt(L0b_Lc.p.absvalue()**2 + L0b_Lc.mass**2)
                
                    L0b_smass = np.sqrt((L0b_mu_sEn + L0b_Lc_sEn + Bd_X_sEt)**2 - L0b_spt)
                    
                    #print plist_L0b_mu[0].p.absvalue(), L0b_mu.p.absvalue(), np.sqrt(np.dot([L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz],[L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz]))
                                
                    #if L0b_smass > 5.5:
                    #    print 'Mass', L0b_mass, L0b_smass, Bd_X_sEt, plist_Bd[0].p.absvalue()
        
                    for ptc_gen2 in ptcs:
                    
                        if (len(plist_L0b_Lc) == 1 and (ptc_gen2.start_vertex == plist_L0b_Lc[0].end_vertex)):
                            plist_Lc_Kpipi.append(ptc_gen2)
                        
                        #if (len(plist_L0b_Lc) == 1 and (ptc_gen2.start_vertex == plist_L0b_Lc[0].end_vertex) and (ptc_gen2.start_vertex != ptc_gen2.end_vertex)):
                        #if (len(plist_L0b_Lc) == 1 and (ptc_gen2.start_vertex == plist_L0b_Lc[0].end_vertex)):
                        #    if abs(ptc_gen2.pdgid) == 13:
                        #        plist_L0b_mu.append(ptc_gen2)
                                
                        #        self.mu_counter += 1
                            
                            #D_X_mctruth = ptc_gen2
                            #D_X_sp = copy.deepcopy(D_X_mctruth)
                            
                            #D_X_pT = np.sqrt(D_X_mctruth.p.px**2 + D_X_mctruth.p.py**2)
                            #D_X_sp_presol = momentum_res(D_X_pT)
                            
                            #if self.cfg_ana.smear_momentum:
                            #    D_X_sp.p = smear_momentum(D_X_sp.p, D_X_sp_presol, D_X_sp_presol, D_X_sp_presol)
                        
                            #if abs(ptc_gen2.pdgid) not in [13,12,14,16]:
                            #if (abs(ptc_gen2.pdgid) not in [13,12,14,16]) and ptc_gen2.charge != 0:
                                
                            #    D_X_Et += D_X_mctruth.energy
                            #    D_X_px += D_X_mctruth.p.px
                            #    D_X_py += D_X_mctruth.p.py
                            #    D_X_pz += D_X_mctruth.p.pz
                            #    D_X_pmag += D_X_mctruth.p.absvalue()
                            #    
                            #    D_X_sEt += np.sqrt(D_X_sp.p.absvalue()**2 + D_X_sp.mass)
                            #    D_X_spx += D_X_sp.p.px
                            #    D_X_spy += D_X_sp.p.py
                            #    D_X_spz += D_X_sp.p.pz
                            #    D_X_spmag += D_X_sp.p.absvalue()
                                
                                #print ptc_gen2.pdgid
                    
                    #D_X_pt = np.dot([D_X_px, D_X_py, D_X_pz],[D_X_px, D_X_py, D_X_pz])
                    #D_X_mass = np.sqrt(D_X_Et**2 - D_X_pt) 
                    
                    #print plist_Lc_Kpipi[0].pdgid, plist_Lc_Kpipi[1].pdgid, plist_Lc_Kpipi[2].pdgid
                    
                    # plist_Lc_Kpipi[0]==proton, plist_Lc_Kpipi[1]==kaon, plist_Lc_Kpipi[2]==pion
                    
                    Lc_proton_px, Lc_proton_py, Lc_proton_pz = plist_Lc_Kpipi[0].p.px, plist_Lc_Kpipi[0].p.py, plist_Lc_Kpipi[0].p.pz
                    Lc_K_px, Lc_K_py, Lc_K_pz = plist_Lc_Kpipi[1].p.px, plist_Lc_Kpipi[1].p.py, plist_Lc_Kpipi[1].p.pz
                    Lc_pi_px, Lc_pi_py, Lc_pi_pz = plist_Lc_Kpipi[2].p.px, plist_Lc_Kpipi[2].p.py, plist_Lc_Kpipi[2].p.pz
                    L0b_mu_px, L0b_mu_py, L0b_mu_pz = plist_L0b_mu[0].p.px, plist_L0b_mu[0].p.py, plist_L0b_mu[0].p.pz
        
                    Lc_proton_p, Lc_K_p, Lc_pi_p, L0b_mu_p = plist_Lc_Kpipi[0].p.absvalue(), plist_Lc_Kpipi[0].p.absvalue(), plist_Lc_Kpipi[0].p.absvalue(), plist_L0b_mu[0].p.absvalue()
                    
                    Lc_K_En, Lc_pi_En, L0b_mu_En = plist_Lc_Kpipi[1].energy, plist_Lc_Kpipi[2].energy, plist_L0b_mu[0].energy
                    
                    #Misidentify proton as pion
                    Lc_proton_En = np.sqrt(Lc_proton_p**2 + plist_Lc_Kpipi[2].mass**2)
                    
        
                    Lc_Kpipimu_En = Lc_proton_En + Lc_K_En + Lc_pi_En + L0b_mu_En
                                                            
                    L0b_Kpipimu_pt = np.dot([Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px, Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py, Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz],[Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px, Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py, Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz])
                    
                    L0b_KpipimuX_pt = np.dot([Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px + Bd_X_px, Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py + Bd_X_py, Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz + Bd_X_pz],[Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px + Bd_X_px, Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py + Bd_X_py, Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz + Bd_X_pz])
                    
                    L0b_Kpipimu_mass = np.sqrt((Lc_Kpipimu_En)**2 - L0b_Kpipimu_pt)
                    #L0b_Kpipimu_mass = np.sqrt((Lc_Kpipimu_En + Bd_X_Et)**2 - L0b_Kpipimu_pt)
                    #L0b_Kpipimu_mass = np.sqrt((Lc_Kpipimu_En + D_X_Et)**2 - L0b_Kpipimu_pt)
                                        
                    L0b_KpipimuX_mass = np.sqrt((Lc_Kpipimu_En + Bd_X_Et)**2 - L0b_KpipimuX_pt)
                    
                    Lc_proton = copy.deepcopy(plist_Lc_Kpipi[0])
                    Lc_K = copy.deepcopy(plist_Lc_Kpipi[1])
                    Lc_pi = copy.deepcopy(plist_Lc_Kpipi[2])
                    L0b_mu  = copy.deepcopy(plist_L0b_mu[0])
                    
                    Lc_proton_pT = np.sqrt(plist_Lc_Kpipi[0].p.px**2 + plist_Lc_Kpipi[0].p.py**2)
                    Lc_K_pT = np.sqrt(plist_Lc_Kpipi[1].p.px**2 + plist_Lc_Kpipi[1].p.py**2)
                    Lc_pi_pT = np.sqrt(plist_Lc_Kpipi[2].p.px**2 + plist_Lc_Kpipi[2].p.py**2)
                    L0b_mu_pT  = np.sqrt(plist_L0b_mu[0].p.px**2 + plist_L0b_mu[0].p.py**2)
                    
                    Lc_proton_presol = momentum_res(Lc_proton_pT)
                    Lc_K_presol = momentum_res(Lc_K_pT)
                    Lc_pi_presol = momentum_res(Lc_pi_pT)
                    L0b_mu_presol  = momentum_res(L0b_mu_pT)
                    
                    if self.cfg_ana.smear_momentum:
                        Lc_proton.p = smear_momentum(Lc_proton.p, Lc_proton_presol, Lc_proton_presol, Lc_proton_presol)
                        Lc_K.p = smear_momentum(Lc_K.p, Lc_K_presol, Lc_K_presol, Lc_K_presol)
                        Lc_pi.p = smear_momentum(Lc_pi.p, Lc_pi_presol, Lc_pi_presol, Lc_pi_presol)
                        L0b_mu.p  = smear_momentum(L0b_mu.p, L0b_mu_presol, L0b_mu_presol, L0b_mu_presol)
        
                    Lc_proton_spx, Lc_proton_spy, Lc_proton_spz = Lc_proton.p.px, Lc_proton.p.py, Lc_proton.p.pz
                    Lc_K_spx, Lc_K_spy, Lc_K_spz = Lc_K.p.px, Lc_K.p.py, Lc_K.p.pz
                    Lc_pi_spx, Lc_pi_spy, Lc_pi_spz = Lc_pi.p.px, Lc_pi.p.py, Lc_pi.p.pz
                    L0b_mu_spx, L0b_mu_spy, L0b_mu_spz = L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz
                    
                    L0b_mu_spT = np.sqrt(L0b_mu_spx**2 + L0b_mu_spy**2)
                    
                    Lc_Kpipi_spT = np.sqrt((Lc_proton_spx + Lc_K_spx + Lc_pi_spx)**2 + (Lc_proton_spy + Lc_K_spy + Lc_pi_spy)**2)
                    
                    #Misidentify proton as pion
                    Lc_proton_sEn = np.sqrt(Lc_proton.p.absvalue()**2 + Lc_pi.mass**2) 
                    Lc_K_sEn = np.sqrt(Lc_K.p.absvalue()**2 + Lc_K.mass**2)
                    Lc_pi_sEn = np.sqrt(Lc_pi.p.absvalue()**2 + Lc_pi.mass**2)
                    L0b_mu_sEn = np.sqrt(L0b_mu.p.absvalue()**2 + L0b_mu.mass**2)
                    
                    Lc_Kpipi_sptot = np.dot([(Lc_proton_spx + Lc_K_spx + Lc_pi_spx),(Lc_proton_spy + Lc_K_spy + Lc_pi_spy), (Lc_proton_spz + Lc_K_spz + Lc_pi_spz)],[(Lc_proton_spx + Lc_K_spx + Lc_pi_spx),(Lc_proton_spy + Lc_K_spy + Lc_pi_spy), (Lc_proton_spz + Lc_K_spz + Lc_pi_spz)])

                    Lc_Kpipi_smass = np.sqrt((Lc_proton_sEn + Lc_K_sEn + Lc_pi_sEn)*(Lc_proton_sEn + Lc_K_sEn + Lc_pi_sEn) - Lc_Kpipi_sptot)
                    
                    Ds_KKpimu_sEn = Lc_proton_sEn + Lc_K_sEn + Lc_pi_sEn + L0b_mu_sEn
                    
                    L0b_Kpipimu_spt = np.dot([Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx, Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy, Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz],[Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx, Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy, Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz])
                    
                    L0b_KpipimuX_spt = np.dot([Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx + Bd_X_spx, Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy + Bd_X_spy, Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz + Bd_X_spz],[Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx + Bd_X_spx, Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy + Bd_X_spy, Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz + Bd_X_spz])
                    
                    #L0b_KpipimuX_spt = np.dot([Lc_proton.p.px + Lc_K.p.px + Lc_pi.p.px + L0b_mu.p.px, Lc_proton.p.py + Lc_K.p.py + Lc_pi.p.py + L0b_mu.p.py, Lc_proton.p.pz + Lc_K.p.pz + Lc_pi.p.pz + L0b_mu.p.pz],[Lc_proton.p.px + Lc_K.p.px + Lc_pi.p.px + L0b_mu.p.px, Lc_proton.p.py + Lc_K.p.py + Lc_pi.p.py + L0b_mu.p.py, Lc_proton.p.pz + Lc_K.p.pz + Lc_pi.p.pz + L0b_mu.p.pz])

                    #L0b_Kpipimu_smass = np.sqrt((Ds_KKpimu_sEn + Bd_X_sEt)**2 - L0b_Kpipimu_spt)
                    L0b_Kpipimu_smass = np.sqrt((Ds_KKpimu_sEn)**2 - L0b_Kpipimu_spt)
                    
                    L0b_KpipimuX_smass = np.sqrt((Ds_KKpimu_sEn + Bd_X_sEt + D_X_sEt)**2 - L0b_KpipimuX_spt)
                    #L0b_KpipimuX_smass = np.sqrt(Ds_KKpimu_sEn**2 - L0b_KpipimuX_spt)
                    
                    if self.cfg_ana.smear_pv:
                        pv = smear_vertex(pv, self.cfg_ana.pv_x_resolution, self.cfg_ana.pv_y_resolution, self.cfg_ana.pv_z_resolution)

                    if self.cfg_ana.smear_sv:
                        sv = smear_vertex(sv, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
                        
                    L0b_sfd = np.sqrt((sv.x - pv.x)**2 + (sv.y - pv.y)**2 + (sv.z - pv.z)**2)
        
                    #============Muon transverse momentum wrt B flight direction=====================================
                            
                    Fd_L0b = np.subtract([sv.x, sv.y, sv.z],[pv.x, pv.y, pv.z])/L0b_sfd
                            
                    L0b_mu_sppar_B = np.dot([Fd_L0b[0],Fd_L0b[1],Fd_L0b[2]],[L0b_mu_spx, L0b_mu_spy,L0b_mu.p.pz])*Fd_L0b
                            
                    L0b_mu_spT_FdB = np.sqrt(np.dot([L0b_mu_spx-L0b_mu_sppar_B[0],L0b_mu_spy-L0b_mu_sppar_B[1],L0b_mu.p.pz-L0b_mu_sppar_B[2]],[L0b_mu_spx-L0b_mu_sppar_B[0],L0b_mu_spy-L0b_mu_sppar_B[1],L0b_mu.p.pz-L0b_mu_sppar_B[2]]))
                    
                    Lc_Kpipi_spx = Lc_proton_spx + Lc_K_spx + Lc_pi_spx
                    Lc_Kpipi_spy = Lc_proton_spy + Lc_K_spy + Lc_pi_spy
                    Lc_Kpipi_spz = Lc_proton_spz + Lc_K_spz + Lc_pi_spz
                    
                    Lc_Kpipi_sppar_L0b = np.dot([Fd_L0b[0],Fd_L0b[1],Fd_L0b[2]],[Lc_Kpipi_spx,Lc_Kpipi_spy,Lc_Kpipi_spz])*Fd_L0b
                    
                    Lc_Kpipi_spT_FdB = np.sqrt(np.dot([Lc_Kpipi_spx-Lc_Kpipi_sppar_L0b[0],Lc_Kpipi_spy-Lc_Kpipi_sppar_L0b[1],Lc_Kpipi_spz-Lc_Kpipi_sppar_L0b[2]],[Lc_Kpipi_spx-Lc_Kpipi_sppar_L0b[0],Lc_Kpipi_spy-Lc_Kpipi_sppar_L0b[1],Lc_Kpipi_spz-Lc_Kpipi_sppar_L0b[2]]))
                            
                    #================================================================================================
                                
                    KKpimu_par = np.dot([sv_mctruth.x - pv_mctruth.x, sv_mctruth.y - pv_mctruth.y, sv_mctruth.z - pv_mctruth.z],[Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px, Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py, Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz])/L0b_fd

                    KKpimu_pTmis = np.dot([(Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/L0b_fd, (Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/L0b_fd, (Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/L0b_fd],[(Lc_proton_px + Lc_K_px + Lc_pi_px + L0b_mu_px) - KKpimu_par*(sv_mctruth.x - pv_mctruth.x)/L0b_fd, (Lc_proton_py + Lc_K_py + Lc_pi_py + L0b_mu_py) - KKpimu_par*(sv_mctruth.y - pv_mctruth.y)/L0b_fd, (Lc_proton_pz + Lc_K_pz + Lc_pi_pz + L0b_mu_pz) - KKpimu_par*(sv_mctruth.z - pv_mctruth.z)/L0b_fd])
                            
                    L0b_Kpipimu_mcorr = np.sqrt(L0b_Kpipimu_mass**2 + KKpimu_pTmis) + np.sqrt(KKpimu_pTmis)                    
                    
                    Kpipimu_spar = np.dot([sv.x - pv.x, sv.y - pv.y, sv.z - pv.z],[Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx, Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy, Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz])/L0b_sfd

                    Kpipimu_spTmis = np.dot([(Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx) - Kpipimu_spar*(sv.x - pv.x)/L0b_sfd, (Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy) - Kpipimu_spar*(sv.y - pv.y)/L0b_sfd, (Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz) - Kpipimu_spar*(sv.z - pv.z)/L0b_sfd],[(Lc_proton_spx + Lc_K_spx + Lc_pi_spx + L0b_mu_spx) - Kpipimu_spar*(sv.x - pv.x)/L0b_sfd, (Lc_proton_spy + Lc_K_spy + Lc_pi_spy + L0b_mu_spy) - Kpipimu_spar*(sv.y - pv.y)/L0b_sfd, (Lc_proton_spz + Lc_K_spz + Lc_pi_spz + L0b_mu_spz) - Kpipimu_spar*(sv.z - pv.z)/L0b_sfd])
                            
                    L0b_Kpipimu_smcorr = np.sqrt(L0b_Kpipimu_smass**2 + Kpipimu_spTmis) + np.sqrt(Kpipimu_spTmis)
                    
                    pv_Ds_mctruth = L0b_Lc.start_vertex
                    sv_Ds_mctruth = L0b_Lc.end_vertex
                    
                    pv_K1_mctruth = Lc_proton.start_vertex                    
                    pv_K2_mctruth = Lc_K.start_vertex
                    pv_pi_mctruth = Lc_pi.start_vertex
                    pv_mu_mctruth = L0b_mu.start_vertex
                    
                    pv_Ds = copy.deepcopy(pv_Ds_mctruth)
                    sv_Ds = copy.deepcopy(sv_Ds_mctruth)
                    
                    pv_K1 = copy.deepcopy(pv_K1_mctruth)
                    pv_K2 = copy.deepcopy(pv_K2_mctruth)
                    pv_pi = copy.deepcopy(pv_pi_mctruth)
                    pv_mu = copy.deepcopy(pv_mu_mctruth)
                    
                    #========================Smeared Muon Impact parameter=========================================
                    diffx = sv_mctruth.x - pv_mu_mctruth.x
                    diffy = sv_mctruth.y - pv_mu_mctruth.y
                    
                    mu_IPdist = abs(L0b_mu_px*diffy - L0b_mu_py*diffx)/np.sqrt(L0b_mu_px**2 + L0b_mu_py**2)
                    
                    #==============================================================================================
                    
                    
                    if self.cfg_ana.smear_sv:                        
                        pv_Ds = smear_vertex(pv_Ds, self.cfg_ana.sv_x_resolution, self.cfg_ana.sv_y_resolution, self.cfg_ana.sv_z_resolution)
                                
                    if self.cfg_ana.smear_tv:
                        sv_Ds = smear_vertex(sv_Ds, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                        pv_K1 = smear_vertex(pv_K1, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                        pv_K2 = smear_vertex(pv_K2, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                        pv_pi = smear_vertex(pv_pi, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                        pv_mu = smear_vertex(pv_mu, self.cfg_ana.tv_x_resolution, self.cfg_ana.tv_y_resolution, self.cfg_ana.tv_z_resolution)
                    
                    #========================Smeared Muon Impact parameter=========================================
                    sdiffx = sv.x - pv_mu.x
                    sdiffy = sv.y - pv_mu.y
                    
                    mu_sIPdist = abs(L0b_mu_spx*sdiffy - L0b_mu_spy*sdiffx)/np.sqrt(L0b_mu_spx**2 + L0b_mu_spy**2)                    
                    
                    #=====================Vertex fitting=========================================================================
                    
                    covmat_K1 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Lc_proton_presol*Lc_proton_presol,0.0,0.0], [0.0,0.0,0.0,0.0,Lc_proton_presol*Lc_proton_presol,0.0],[0.0,0.0,0.0,0.0,0.0,Lc_proton_presol*Lc_proton_presol]])
                                
                    covmat_K2 = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Lc_K_presol*Lc_K_presol,0.0,0.0], [0.0,0.0,0.0,0.0,Lc_K_presol*Lc_K_presol,0.0],[0.0,0.0,0.0,0.0,0.0,Lc_K_presol*Lc_K_presol]])
                                                                
                    covmat_pi = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,Lc_pi_presol*Lc_pi_presol,0.0,0.0], [0.0,0.0,0.0,0.0,Lc_pi_presol*Lc_pi_presol,0.0],[0.0,0.0,0.0,0.0,0.0,Lc_pi_presol*Lc_pi_presol]])
                            
                    #Ds_vtx.setDaughter(0,  Lc_proton.charge, np.array([Lc_proton.p.px, Lc_proton.p.py, Lc_proton.p.pz]), np.array([pv_K1.x, pv_K1.y, pv_K1.z]), np.diag([0.01] * 6))
                    #Ds_vtx.setDaughter(1,  DsK2.charge, np.array([Lc_K.p.px, Lc_K.p.py, Lc_K.p.pz]), np.array([pv_K2.x, pv_K2.y, pv_K2.z]), np.diag([0.01] * 6))
                    #Ds_vtx.setDaughter(2,  Lc_pi.charge, np.array([Lc_pi.p.px, Lc_pi.p.py, Lc_pi.p.pz]), np.array([pv_pi.x, pv_pi.y, pv_pi.z]), np.diag([0.01] * 6))
                            
                    Ds_vtx.setDaughter(0,  Lc_proton.charge, np.array([Lc_proton.p.px, Lc_proton.p.py, Lc_proton.p.pz]), 0.1*np.array([pv_K1.x, pv_K1.y, pv_K1.z]), covmat_K1)
                    Ds_vtx.setDaughter(1,  Lc_K.charge, np.array([Lc_K.p.px, Lc_K.p.py, Lc_K.p.pz]), 0.1*np.array([pv_K2.x, pv_K2.y, pv_K2.z]), covmat_K2)
                    Ds_vtx.setDaughter(2,  Lc_pi.charge, np.array([Lc_pi.p.px, Lc_pi.p.py, Lc_pi.p.pz]), 0.1*np.array([pv_pi.x, pv_pi.y, pv_pi.z]), covmat_pi)
    
                    Ds_vtx.fit(100)                            
                    Ds_fitvtx = 10.0*Ds_vtx.getVertex()
                            
                    Ds_fitvtx_diffx = sv_Ds.x - Ds_fitvtx[0]
                    Ds_fitvtx_diffy = sv_Ds.y - Ds_fitvtx[1]
                    Ds_fitvtx_diffz = sv_Ds.z - Ds_fitvtx[2]
                            
                    Ds_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Ds_vtx.getChi2(), Ds_vtx.getNDF())
                                
                    Lc_proton_fitp  = Ds_vtx.getDaughterMomentum(0)                                
                    Lc_K_fitp  = Ds_vtx.getDaughterMomentum(1)                                
                    Lc_pi_fitp  = Ds_vtx.getDaughterMomentum(2)
                            
                    Lc_proton_fitptot = np.sqrt(Lc_proton_fitp[0]*Lc_proton_fitp[0] + Lc_proton_fitp[1]*Lc_proton_fitp[1] + Lc_proton_fitp[2]*Lc_proton_fitp[2])
                    Lc_K_fitptot = np.sqrt(Lc_K_fitp[0]*Lc_K_fitp[0] + Lc_K_fitp[1]*Lc_K_fitp[1] + Lc_K_fitp[2]*Lc_K_fitp[2])
                    Lc_pi_fitptot = np.sqrt(Lc_pi_fitp[0]*Lc_pi_fitp[0] + Lc_pi_fitp[1]*Lc_pi_fitp[1] + Lc_pi_fitp[2]*Lc_pi_fitp[2])
                        
                    Lc_proton_fitpT = np.sqrt(Lc_proton_fitp[0]*Lc_proton_fitp[0] + Lc_proton_fitp[1]*Lc_proton_fitp[1])
                    Lc_K_fitpT = np.sqrt(Lc_K_fitp[0]*Lc_K_fitp[0] + Lc_K_fitp[1]*Lc_K_fitp[1])
                    Lc_pi_fitpT = np.sqrt(Lc_pi_fitp[0]*Lc_pi_fitp[0] + Lc_pi_fitp[1]*Lc_pi_fitp[1])
                                
                    Ds_KKpi_fitpx = Lc_proton_fitp[0] + Lc_K_fitp[0] + Lc_pi_fitp[0]
                    Ds_KKpi_fitpy = Lc_proton_fitp[1] + Lc_K_fitp[1] + Lc_pi_fitp[1]
                    Ds_KKpi_fitpz = Lc_proton_fitp[2] + Lc_K_fitp[2] + Lc_pi_fitp[2]
                    
                    Ds_KKpi_fitp = np.array([Ds_KKpi_fitpx,Ds_KKpi_fitpy,Ds_KKpi_fitpz])
                    Ds_KKpi_fitpmag = np.sqrt(Ds_KKpi_fitpx**2 + Ds_KKpi_fitpy**2 + Ds_KKpi_fitpz**2)
                    
                    mu_ptot = np.array([L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz])
                    mu_pv = np.array([pv_mu.x, pv_mu.y, pv_mu.z])
                            
                    Ds_vtxfit_charge = Lc_proton.charge + Lc_K.charge + Lc_pi.charge
                    
                    #========Calculate point of closest approach (DCA) along between Ds and mu=================================
                                
                    N_Dsmu = (np.cross([Ds_KKpi_fitp[0], Ds_KKpi_fitp[1], Ds_KKpi_fitp[2]],[L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz]))/(Ds_KKpi_fitpmag*L0b_mu.p.absvalue())
                    
                    N_Dsmu_mag = np.sqrt(np.dot([N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]]))
                                
                    pv_Ds_PCA = Ds_fitvtx + (Ds_KKpi_fitp/np.sqrt(Ds_KKpi_fitp[0]**2 + Ds_KKpi_fitp[1]**2 + Ds_KKpi_fitp[2]**2))*Ds_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,L0b_mu)
                                
                    pv_mu_PCA = mu_pv + (mu_ptot/np.sqrt(L0b_mu.p.px**2 + L0b_mu.p.py**2 + L0b_mu.p.pz**2))*mu_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,L0b_mu)
                                
                    #Calculate distance of closest approach (PCA) of Ds and mu
                                
                    Dsmu_DCA = np.dot([(pv_Ds_PCA[0]-pv_mu_PCA[0]),(pv_Ds_PCA[1]-pv_mu_PCA[1]),(pv_Ds_PCA[2]-pv_mu_PCA[2])],[N_Dsmu[0],N_Dsmu[1],N_Dsmu[2]])/N_Dsmu_mag
                            
                    #if np.abs(Dsmu_DCA) < 0.002:
                    #    print pv_mu_PCA[0], pv_mu.x, D_mc_truth.start_vertex.x,Dsmu_DCA
                    
                    #print Ds_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,L0b_mu), mu_PCA(Ds_fitvtx,pv_mu,Ds_KKpi_fitp,L0b_mu) 
                    #print pv_Ds_PCA[0], pv_Ds.x, pv_Ds_PCA[1], pv_Ds.y, pv_Ds_PCA[2], pv_Ds.z
                    #print pv_mu_PCA[0], pv_mu.x, D_mc_truth.start_vertex.x,Dsmu_DCA
                    #print sv_Dsfit[0], sv_Dsfit[1], sv_Dsfit[2], pv_Ds.x, pv_Ds.y, pv_Ds.z
                    #==========================================================================================================
                    
                    Ds_KKpi_pT = np.sqrt((Lc_proton_px + Lc_K_px + Lc_pi_px)**2 + (Lc_proton_py + Lc_K_py + Lc_pi_py)**2)
                    KKpi_presol = momentum_res(Ds_KKpi_pT)
                    
                    covmat_Ds = np.array([[4.9e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 4.9e-7,0.0,0.0,0.0,0.0],[0.0,0.0, 4.9e-7,0.0,0.0,0.0],[0.0,0.0,0.0,KKpi_presol*KKpi_presol,0.0,0.0], [0.0,0.0,0.0,0.0,KKpi_presol*KKpi_presol,0.0],[0.0,0.0,0.0,0.0,0.0,KKpi_presol*KKpi_presol]])
                            
                    covmat_mu = np.array([[2.5e-7,0.0,0.0,0.0,0.0,0.0],[0.0, 2.5e-7,0.0,0.0,0.0,0.0],[0.0,0.0,2.5e-7,0.0,0.0,0.0],[0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol,0.0,0.0], [0.0,0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol,0.0],[0.0,0.0,0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol]])
                            
                    #covmat_mu = np.array([[0.0016,0.0,0.0,0.0,0.0,0.0],[0.0, 0.0016,0.0,0.0,0.0,0.0],[0.0,0.0,0.0016,0.0,0.0,0.0],[0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol,0.0,0.0], [0.0,0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol,0.0],[0.0,0.0,0.0,0.0,0.0,L0b_mu_presol*L0b_mu_presol]])
                            
                    #covmat_Ds = np.array([[0.0016,0.0,0.0,0.0,0.0,0.0],[0.0, 0.0016,0.0,0.0,0.0,0.0],[0.0,0.0,0.0016,0.0,0.0,0.0],[0.0,0.0,0.0,Ds_resol*Ds_resol,0.0,0.0], [0.0,0.0,0.0,0.0,Ds_resol*Ds_resol,0.0],[0.0,0.0,0.0,0.0,0.0,Ds_resol*Ds_resol]])                            
                            
                    Bd_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([Ds_fitvtx[0], Ds_fitvtx[1],Ds_fitvtx[2]]),Ds_vtx.getVariance())
                            
                    #Bd_Dsmu_vtx.setDaughter(0, Ds_vtxfit_charge, np.array([Ds_KKpi_fitpx, Ds_KKpi_fitpy, Ds_KKpi_fitpz]), 0.1*np.array([pv_Ds_PCA[0], pv_Ds_PCA[1],pv_Ds_PCA[2]]),covmat_Ds)
                    
                    #Bd_Dsmu_vtx.setDaughter(1, Dmu.charge, np.array([Dmu.p.px, Dmu.p.py, Dmu.p.pz]), 0.1*np.array([pv_mu_PCA[0], pv_mu_PCA[1], pv_mu_PCA[2]]), covmat_mu)
                            
                    Bd_Dsmu_vtx.setDaughter(1, L0b_mu.charge, np.array([L0b_mu.p.px, L0b_mu.p.py, L0b_mu.p.pz]), 0.1*np.array([pv_mu.x, pv_mu.y, pv_mu.z]), covmat_mu)
                    
                    Bd_Dsmu_vtx.fit(100)
                    Bd_Dsmu_fitvtx = 10*Bd_Dsmu_vtx.getVertex()
                    Bd_Dsmu_fitvtx_CDF = 1 - scipy.stats.chi2.cdf(Bd_Dsmu_vtx.getChi2(), Bd_Dsmu_vtx.getNDF())
                            
                    #if np.abs(Dsmu_DCA) < 0.02:
                    #    print Bd_Dsmu_fitvtx, sv
                            
                    #print Bs_Dsmu_vtx.getDaughterVariance(1)
                    #if np.abs(Dsmu_DCA) < 0.03:
                    #    print "Smeared Bs vertex", sv.x, sv.y, sv.z 
                    #    print "Fitted Bs vertex", Bd_Dsmu_fitvtx[0], Bd_Dsmu_fitvtx[1], Bd_Dsmu_fitvtx[2]
                          
        
                    self.mc_truth_tree.fill('event_number',     event_number)
                    self.mc_truth_tree.fill('n_particles',      n_particles)
                    self.mc_truth_tree.fill('pv_x',             pv_mctruth.x)
                    self.mc_truth_tree.fill('pv_y',             pv_mctruth.y)
                    self.mc_truth_tree.fill('pv_z',             pv_mctruth.z)
                    self.mc_truth_tree.fill('sv_x',             sv_mctruth.x)
                    self.mc_truth_tree.fill('sv_y',             sv_mctruth.y)
                    self.mc_truth_tree.fill('sv_z',             sv_mctruth.z)
                    self.mc_truth_tree.fill('pv_Ds_x',          pv_Ds_mctruth.x)
                    self.mc_truth_tree.fill('pv_Ds_y',          pv_Ds_mctruth.y)
                    self.mc_truth_tree.fill('pv_Ds_z',          pv_Ds_mctruth.z)
                    self.mc_truth_tree.fill('sv_Ds_x',          sv_Ds_mctruth.x)
                    self.mc_truth_tree.fill('sv_Ds_y',          sv_Ds_mctruth.y)
                    self.mc_truth_tree.fill('sv_Ds_z',          sv_Ds_mctruth.z)
                    self.mc_truth_tree.fill('pv_Ds_x',          pv_Ds_mctruth.x)
                    self.mc_truth_tree.fill('pv_Ds_y',          pv_Ds_mctruth.y)
                    self.mc_truth_tree.fill('pv_Ds_z',          pv_Ds_mctruth.z)
                    self.mc_truth_tree.fill('sv_Ds_x',          sv_Ds_mctruth.x)
                    self.mc_truth_tree.fill('sv_Ds_y',          sv_Ds_mctruth.y)
                    self.mc_truth_tree.fill('sv_Ds_z',          sv_Ds_mctruth.z)
                    self.mc_truth_tree.fill('pvsv_distance',    L0b_fd)
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
                    self.mc_truth_tree.fill('L0b_Lc_En',        L0b_Lc_En)
                    self.mc_truth_tree.fill('L0b_Lc_px',        L0b_Lc_px)                            
                    self.mc_truth_tree.fill('L0b_Lc_py',        L0b_Lc_py)
                    self.mc_truth_tree.fill('L0b_Lc_pz',        L0b_Lc_pz)
                    self.mc_truth_tree.fill('L0b_Lc_p',         L0b_Lc_p)
                    self.mc_truth_tree.fill('Lc_proton_En',     Lc_proton_En)
                    self.mc_truth_tree.fill('Lc_proton_px',     Lc_proton_px)
                    self.mc_truth_tree.fill('Lc_proton_py',     Lc_proton_py)
                    self.mc_truth_tree.fill('Lc_proton_pz',     Lc_proton_pz)
                    self.mc_truth_tree.fill('Lc_proton_p',      Lc_proton_p)
                    self.mc_truth_tree.fill('Lc_K_En',          Lc_K_En)
                    self.mc_truth_tree.fill('Lc_K_px',          Lc_K_px)
                    self.mc_truth_tree.fill('Lc_K_py',          Lc_K_py)
                    self.mc_truth_tree.fill('Lc_K_pz',          Lc_K_pz)
                    self.mc_truth_tree.fill('Lc_K_p',           Lc_K_p)
                    self.mc_truth_tree.fill('Lc_pi_En',         Lc_pi_En)
                    self.mc_truth_tree.fill('Lc_pi_px',         Lc_pi_px)
                    self.mc_truth_tree.fill('Lc_pi_py',         Lc_pi_py)
                    self.mc_truth_tree.fill('Lc_pi_pz',         Lc_pi_pz)
                    self.mc_truth_tree.fill('Lc_pi_p',          Lc_pi_p)
                    self.mc_truth_tree.fill('L0b_mu_En',        L0b_mu_En)
                    self.mc_truth_tree.fill('L0b_mu_px',        L0b_mu_px)
                    self.mc_truth_tree.fill('L0b_mu_py',        L0b_mu_py)
                    self.mc_truth_tree.fill('L0b_mu_pz',        L0b_mu_pz)
                    self.mc_truth_tree.fill('L0b_mu_p',         L0b_mu_p)
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
                    self.mc_truth_tree.fill('KKpimu_pTmis',     np.sqrt(KKpimu_pTmis))
                    self.mc_truth_tree.fill('KKpimu_par',       KKpimu_par)
                    self.mc_truth_tree.fill('B_KKpimu_mass',   L0b_Kpipimu_mass)
                    self.mc_truth_tree.fill('B_KKpimu_mcorr',  L0b_Kpipimu_mcorr)
                    
                    self.mc_truth_tree.tree.Fill()
        
                    self.smeared_tree.fill('event_number',       event_number)
                    self.smeared_tree.fill('n_particles',        n_particles)
                    self.smeared_tree.fill('pv_sx',              pv.x)
                    self.smeared_tree.fill('pv_sy',              pv.y)
                    self.smeared_tree.fill('pv_sz',              pv.z)
                    self.smeared_tree.fill('sv_sx',              sv.x)
                    self.smeared_tree.fill('sv_sy',              sv.y)
                    self.smeared_tree.fill('sv_sz',              sv.z)
                    self.smeared_tree.fill('pv_sdiffx',          pv.x-pv_mctruth.x)
                    self.smeared_tree.fill('pv_sdiffy',          pv.y-pv_mctruth.y)
                    self.smeared_tree.fill('pv_sdiffz',          pv.z-pv_mctruth.z)
                    self.smeared_tree.fill('sv_sdiffx',          sv.x-sv_mctruth.x)
                    self.smeared_tree.fill('sv_sdiffy',          sv.y-sv_mctruth.y)
                    self.smeared_tree.fill('sv_sdiffz',          sv.z-sv_mctruth.z)
                    self.smeared_tree.fill('pv_Ds_sx',           pv_Ds.x)
                    self.smeared_tree.fill('pv_Ds_sy',           pv_Ds.y)
                    self.smeared_tree.fill('pv_Ds_sz',           pv_Ds.z)
                    self.smeared_tree.fill('sv_Ds_sx',           sv_Ds.x)
                    self.smeared_tree.fill('sv_Ds_sy',           sv_Ds.y)
                    self.smeared_tree.fill('sv_Ds_sz',           sv_Ds.z)
                    self.smeared_tree.fill('pvsv_sdistance',     L0b_sfd)
                    self.smeared_tree.fill('Pvis_SS_sp',         Pvis_SS_sp)
                    self.smeared_tree.fill('Pvis_SS_sE',         Pvis_SS_sE)
                    self.smeared_tree.fill('Pvis_OS_sp',         Pvis_OS_sp)
                    self.smeared_tree.fill('Pvis_OS_sE',         Pvis_OS_sE)
                    self.smeared_tree.fill('L0b_Lc_sEn',         L0b_Lc_sEn)
                    self.smeared_tree.fill('L0b_Lc_spx',         L0b_Lc_spx)                            
                    self.smeared_tree.fill('L0b_Lc_spy',         L0b_Lc_spy)
                    self.smeared_tree.fill('L0b_Lc_spz',         L0b_Lc_spz)
                    self.smeared_tree.fill('L0b_Lc_sp',          L0b_Lc.p.absvalue())                    
                    self.smeared_tree.fill('Lc_proton_sEn',      Lc_proton_sEn)
                    self.smeared_tree.fill('Lc_proton_spx',      Lc_proton_spx)
                    self.smeared_tree.fill('Lc_proton_spy',      Lc_proton_spy)
                    self.smeared_tree.fill('Lc_proton_spz',      Lc_proton_spz)
                    self.smeared_tree.fill('Lc_proton_sp',       Lc_proton.p.absvalue())
                    self.smeared_tree.fill('Lc_K_sEn',           Lc_K_sEn)
                    self.smeared_tree.fill('Lc_K_spx',           Lc_K_spx)
                    self.smeared_tree.fill('Lc_K_spy',           Lc_K_spy)
                    self.smeared_tree.fill('Lc_K_spz',           Lc_K_spz)
                    self.smeared_tree.fill('Lc_K_sp',            Lc_K.p.absvalue())
                    self.smeared_tree.fill('Lc_pi_sEn',          Lc_pi_sEn)
                    self.smeared_tree.fill('Lc_pi_spx',          Lc_pi_spx)
                    self.smeared_tree.fill('Lc_pi_spy',          Lc_pi_spy)
                    self.smeared_tree.fill('Lc_pi_spz',          Lc_pi_spz)
                    self.smeared_tree.fill('Lc_pi_sp',           Lc_pi.p.absvalue())
                    self.smeared_tree.fill('L0b_mu_sEn',         L0b_mu_sEn)
                    self.smeared_tree.fill('L0b_mu_spx',         L0b_mu_spx)
                    self.smeared_tree.fill('L0b_mu_spy',         L0b_mu_spy)
                    self.smeared_tree.fill('L0b_mu_spz',         L0b_mu_spz)
                    self.smeared_tree.fill('L0b_mu_sp',          L0b_mu.p.absvalue())
                    self.smeared_tree.fill('L0b_mu_spT',         L0b_mu_spT)
                    self.smeared_tree.fill('L0b_mu_spT_B',       L0b_mu_spT_FdB)
                    self.smeared_tree.fill('mu_sIPdist',         mu_sIPdist)
                    self.smeared_tree.fill('Ds_Chi2',            Ds_vtx.getChi2())
                    self.smeared_tree.fill('Ds_NDF',             Ds_vtx.getNDF())
                    self.smeared_tree.fill('Ds_CDF',             Ds_fitvtx_CDF)
                    self.smeared_tree.fill('Ds_fitvtx_x',        Ds_fitvtx[0])
                    self.smeared_tree.fill('Ds_fitvtx_y',        Ds_fitvtx[1])
                    self.smeared_tree.fill('Ds_fitvtx_z',        Ds_fitvtx[2])
                    self.smeared_tree.fill('Ds_fitvtx_diffx',    Ds_fitvtx_diffx)
                    self.smeared_tree.fill('Ds_fitvtx_diffy',    Ds_fitvtx_diffy)
                    self.smeared_tree.fill('Ds_fitvtx_diffz',    Ds_fitvtx_diffz)
                    self.smeared_tree.fill('B_Dsmu_fitvtx_Chi2', Bd_Dsmu_vtx.getChi2())
                    self.smeared_tree.fill('B_Dsmu_fitvtx_NDF',  Bd_Dsmu_vtx.getNDF())
                    self.smeared_tree.fill('B_Dsmu_fitvtx_CDF',  Bd_Dsmu_fitvtx_CDF)
                    self.smeared_tree.fill('B_Dsmu_fitvtx_x',    Bd_Dsmu_fitvtx[0])
                    self.smeared_tree.fill('B_Dsmu_fitvtx_y',    Bd_Dsmu_fitvtx[1])
                    self.smeared_tree.fill('B_Dsmu_fitvtx_z',    Bd_Dsmu_fitvtx[2])
                    self.smeared_tree.fill('B_Dsmu_fitvtx_diffx',Bd_Dsmu_fitvtx[0]-sv.x)
                    self.smeared_tree.fill('B_Dsmu_fitvtx_diffy',Bd_Dsmu_fitvtx[1]-sv.y)
                    self.smeared_tree.fill('B_Dsmu_fitvtx_diffz',Bd_Dsmu_fitvtx[2]-sv.z)
                    self.smeared_tree.fill('pv_DsPCA_x',         pv_Ds_PCA[0])
                    self.smeared_tree.fill('pv_DsPCA_y',         pv_Ds_PCA[1])
                    self.smeared_tree.fill('pv_DsPCA_z',         pv_Ds_PCA[2])
                    self.smeared_tree.fill('pv_muPCA_x',         pv_mu_PCA[0])
                    self.smeared_tree.fill('pv_muPCA_y',         pv_mu_PCA[1])
                    self.smeared_tree.fill('pv_muPCA_z',         pv_mu_PCA[2])                                
                    self.smeared_tree.fill('Dsmu_DCA',           np.abs(Dsmu_DCA))
                    self.smeared_tree.fill('Lc_Kpipi_smass',     Lc_Kpipi_smass)
                    self.smeared_tree.fill('Lc_Kpipi_spT',       Lc_Kpipi_spT)
                    self.smeared_tree.fill('Lc_Kpipi_spT_FdB',   Lc_Kpipi_spT_FdB)
                    self.smeared_tree.fill('Kpipimu_spar',       Kpipimu_spar)
                    self.smeared_tree.fill('Kpipimu_spTmis',     np.sqrt(Kpipimu_spTmis))
                    self.smeared_tree.fill('B_KKpimu_mcorr',     L0b_Kpipimu_mcorr)
                    self.smeared_tree.fill('B_KKpimu_smcorr',    L0b_Kpipimu_smcorr)
                    self.smeared_tree.fill('B_KKpimuX_mass',     L0b_KpipimuX_mass)
                    self.smeared_tree.fill('B_KKpimuX_smass',    L0b_KpipimuX_smass)
                    
                    self.smeared_tree.tree.Fill()
        
        #===============================================================================================#
        
                            
                            
    def write(self, unusefulVar):
        self.rootfile.Write()
        self.rootfile.Close()

        pb_canvas = TCanvas('pb_canvas', 'B momentum', 600, 400)
        pb_canvas.cd()
        self.pb_hist.Draw()
        pb_canvas.Update()


        print('Total decays processed: {}'.format(self.counter))
        print('B0 events with momentum cut: {}'.format(self.pb_counter))
        print('Lc baryon from L0b: {}'.format(self.Lc_counter)) 
        print('Muons from L0b: {}'.format(self.L0b_mu_counter))                
        print('Lc and mu events: {}'.format(self.Lcmu_counter))   
        print('Elapsed time: {:.1f} s ({:.1f} decays / s)'.format(time.time() - self.start_time, float(self.counter) / (time.time() - self.start_time)))
        print('Efficiency:\n\tMomentum of B cut: {:.3f}'.format (float(self.pb_counter)/float(self.counter)))
        raw_input('Press ENTER when finished')
