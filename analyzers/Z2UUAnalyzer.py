from heppy.framework.analyzer import Analyzer

import math
import copy

import numpy

from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex
from heppy_fcc.utility.Particle import Particle

particles = {2212: 'proton',
             -2212: 'antiproton',
             2112: 'neutron',
             -2112: 'antineutron',
             211: 'pi+',
             -211: 'pi-',
             321: 'K+',
             -321: 'K-',
             130: 'K0',
             -130: 'anti-K0',
             11: 'e-',
             -11: 'e+',
             13: 'mu-',
             -13: 'mu+',
             22: 'photon'}

class Z2UUAnalyzer(Analyzer):
    def contains_high_energy_track(self):
        ret = False

        for ptc in self.u_hemisphere:
            if ptc.p.absvalue() > 40 and (abs(ptc.pdgid) in [211, 321, 2212]):
                ret = True

        for ptc in self.ubar_hemisphere:
            if ptc.p.absvalue() > 40 and (abs(ptc.pdgid) in [211, 321, 2212]):
                ret = True

        return ret

    def beginLoop(self, setup):
        super(Z2UUAnalyzer, self).beginLoop(setup)

    def process(self, event):
        self.u = None
        self.ubar = None

        self.u_hemisphere = set([])
        self.ubar_hemisphere = set([])

        store = event.input # This is just a shortcut
        event_info = store.get("EventInfo")
        particles_info = store.get("GenParticle")
        vertices_info = store.get("GenVertex")

        event_number = event_info.at(0).Number()
        ptcs = list(map(Particle.fromfccptc, particles_info))
        n_particles = len(ptcs)

        # looking for Z
        for ptc in ptcs:
            # looking for initial uubar pair. This is a dirty hack. Works only because both PYTHIA/HepMC and PODIO store particles ordered. But IT'S NOT GUARANTEED
            index = 0
            while (self.u == None or self.ubar == None) and index < len(ptcs):
                if ptcs[index].pdgid == 2:
                    self.u = ptcs[index]
                if ptcs[index].pdgid == -2:
                    self.ubar = ptcs[index]

                index += 1

        for ptc in ptcs:
            if ptc.status == 1:
                if numpy.dot([self.u.p.px, self.u.p.py, self.u.p.pz], [ptc.p.px, ptc.p.py, ptc.p.pz]) > 0:
                    self.u_hemisphere.add(ptc)
                else:
                    self.ubar_hemisphere.add(ptc)

        if self.contains_high_energy_track():
            print('Event #{}'.format(event_number))
            print('\tu hemisphere')
            print('\t\t{:<12}{:<10}'.format('Particle', 'Momentum'))
            for ptc in self.u_hemisphere:
                print('\t\t{:<12}{:<.3f} GeV'.format(particles[ptc.pdgid], ptc.p.absvalue()))
            print('\tubar hemisphere')
            print('\t\t{:<12}{:<8}'.format('Particle', 'Momentum'))
            for ptc in self.ubar_hemisphere:
                print('\t\t{:<12}{:.3f} GeV'.format(particles[ptc.pdgid], ptc.p.absvalue()))


    def write(self, unusefulVar):
        pass
