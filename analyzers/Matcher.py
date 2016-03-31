from heppy.framework.analyzer import Analyzer
from heppy.utils.deltar import matchObjectCollection, deltaR

import collections

class MatchInfo(object):
    '''Carries matching information.'''
    def __init__(self, obj, distance):
        '''
        Paramaters:
          - obj: the matched object.
          - distance: the distance to the matched object (an angle, delta r, delta r**2, ...)
        '''
        self.obj = obj
        self.distance = distance

    def __str__(self):
        return '{obj} ({dr2:5.1f})'.format(obj=str(self.obj), dr2=self.distance)

    

class Matcher(Analyzer):
    '''Particle matcher. 

    Works with any kind of object with a p4 function. 

    Simple example configuration: 
    
    from heppy_fcc.analyzers.Matcher import Matcher
    papas_jet_match = cfg.Analyzer(
      Matcher,
      instance_label = 'papas', 
      delta_r = 0.3,
      match_particles = 'gen_jets',
      particles = 'papas_jets'
    )

    particles: Name of the collection containing the particles to be matched. 
    match_particles: Name of the collection containing the particles where a match 
               is to be found. 

    In this particular case, each jet in "papas_jets" will end up with a new 
    attribute called "match". This attribute can be either the closest gen jet in the 
    "gen_jets" collection in case a gen_jet is found within delta R = 0.3, 
    or None in case a match cannot be found in this cone.

    More complex example configuration: 

    papas_particle_match_g2r = cfg.Analyzer(
      Matcher,
      instance_label = 'papas_g2r', 
      delta_r = 0.3, 
      particles = 'gen_particles_stable',
      match_particles = [
        ('papas_rec_particles', None),
        ('papas_rec_particles', 211),
        ('papas_rec_particles', 130),
        ('papas_rec_particles', 22)
      ] 
      )

    In this case, each gen particle in gen_particles_stable will end up with the following 
    new attributes: 
      - "match"    : closest reconstructed particle in "papas_rec_particles", if any. 
      - "match_211": closest reconstructed particle of pdgId 211 in "papas_rec_particles", 
                     if any. 
      - etc. 

    TODO: Colin: was well adapted, but probably better to do something more modular.
    for example: 
    papas_jet_match = cfg.Analyzer(
      Matcher,
      instance_label = 'gen_jets_match', 
      delta_r = 0.3,
      match_particles = 'gen_jets',
      particles = 'papas_jets'
    )
    would create for each papas_jet: 
      papas_jet.gen_jets_match
    that is a match object with 2 attributes: particle, distance
    in the more complicated case, just need to use a Filter to select the particles,
    and have several Matcher instances 

    note: one cannot attach the distance to the matched particle as 
    the match particle can be matched to another object... 
    '''
    def beginLoop(self, setup):
        super(Matcher, self).beginLoop(setup)
        self.dr2=self.cfg_ana.delta_r**2
        
    def process(self, event):
        particles = getattr(event, self.cfg_ana.particles)
        match_particles = getattr(event, self.cfg_ana.match_particles)
        pairs = matchObjectCollection(particles, match_particles,
                                      self.dr2)
        for ptc in particles:
            match = pairs[ptc]
            if match:
                dr = deltaR(ptc.theta(), ptc.phi(),
                            match.theta(), match.phi())
                setattr(ptc, self.instance_label, MatchInfo(match, dr))
