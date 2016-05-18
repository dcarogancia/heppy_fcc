from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex

class Particle(object):
	"""
		A class that represents a generated particle

		Attributes:
		pdgid (int): PDG ID of the particle
		p (Momentum): the momentum of the particle
		start_vertex (Vertex): the production vertex of the particle
		end_vertex (Vertex): the decay vertex of the particle
		status (int): HepMC status code of the particle
		charge (float): the charge of the particle
	"""

	def __init__(self, pdgid = None, p = None, start_vertex = None, end_vertex = None, status = None, charge = None):
		"""
			Constructor

			Args:
			pdgid (optional, [int]): PDG ID of the particle. Defaults to None
			p (optional, [Momentum]): the momentum of the particle. Defaults to None
			start_vertex (optional, [Vertex]): the production vertex of the particle. Defaults to None
			end_vertex (optional, [Vertex]): the decay vertex of the particle. Defaults to None
			status (optional, [int]): HepMC status code of the particle. Defaults to None
			charge (optional, [float]): the charge of the particle. Defaults to None
		"""

		super(Particle, self).__init__()

		self.pdgid = pdgid
		self.p = p
		self.start_vertex = start_vertex
		self.end_vertex = end_vertex
		self.status = status
		self.charge = charge

	@classmethod
	def fromfccptc(cls, fccptc):
		"""
			Classmethod that creates Particle from FCC EDM MCParticle

			Args:
			fccptc [MCParticle]: the MCParticle to use
		"""

		pdgid = fccptc.Core().Type
		p = Momentum(fccptc.Core().P4.Px, fccptc.Core().P4.Py, fccptc.Core().P4.Pz)
		start_vertex = Vertex(fccptc.StartVertex().Position().X, fccptc.StartVertex().Position().Y, fccptc.StartVertex().Position().Z) if fccptc.StartVertex().isAvailable() else None
		end_vertex = Vertex(fccptc.EndVertex().Position().X, fccptc.EndVertex().Position().Y, fccptc.EndVertex().Position().Z) if fccptc.EndVertex().isAvailable() else None
		status = fccptc.Core().Status
		charge = fccptc.Core().Charge

		return cls(pdgid, p, start_vertex, end_vertex, status, charge)

	def __repr__(self):
		return 'Particle({}, {}, {}, {}, {})'.format(self.pdgid, self.charge, self.p, self.start_vertex, self.end_vertex)

	def __str__(self):
		return 'Particle:\n\tPDGID: {}\n\tCharge: {}\n\tP: {}\n\tStartVertex: {}\n\tEndVertex: {}'.format(self.pdgid, self.charge, self.p, self.start_vertex, self.end_vertex)

	def is_valid(self):
		"""
			Checks the validity of the Particle

			Returns:
			bool: True if all the attributes but end_vertex are set, False otherwise
		"""

		return (self.pdgid is not None) and (self.charge is not None) and (self.p is not None) and self.p.is_valid() and (self.start_vertex is not None) and self.start_vertex.is_valid()
