from heppy_fcc.utility.Momentum import Momentum
from heppy_fcc.utility.Vertex import Vertex

class Particle(object):
	"""A class that represents a generated particle"""
	def __init__(self, pdgid = None, charge = None, p = None, start_vertex = None, end_vertex = None):
		super(Particle, self).__init__()
		self.pdgid = pdgid
		self.charge = charge
		self.p = p
		self.start_vertex = start_vertex
		self.end_vertex = end_vertex

	@classmethod
	def fromfccptc(cls, fccptc):
		pdgid = fccptc.Core().Type
		charge = fccptc.Core().Charge
		p = Momentum(fccptc.Core().P4.Px, fccptc.Core().P4.Py, fccptc.Core().P4.Pz)
		start_vertex = Vertex(fccptc.StartVertex().Position().X, fccptc.StartVertex().Position().Y, fccptc.StartVertex().Position().Z) if fccptc.StartVertex().isAvailable() else None
		end_vertex = Vertex(fccptc.EndVertex().Position().X, fccptc.EndVertex().Position().Y, fccptc.EndVertex().Position().Z) if fccptc.EndVertex().isAvailable() else None

		return cls(pdgid, charge, p, start_vertex, end_vertex)

	def __repr__(self):
		return 'Particle({}, {}, {}, {}, {})'.format(self.pdgid, self.charge, self.p, self.start_vertex, self.end_vertex)

	def __str__(self):
		return 'Particle:\n\tPDGID: {}\n\tCharge: {}\n\tP: {}\n\tStartVertex: {}\n\tEndVertex: {}'.format(self.pdgid, self.charge, self.p, self.start_vertex, self.end_vertex)

	def is_valid(self):
		return (self.pdgid is not None) and (self.charge is not None) and (self.p is not None) and self.p.is_valid() and (self.start_vertex is not None) and self.start_vertex.is_valid()
