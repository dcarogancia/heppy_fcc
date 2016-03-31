class Momentum(object):
	"""A class that represents a physical momentum"""
	def __init__(self, px = None, py = None, pz = None):
		super(Momentum, self).__init__()
		self.px = px
		self.py = py
		self.pz = pz

	def __repr__(self):
		return 'Momentum({}, {}, {})'.format(self.px, self.py, self.pz)

	def __str__(self):
		return '(px = {}, py = {}, pz = {})'.format(self.px, self.py, self.pz)

	def is_valid(self):
		return (self.px is not None) and (self.py is not None) and (self.pz is not None)
