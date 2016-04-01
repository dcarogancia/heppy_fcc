class Vertex(object):
	"""A class that represents a 3D vertex"""
	
	def __init__(self, x = None, y = None, z = None):
		super(Vertex, self).__init__()
		self.x = x
		self.y = y
		self.z = z

	def __repr__(self):
		return 'Vertex({}, {}, {})'.format(self.x, self.y, self.z)

	def __str__(self):
		return '(x = {}, y = {}, z = {})'.format(self.x, self.y, self.z)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y and self.z == other.z

	def __ne__(self, other):
		return not self == other

	def is_valid(self):
		return (self.x is not None) and (self.y is not None) and (self.z is not None)
