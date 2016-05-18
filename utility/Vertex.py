#!/usr/bin/env python

class Vertex(object):
	"""A class that represents a 3D vertex"""

	def __init__(self, x = None, y = None, z = None):
		"""
			Constructor

			Args:
			x (optional [float]): x-coordinate of the vertex. Defaults to None
			y (optional [float]): y-coordinate of the vertex. Defaults to None
			z (optional [float]): z-coordinate of the vertex. Defaults to None
		"""

		super(Vertex, self).__init__()
		
		self.x = x
		self.y = y
		self.z = z

	@classmethod
	def fromlist(cls, lst):
		"""
			Classmethod that creates a Vertex from a list

			Args:
			lst (list): a list containing 3 elements
		"""

		return cls(lst[0], lst[1], lst[2])

	def __repr__(self):
		return 'Vertex(x = {}, y = {}, z = {})'.format(self.x, self.y, self.z)

	def __str__(self):
		return '({}, {}, {})'.format(self.x, self.y, self.z)

	def __eq__(self, other):
		if isinstance(other, Vertex):
			return self.x == other.x and self.y == other.y and self.z == other.z
		else:
			return False

	def __ne__(self, other):
		return not self == other

	def raw(self):
		"""
			Creates a list with coordinates

			Returns:
			list: a list conatining 3 elements - x, y and z coordinates of the vertex
		"""

		return [self.x, self.y, self.z]

	def is_valid(self):
		"""
			Checks the validity of the vertex

			Returns:
			bool: True if all the coordinates are set, False otherwise
		"""

		return (self.x is not None) and (self.y is not None) and (self.z is not None)
