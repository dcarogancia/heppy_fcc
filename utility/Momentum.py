#!/usr/bin/env python

"""
	Contains the Momentum class definition

	Momentum - a class that represents a physical momentum
"""

import math

class Momentum(object):
	"""
		A class that represents a physical momentum

		Attributes:
		px (float): x-component of the momentum
		py (float): y-component of the momentum
		pz (float): z-component of the momentum
	"""

	def __init__(self, px = None, py = None, pz = None):
		"""
			Constructor

			Args:
			px (optional [float]): x-component of the momentum. Defaults to None
			py (optional [float]): y-component of the momentum. Defaults to None
			pz (optional [float]): z-component of the momentum. Defaults to None
		"""

		super(Momentum, self).__init__()

		self.px = px
		self.py = py
		self.pz = pz

	@classmethod
	def fromlist(cls, lst):
		"""
			Classmethod that creates a Momentum from a list

			Args:
			lst (list): a list containing 3 elements
		"""

		return cls(lst[0], lst[1], lst[2])

	def __repr__(self):
		return 'Momentum({}, {}, {})'.format(self.px, self.py, self.pz)

	def __str__(self):
		return '(px = {}, py = {}, pz = {})'.format(self.px, self.py, self.pz)

	def __eq__(self, other):
		if isinstance(other, Momentum):
			return self.px == other.px and self.py == other.py and self.pz == other.pz
		else:
			return False

	def __ne__(self, other):
		return not self == other

	def raw(self):
		"""
			Creates a list with components of the momentum

			Returns:
			list: a list conatining 3 elements - x, y and z components of the momentum
		"""

		return [self.px, self.py, self.pz]

	def is_valid(self):
		"""
			Checks the validity of the momentum

			Returns:
			bool: True if all the components are set, False otherwise
		"""

		return (self.px is not None) and (self.py is not None) and (self.pz is not None)

	def absvalue(self):
		"""
			Absolute value of the momentum

			Returns:
			float: the absolute value of the momentum
		"""

		return math.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
