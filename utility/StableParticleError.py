#!/usr/bin/env python

"""
    Defines the StableParticleError class

    StableParticleError - a class of the exceptions to be thrown in case of operations typical for decaying particles on a stable particle
"""

class StableParticleError(RuntimeError):
    """An exceptions to be thrown in case of operations typical for decaying particles on a stable particle"""

    def __init__(self, error_string = 'Particle is stable'):
        """
            Constructor

            Args:
            error_string (optional, [str]): error message. Defaults to 'Particle is stable'
        """
        super(StableParticleError, self).__init__(error_string)
