"""
RefinerBase
============
This file provides the interface for the refiner module.
"""

from abc import ABC, abstractmethod


class RefinerBase(ABC):
    """
    Defines the interface that refiner modules should possess.
    """

    @abstractmethod
    def refine(self, input_text) -> list:
        """
        Refines the input text.

        :param str input_text: The input text.
        :return str: The refined input_text.
        """
        pass