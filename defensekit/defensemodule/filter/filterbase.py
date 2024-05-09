"""
FilterBase
==========
This file provides the interface for the filter module.
"""

from abc import ABC, abstractmethod


class FilterBase:
    """
    Defines the interface that filter modules should possess.
    """

    @abstractmethod
    def filter(self, input_text) -> bool:
        """
        Filters the input text.

        :param str input_text: The input text.
        :return bool: filter result.
        """
        pass