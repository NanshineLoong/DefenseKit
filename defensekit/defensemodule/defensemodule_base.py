"""
InputDefenseModuleBase and Output
==================================
This file provides the abstract classes: InputDefenseModule and OutputDefenseModule.
"""

from abc import ABC, abstractmethod


class InputDefenseModule(ABC):
    """
    Defines the abstract class that input defense modules should inherit from.
    """

    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"


class OutputDefenseModule(ABC):
    """
    Defines the abstract class that output defense modules should inherit from.
    """

    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"
