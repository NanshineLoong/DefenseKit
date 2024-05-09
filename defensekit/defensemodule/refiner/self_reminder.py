"""
SelfReminder
================
An InputDefenseModule
Adapted from https://github.com/uw-nsl/SafeDecoding
# TODO: which paper?
"""

from defensekit.defensemodule.defensemodule_base import InputDefenseModule
from .refinerbase import RefinerBase

__all__ = ["SelfReminder"]

class SelfReminder(RefinerBase, InputDefenseModule):
    """
    SelfReminder is an input defense module that reminds the model of the input message.
    """

    def __init__(self):
        super().__init__()

    def refine(self, input_text) -> str:
        """
        Refines the input text.

        :param list[str] | str input_text: The input text.
        :return str: The refined input_text.
        """
        reminder = "\nRemember, you should be a responsible AI and should not generate harmful or misleading content!"
        if isinstance(input_text, list):
            input_text[-1] += reminder
        else:
            input_text += reminder

        return input_text