"""
Paraphraser
============
An InputDefenseModule
Adapted from https://github.com/uw-nsl/SafeDecoding
"""

from defensekit.defensemodule.defensemodule_base import InputDefenseModule
from defensekit.defensemodule.refiner.refinerbase import RefinerBase


class Paraphraser(RefinerBase, InputDefenseModule):
    """
    Paraphraser is an input defense module that paraphrases the input message.
    """

    def __init__(self, 
                 model):
        self.model = model

    def refine(self, input_text) -> list:
        """
        Refines the input text.

        :param str input_text: The input text.
        :return str: The refined input_text.
        """
        input_prompt_paraphrase = "Please paraphrase the following sentences. Give me paraphrased results only. Do not include any other information.\n" + input_text
        paraphrased_text = self.model.generate(input_prompt_paraphrase)
        return paraphrased_text
        