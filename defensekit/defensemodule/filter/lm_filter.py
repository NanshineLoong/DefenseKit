"""
LMFilter
========
An OutputDefenseModule / InputDefenseModule
Adapted from https://github.com/uw-nsl/SafeDecoding
"""

from defensekit.defensemodule.defensemodule_base import OutputDefenseModule, InputDefenseModule
from .filterbase import FilterBase

__all__ = ["LMFilter"]

class LMFilter(FilterBase, OutputDefenseModule, InputDefenseModule):
    """
    LMFilter is an output defense module that examines the generated text.
    """

    def __init__(self,
                 model):
        self.model = model

    def filter(self, text) -> bool:
        """
        Filters the text.

        :param list[str] | str text: The generated text.
        :return bool: True if the text is safe, False otherwise.
        """
        if isinstance(text, list):
            text = text[-1]
        check_user_prompt = f"Does the following text contain harmful content: {text} \nRespond with 'Yes, this is harmful' or 'No, this is not harmful'."
        output_check = self.model.generate(check_user_prompt)
        if "yes" in output_check.lower():
            return False
        else:
            return True
