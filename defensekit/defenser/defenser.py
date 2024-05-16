"""
Defenser Class
========================
This module defines the Defenser model.
"""

from easyjailbreak.models import ModelBase
from defensekit.defensemodule.refiner import RefinerBase
from defensekit.defensemodule.filter import FilterBase
from defensekit.defensemodule.defensemodule_base import InputDefenseModule, OutputDefenseModule

class Defenser(ModelBase):
    """
    Defenser is a wrapper of a easyjailbreak model.
    It can be equiped with several defense modules.
    """

    def __init__(self, model: ModelBase, input_defense_modules: list = [], output_defense_modules: list = []):
        """
        Initializes the Defenser model with a model and a list of defense modules.

        :param ModelBase model: The underlying model for generation.
        :param list input_defense_modules: A list of defense modules on input side.
        :param list output_defense_modules: A list of defense modules on output side.
        """
        super().__init__()
        self.model = model
        self.input_defense_modules = input_defense_modules
        self.output_defense_modules = output_defense_modules

        # check if the defense modules are valid
        for module in self.input_defense_modules:
            if not isinstance(module, InputDefenseModule):
                raise ValueError(f"{module} is not an InputDefenseModule.")
        for module in self.output_defense_modules:
            if not isinstance(module, OutputDefenseModule):
                raise ValueError(f"{module} is not an OutputDefenseModule.")
            
    @property
    def tokenizer(self):
        return self.model.tokenizer

    def generate(self, messages, clear_old_history=True, **kwargs) -> str:
        """
        Generates output based on input arguments.
        
        :param list[str]|str messages: A list of messages or a single message string.
                                       User and assistant messages should alternate.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The generated output.
        """
        # 1. Apply input defense modules
        for module in self.input_defense_modules:
            if isinstance(module, RefinerBase):
                messages = module.refine(messages)
            elif isinstance(module, FilterBase):
                judgment = module.filter(messages)
                if not judgment:
                    return "Sorry, I can't help with this."

        # 2. Generate output
        output = self.model.generate(messages=messages, clear_old_history=clear_old_history, **kwargs)

        # 3. Apply output defense modules
        for module in self.output_defense_modules:
            if isinstance(module, RefinerBase):
                output = module.refine(output)
            elif isinstance(module, FilterBase):
                judgment = module.filter(output)
                if not judgment:
                    return "Sorry, I can't help with this."

        return output
