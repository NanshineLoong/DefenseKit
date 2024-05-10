"""
RegulatedDefenser class
====================

This class defines a model that has a supervised decoding process.
"""

from easyjailbreak.models import HuggingfaceModel
from typing import Any, Callable, Optional, Dict
import logging


class RegulatedDefenser(HuggingfaceModel):
    """
    RegulatedDefenser is a wrapper of a HuggingfaceModel.
    It can be equiped with a decoding method.
    """

    def __init__(
            self, 
            model: Any, 
            tokenizer: Any,
            model_name: str,
            decoding_method: Callable[[str, Any, Any], str],
            generation_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the RegulatedDefenser.

        :param Callable[[str, Any, Any], str] decoding_method: The decoding method.
        """
        super().__init__(model, tokenizer, model_name, generation_config)
        self.decoding_method = decoding_method

    def generate(self, messages, clear_old_history=True, **kwargs):
        r"""
        Generates a response for the given messages within a single conversation.

        :param list[str]|str messages: The text input by the user. Can be a list of messages or a single message.
        :param bool clear_old_history: If True, clears the conversation history before generating a response.
        :param dict kwargs: Optional parameters for the model's generation function, such as 'temperature' and 'top_p'.
        :return: A string representing the pure response from the model, containing only the text of the response.
        """
        if isinstance(messages, str):
            messages = [messages]
        prompt = self.create_conversation_prompt(messages, clear_old_history=clear_old_history)

        output = self.decoding_method(prompt, self.model, self.tokenizer, **kwargs)

        return output
