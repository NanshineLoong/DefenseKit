"""
RegulatedDefenser class
====================

This class defines a model that has a supervised decoding process.
Adapted from EasyJailbreak/huggingface_model
"""

from easyjailbreak.models import WhiteBoxModelBase
from typing import Any, Callable
from fastchat.conversation import get_conv_template
import logging


class RegulatedDefenser(WhiteBoxModelBase):
    """
    RegulatedDefenser is a wrapper of a easyjailbreak model.
    It can be equiped with a decoding method.
    """

    def __init__(
            self, 
            model: Any, 
            tokenizer: Any,
            model_name: str,
            decoding_method: Callable[[str, Any, Any], str]):
        """
        Initializes the RegulatedDefenser.

        :param Any model: A huggingface model.
        :param Any tokenizer: A huggingface tokenizer.
        :param str model_name: The name of the model being userd. Refer to
            `FastChat conversation.py <https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>`_
            for possible options and templates.
        :param Callable[[str, Any, Any], str] decoding_method: The decoding method.
        """
        super().__init__(model, tokenizer)
        self.decoding_method = decoding_method
        self.model_name = model_name

        try:
            self.conversation = get_conv_template(model_name)
        except KeyError:
            logging.error(f'Invalid model_name: {model_name}. Refer to '
                          'https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py '
                          'for possible options and templates.')
            raise  # Continue raising the KeyError

        if model_name == 'llama-2':
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'

        self.format_str = self.create_format_str()

        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config

    def set_system_message(self, system_message: str):
        r"""
        Sets a system message to be used in the conversation.

        :param str system_message: The system message to be set for the conversation.
        """
        # TODO check llama2 add system prompt
        self.conversation.system_message = system_message

    def create_format_str(self):
        self.conversation.messages = []
        self.conversation.append_message(self.conversation.roles[0], "{prompt}")
        self.conversation.append_message(self.conversation.roles[1], "{response}")
        format_str = self.conversation.get_prompt()
        self.conversation.messages = []  # clear history
        return format_str

    def create_conversation_prompt(self, messages, clear_old_history=True):
        r"""
        Constructs a conversation prompt that includes the conversation history.

        :param list[str] messages: A list of messages that form the conversation history.
                                   Messages from the user and the assistant should alternate.
        :param bool clear_old_history: If True, clears the previous conversation history before adding new messages.
        :return: A string representing the conversation prompt including the history.
        """
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        self.conversation.append_message(self.conversation.roles[-1], None)
        return self.conversation.get_prompt()

    def clear_conversation(self):
        r"""
        Clears the current conversation history.
        """
        self.conversation.messages = []

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
