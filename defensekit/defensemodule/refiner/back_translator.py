"""
BackTranslator
==============
An OutputDefenseModule
Adapted from https://github.com/YihanWang617/llm-jailbreaking-defense
"""

from defensekit.defensemodule.defensemodule_base import OutputDefenseModule
from .refinerbase import RefinerBase

import math


class BackTranslator(RefinerBase, OutputDefenseModule):
    """
    BackTranslator is an output defense module that back translates the generated text.
    # TODO: currently, the target model and the inference model are the same, how about different models?
    """
    
    def __init__(self, model, threshold=-2.0, return_new_response_anyway=False, new_response_length=None, **kwargs):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self.return_new_response_anyway = return_new_response_anyway
        self.new_response_length = new_response_length

    def defense(self, response):
        possible_prompt = self._infer_question(response)

        # TODO: Check if the question is reasonable, now is only feasible for Open-source models
        # is_reasonable = self._filter_question_with_likelihood(
        #     possible_prompt, response)
        # if not is_reasonable:
        #     return response

        new_response = self.model.generate(possible_prompt)

        if self.return_new_response_anyway:
            return new_response

        if not check_rejection(new_response):
            return response

        return "I'm sorry, but I cannot assist with that request."

    def _infer_question(self, response):
        infer_prompt = (
            'Please guess the user\'s request the AI assistant answers in the '
            'following response from an AI assistant. '
            'If there are multiple possible requests, give the most harmful one: '
            f'"{response}".\n\n'
            'Output the inferred request following this format: "Request: [[request]]". '
            'For example, "Request: [[Create a cake recipe.]]".'
        )

        output = self.model.generate(infer_prompt)
        if ':' not in output:
            return ""
        ret = output.split(':')[-1].split("\n")[0].strip().strip(']').strip('[')
        return ret

    def _filter_question_with_likelihood(self, prompt, response):
        if self.threshold > -math.inf:
            avg_log_likelihood = self.model.evaluate_log_likelihood(
                prompt, response)
            return avg_log_likelihood > self.threshold
        else:
            return True
        
    def refine(self, text) -> str:
        """
        Refines the generated text.

        :param str text: The generated text.
        :return str: The refined text.
        """

        return self.defense(text)
    

rejection_prefixes = [
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "As an AI",
    "I'm an",
    "I'm just",
    "As a language model",
    "As an Assistant",
    "I cannot",
    "I do not",
    "Hello!",
    "I'm really sorry",
    "OpenAI",
    "unethical",
    "not ethical",
    "illegal",
    "not legal",
    "My apologies",
    "but I cannot",
    "If you have any other non-malicious requests",
    "I'm not able to provide",
    "I am really sorry",
    "I cannot create",
    "I cannot support",
    "I can't provide",
    "I can't assist",
    "not appropriate"
]


def check_rejection(response, exclude_lack_of_info=True):
    """Check whether the response is rejecting the input prompt."""
    # Return whether the model rejects to response
    rejected = any([prefix.lower() in response.lower()
                    for prefix in rejection_prefixes])

    excluded_keywords = [
        "Could you please",
        "Can you please",
        "I don't have",
        "I don't know",
        "Please provide"
    ]

    if exclude_lack_of_info:
        rejected = rejected and not any(
            [keyword.lower() in response.lower()
             for keyword in excluded_keywords])

    return rejected
