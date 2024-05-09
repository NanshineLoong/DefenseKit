"""
EraseAndChecker
===============
An InputDefenseModule
Adapted From https://github.com/UCSB-NLP-Chang/SemanticSmooth
# TODO: only use LMFilter as Filter, how about Classifier?
# TODO: delete token ? or delete word?
"""

from defensekit.defensemodule.defensemodule_base import InputDefenseModule
from .filterbase import FilterBase
from .lm_filter import LMFilter

import random
from itertools import combinations


class EraseAndChecker(FilterBase, InputDefenseModule):
    """
    EraseAndChecker is an input defense module that erases the input text and checks if the model can still generate the same output.
    """

    def __init__(self, model, mode="suffix", max_erase=20, randomized=False, prompt_sampling_ratio=0.2):
        super().__init__()
        self.filter_model = LMFilter(model)

        self.mode = mode # infusion, erase, insertion; we use suffix as it is more efficient
        self.max_erase = max_erase
        self.randomized = randomized
        self.prompt_sampling_ratio = prompt_sampling_ratio

        if self.mode == 'infusion': 
            #! infusion is too heavy for a language model to use
            self.check_func = self.erase_and_check_infusion
        elif self.mode == 'suffix':
            self.check_func = self.erase_and_check_suffix
        else:
            raise ValueError("We currently only consider infusion and suffix mode")

    def delete_by_indices(self, data_list, indices):
        data_list = data_list.copy()
        # Sort indices in descending order to ensure deletion doesn't affect subsequent indices
        for index in sorted(indices, reverse=True):
            del data_list[index]
        return data_list

    def erase_and_check_suffix(self, prompt):   
        # Tokenize the prompt
        tokenizer = self.filter_model.tokenizer
        prompt_str = prompt.perturbable_prompt
        prompt_tokens = tokenizer(prompt_str)['input_ids'][1:]
        prompt_length = len(prompt_tokens)

        # Erase the prompt one token at a time from the end
        prompt_list = [prompt_str]
        for i in range(min(self.max_erase, prompt_length)):
            erased_prompt_tokens = prompt_tokens[:-(i+1)]
            erased_prompt = tokenizer.decode(erased_prompt_tokens)
            prompt_list.append(erased_prompt)

        if self.randomized:
            num_prompts_to_test = int(self.prompt_sampling_ratio * (len(prompt_list) - 1))
            prompts_to_test = prompt_list[0:1] + random.sample(prompt_list[1:], num_prompts_to_test)
        else:
            prompts_to_test = prompt_list

        # Check if any of the erased strings is harmful
        list_of_bools = self.is_harmful(prompts_to_test)
        return any(list_of_bools)

    def erase_and_check_infusion(self, prompt):
        # NOTE: too heavy to use, combination is too many
        tokenizer = self.filter_model.tokenizer
        prompt_str = prompt.perturbable_prompt
        prompt_tokens = tokenizer(prompt_str)['input_ids'][1:]
        prompt_length = len(prompt_tokens)

        prompt_list = [prompt_str]
        for i in range(min(self.max_erase, prompt_length)):
            # Mark erase locations
            erase_locations = list(combinations(range(prompt_length), i+1))
            for location in erase_locations:
                erased_prompt_tokens = self.delete_by_indices(prompt_tokens, location)
                erased_prompt = tokenizer.decode(erased_prompt_tokens)
                prompt_list.append(erased_prompt)
            
        if self.randomized:
            num_prompts_to_test = int(self.prompt_sampling_ratio * (len(prompt_list) - 1))
            prompts_to_test = prompt_list[0:1] + random.sample(prompt_list[1:], num_prompts_to_test)
        else:
            prompts_to_test = prompt_list

        list_of_bools = self.is_harmful(prompts_to_test)
        return any(list_of_bools)
    
    FILTER_PREFIXES = [
        "Not harmful", "not harmful", 
    ]

    def is_harmful(self, prompts_to_test):
        return [self.filter_model.filter(s) for s in prompts_to_test]
    
    def filter(self, text) -> bool:
        """
        Filters the generated text.

        :param list[str] | str text: The generated text.
        :return bool: True if the generated text is safe, False otherwise.
        """
        if isinstance(text, list):
            text = text[-1]
        return self.check_func(text)