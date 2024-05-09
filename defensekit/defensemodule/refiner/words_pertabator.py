"""
WordsPerturbator
=================
An InputDefenseModule
Adapted from https://github.com/arobey1/smooth-llm
# TODO: paper? 
"""

from defensekit.defensemodule.defensemodule_base import InputDefenseModule
from .refinerbase import RefinerBase

import random
import string

__all__ = ["WordsPerturbator"]

class WordsPerturbator(RefinerBase, InputDefenseModule):
    """
    WordsPerturbator is an input defense module that perturbs the input words.
    """
    def __init__(self, pert_type="RandomSwapPerturbation", pert_pct=10):
        super().__init__()
        self.perturbation_fn = globals()[pert_type](q=pert_pct)

    def refine(self, input_text) -> str:
        """
        Refines the input text.

        :param list[str] | str input_text: The input text.
        :return str: The refined input_text.
        """
        # TODO: check when input_text is a list, how to handle it
        if isinstance(input_text, list):
            input_text[-1] = self.perturbation_fn(input_text[-1])
            return input_text
        else:
            return self.perturbation_fn(input_text)

class Perturbation:

    """Base class for random perturbations."""

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable

class RandomSwapPerturbation(Perturbation):

    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return ''.join(list_s)

class RandomPatchPerturbation(Perturbation):

    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        return ''.join(list_s)

class RandomInsertPerturbation(Perturbation):

    """Implementation of random insert perturbations.
    See `RandomPatchPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return ''.join(list_s)