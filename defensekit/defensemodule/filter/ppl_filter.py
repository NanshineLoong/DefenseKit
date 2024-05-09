"""
PPLFilter
==================
An InputDefenseModule, filters the input text based on the perplexity of the text.
Adapted from https://github.com/uw-nsl/SafeDecoding
"""

from .filterbase import FilterBase
from defensekit.defensemodule.defensemodule_base import InputDefenseModule

import os
import gc
from math import exp
from typing import List
import urllib
import transformers
import torch

def internet_connection(host: str = 'http://google.com'):
    """ check if internet connection is available """
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))


class PPLFilter(FilterBase, InputDefenseModule):
    """
    PPLFilter is an input defense module that filters the input text based on the perplexity of the text.
    """

    def __init__(self, 
                 model: str = 'gpt2', 
                 threshold: float = 10.0,   # TODO: what?
                 use_auth_token: bool = None,
                 max_length: int = None,
                 num_gpus: int = None,
                 torch_dtype=None,
                 device_map: str = None,
                 low_cpu_mem_usage: bool = False,
                 trust_remote_code: bool = True,
                 offload_folder: str = None,
                 hf_cache_dir: str = None):
        """
        Initializes the PPLFilter
        
        :param str model: The model to use for perplexity calculation.
        :param float threshold: The threshold for filtering.
        """
        # load model
        params = {"local_files_only": not internet_connection(), "use_auth_token": use_auth_token,
                  "trust_remote_code": trust_remote_code}
        if hf_cache_dir is not None:
            params["cache_dir"] = hf_cache_dir
        if offload_folder is not None:
            params["offload_folder"] = offload_folder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, **params)
        self.config = transformers.AutoConfig.from_pretrained(model, **params)
        self.threshold = threshold

        params.update({"config": self.config, "low_cpu_mem_usage": low_cpu_mem_usage})
        if torch_dtype is not None:
            params['torch_dtype'] = torch_dtype
        if device_map is not None:
            params['device_map'] = device_map
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **params)

        self.pad_token_initialized = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': "<<PAD>>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.pad_token_initialized = True

        if max_length is None:
            self.max_length = None
        else:
            self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length
            assert self.max_length <= self.tokenizer.model_max_length, f"{self.max_length} > {self.tokenizer.model_max_length}"

        # loss function
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # GPU setup
        self.device = self.model.device
        if device_map is None:
            num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
            if num_gpus == 1:
                self.model.to('cuda')
                self.device = self.model.device
            elif num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.to('cuda')
                self.device = self.model.module.device
        self.model.eval()

    def get_perplexity(self, input_texts: str | List, batch: int = None):
        """ Compute the perplexity on recurrent LM.

        :param input_texts: A string or list of input texts for the encoder.
        :param batch: Batch size
        :return: A value or list of perplexity.
        """

        # batch preparation
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        batch = len(input_texts) if batch is None else batch
        batch_id = list(range(0, len(input_texts), batch)) + [len(input_texts)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))

        loss_list = []
        with torch.no_grad():
            for s, e in batch_id:

                # run model inference
                if self.max_length is not None:
                    model_inputs = self.tokenizer(input_texts[s:e], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
                else:
                    model_inputs = self.tokenizer(input_texts[s:e], truncation=True, padding=True, return_tensors='pt')
                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')

                output = self.model(**{k: v.to(self.device) for k, v in model_inputs.items()})
                logit = output['logits']
                if self.pad_token_initialized:
                    logit = logit[:, :, :-1]

                # shift the label sequence for causal inference
                label = model_inputs['input_ids']
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift so that tokens < n predict n
                shift_logits = logit[..., :-1, :].contiguous()
                shift_label = label[:, 1:].contiguous()

                # compute loss
                valid_length = (shift_label != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                valid_length = valid_length.to(self.device)
                shift_label = shift_label.to(self.device)
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
                loss = loss.view(len(output['logits']), -1)
                loss = torch.sum(loss, -1) / valid_length
                loss_list += loss.cpu().tolist()

                if FORCE_RESET:
                    del model_inputs
                    del loss
                    del output
                    gc.collect()
                    torch.cuda.empty_cache()

        # conversion to perplexity
        ppl = [exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl

    def filter(self, input_text: str) -> bool:
        """
        Filters the input text based on the perplexity of the text.

        :param list[str] | str input_text: The input text.
        :return bool: filter result.
        """
        if isinstance(input_text, list):
            input_text = " ".join(input_text)
        ppl = self.get_perplexity(input_text)
        return ppl < self.threshold
