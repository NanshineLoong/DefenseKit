from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.Jailbroken_wei_2023 import Jailbroken
from easyjailbreak.attacker.CodeChameleon_2024 import *

from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import from_pretrained

from defensekit.defenser import Defenser
from defensekit.defenser import EnsembleDefenser
from defensekit.defenser import RegulatedDefenser

from defensekit.defensemodule.filter import LMFilter
from defensekit.defensemodule.refiner import BackTranslator
from defensekit.defensemodule.refiner import WordsPerturbator
from defensekit.defensemodule.refiner import SelfReminder
from defensekit.decoding import safedecoding

import os

attack_method = 'PAIR'  # Jailbroken, CodeChameleon, PAIR
model_name = 'glm3'  # glm3, llama, vicuna, qwen
defense_method = 'none' # none, selfdefense, backtranslate, smoothllm, safedecoding
data_num = 1

result_file_path = f"results/AdvBench_{attack_method}_{defense_method}_{model_name}.jsonl"
if os.path.exists(result_file_path):
    raise ValueError(f"Result file {result_file_path} already exists.")

# ==== Dataset === #
from easyjailbreak.datasets import JailbreakDataset
dataset = JailbreakDataset('AdvBench')[:data_num]


# ==== Model === #
closed_model = OpenaiModel(model_name='glm-3-turbo', api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')

eval_model = closed_model

if model_name == "glm3":
    target_model = OpenaiModel(model_name='glm-3-turbo', api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')
else:
    import torch
    from easyjailbreak.models.huggingface_model import HuggingfaceModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name == "llama":
        auto_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', device_map='auto', torch_dtype=torch.bfloat16)
        tokenizers = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        target_model = HuggingfaceModel(auto_model,tokenizers,'llama-2')
    else:
        raise ValueError(f"Model name {model_name} is not supported.")

# === Defense === #
if defense_method == "none" or defense_method == 'test':
    defensed_model = Defenser(target_model, 
                              input_defense_modules=[], 
                              output_defense_modules=[])
elif defense_method == "selfdefense":
    defensed_model = Defenser(target_model, 
                              input_defense_modules=[], 
                              output_defense_modules=[LMFilter(model=target_model)])
elif defense_method == "backtranslate":
    defensed_model = Defenser(target_model, 
                              input_defense_modules=[], 
                              output_defense_modules=[BackTranslator(model=target_model)])
elif defense_method == "smoothllm":
    defensed_model = EnsembleDefenser(
        [Defenser(target_model, 
                  input_defense_modules=[WordsPerturbator()], 
                  output_defense_modules=[]) for _ in range(3)])
elif defense_method == "safedecoding":
    assert model_name == "llama", "Safe decoding is only supported for llama model."
    defensed_model = RegulatedDefenser(target_model, tokenizers, "llama-2", safedecoding)
else:
    raise ValueError(f"Defense method {defense_method} is not supported.")

# === Attack === #
if attack_method == "Jailbroken":
    attacker = Jailbroken(attack_model=None,
                          target_model=defensed_model,
                          eval_model=eval_model,
                          jailbreak_datasets=dataset)
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(result_file_path)
elif attack_method == "CodeChameleon":
    attacker = CodeChameleon(attack_model=None,
                  target_model=defensed_model,
                  eval_model=eval_model,
                  jailbreak_datasets=dataset)
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(result_file_path)
elif attack_method == "PAIR":
    attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-7b-v1.5',
                               model_name='vicuna_v1.1')
    attacker = PAIR(attack_model=attack_model,
                    target_model=defensed_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset,
                    n_iterations=2,
                    n_streams=2)
    attacker.attack(save_path=result_file_path)
elif attack_method == "AutoDan":
    raise ValueError("Not prepared yet.")
else:
    raise ValueError(f"Attack method {attack_method} is not supported.")


# attacker = PAIR(attack_model=model,
#                 target_model=target_model,
#                 eval_model=model,
#                 jailbreak_datasets=dataset)

# attacker.attack(save_path='AdvBench_pair_none.jsonl')