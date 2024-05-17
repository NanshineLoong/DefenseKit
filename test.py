from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.Jailbroken_wei_2023 import Jailbroken
from easyjailbreak.attacker.CodeChameleon_2024 import *
from easyjailbreak.attacker.ReNeLLM_ding_2023 import *

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

attack_method = 'CodeChameleon'  # Jailbroken, CodeChameleon, PAIR
model_name = 'llama'  # glm3, llama, 
defense_method = 'lmfilter-backtranslator' # none, selfdefense, backtranslate, smoothllm, safedecoding, 
# lmfilter-backtranslator, 
# selfreminder-selfdefense, selfreminder-safedecoding-selfdefense
# async-smoothllm
data_num = 1

defense_methods = ["none", "selfdefense", "backtranslate", "smoothllm", "safedecoding", "lmfilter-backtranslator", "selfreminder-selfdefense", "selfreminder-safedecoding-selfdefense", "async-smoothllm"]
model_names = ["llama"]
attack_methods = ["jailbroken", "CodeChameleon", "ReNeLLM"]

auto_model = None
tokenizers = None

for model_name in model_names:
    for attack_method in attack_methods:
        for defense_method in defense_methods:
            print(f"Attack: {attack_method}, Model: {model_name}, Defense: {defense_method}")
            result_file_path = f"results/AdvBench_{attack_method}_{defense_method}_{model_name}.jsonl"
            if os.path.exists(result_file_path):
                continue

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
                    auto_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', device_map='auto', torch_dtype=torch.bfloat16) if auto_model is None else auto_model
                    tokenizers = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf') if tokenizers is None else tokenizers
                    target_model = HuggingfaceModel(auto_model,tokenizers,'llama-2')
                elif model_name == "vicuna":
                    auto_model = AutoModelForCausalLM.from_pretrained('lmsys/vicuna-7b-v1.5', device_map='auto', torch_dtype=torch.bfloat16) if auto_model is None else auto_model
                    tokenizers = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5') if tokenizers is None else tokenizers
                    target_model = HuggingfaceModel(auto_model,tokenizers,'vicuna_v1.1')
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
                defensed_model = RegulatedDefenser(auto_model, tokenizers, "llama-2", safedecoding)
            elif defense_method == "lmfilter-backtranslator":
                defensed_model = Defenser(target_model, 
                                        input_defense_modules=[], 
                                        output_defense_modules=[LMFilter(model=target_model), 
                                                                BackTranslator(model=target_model)])
            elif defense_method == "selfreminder-selfdefense":
                defensed_model = Defenser(target_model, 
                                        input_defense_modules=[SelfReminder()], 
                                        output_defense_modules=[LMFilter(model=target_model)])
            elif defense_method == "selfreminder-safedecoding-selfdefense":
                assert model_name == "llama", "Safe decoding is only supported for llama model."
                regulated_model = RegulatedDefenser(auto_model, tokenizers, "llama-2", safedecoding)
                defensed_model = Defenser(regulated_model, 
                                        input_defense_modules=[SelfReminder()], 
                                        output_defense_modules=[LMFilter(model=target_model)])
            elif defense_method == "async-smoothllm":
                defensed_model = EnsembleDefenser(
                    [Defenser(target_model, 
                            input_defense_modules=
                            [WordsPerturbator(pert_type="RandomSwapPerturbation")], 
                            output_defense_modules=[]),
                    Defenser(target_model, 
                            input_defense_modules=
                            [WordsPerturbator(pert_type="RandomPatchPerturbation")], 
                            output_defense_modules=[]),
                    Defenser(target_model, 
                            input_defense_modules=
                            [WordsPerturbator(pert_type="RandomInsertPerturbation")], 
                            output_defense_modules=[]),         
                            ])
            else:
                raise ValueError(f"Defense method {defense_method} is not supported.")

            # === Attack === #
            if attack_method == "jailbroken":
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
            elif attack_method == "ReNeLLM":
                attacker = ReNeLLM(attack_model=OpenaiModel(model_name='glm-3-turbo', api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto'),
                   target_model=defensed_model,
                   eval_model=eval_model,
                   jailbreak_datasets=dataset)
                attacker.attack()
                attacker.jailbreak_datasets.save_to_jsonl(result_file_path)
            elif attack_method == "AutoDan":
                raise ValueError("Not prepared yet.")
            else:
                raise ValueError(f"Attack method {attack_method} is not supported.")

            # 清理cpu和gpu缓存
            # import torch
            # torch.cuda.empty_cache()
            # import gc
            # gc.collect()
            # print("Done!")
