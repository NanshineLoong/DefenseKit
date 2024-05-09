from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.Jailbroken_wei_2023 import Jailbroken
from easyjailbreak.attacker.ICA_wei_2023 import ICA
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel

from defensekit.defenser.defenser import Defenser
from defensekit.defensemodule.filter import LMFilter, PPLFilter, EraseAndChecker
from defensekit.defensemodule.refiner import BackTranslator, InContextDefense, WordsPerturbator, SelfReminder, Paraphraser

# First, prepare models and datasets.
attack_model = OpenaiModel(model_name='glm-3-turbo',
                         api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')
target_model = OpenaiModel(model_name='glm-3-turbo',
                         api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')

eval_model = OpenaiModel(model_name='glm-3-turbo',
                         api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')
dataset = JailbreakDataset('AdvBench')[:3]

# Defense
# self_reminder = SelfReminder()
# self_exam = LMFilter(model=eval_model)
# erase_and_checker = EraseAndChecker(model=eval_model) # 有问题
# ppl_filter = PPLFilter()  # 需要GPT-2？
# back_translator = BackTranslator(model=eval_model)
# in_context_defense = InContextDefense()
words_perturbator = WordsPerturbator()
target_model = Defenser(target_model, input_defense_modules=[words_perturbator])

# === ICA === #
attacker = ICA(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

attacker.attack()
attacker.attack_results.save_to_jsonl('AdvBench_ica.jsonl')


# # === PAIR === #
# # Then instantiate the recipe.
# attacker = PAIR(attack_model=attack_model,
#                 target_model=target_model,
#                 eval_model=eval_model,
#                 jailbreak_datasets=dataset)

# # Finally, start jailbreaking.
# attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')


# # === Jailbroken === #
# attacker = Jailbroken(attack_model=attack_model,
#                         target_model=target_model,
#                         eval_model=eval_model,
#                         jailbreak_datasets=dataset)

# attacker.attack()
# attacker.log()
# attacker.attack_results.save_to_jsonl('AdvBench_jailbroken.jsonl')
