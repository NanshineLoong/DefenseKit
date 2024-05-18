from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.datasets import JailbreakDataset
import os

defense_methods = ["selfreminder-safedecoding-selfdefense"]
model_names = ["llama"]
attack_methods = ["CodeChameleon"]

for model_name in model_names:
    for attack_method in attack_methods:
        for defense_method in defense_methods:
            print(f"Attack: {attack_method}, Model: {model_name}, Defense: {defense_method}")
            file_path = f"AdvBench_{attack_method}_{defense_method}_{model_name}.jsonl"

            if not os.path.exists("results/" + file_path):
                print("File not found: ", file_path)
                continue

            if  os.path.exists("evaluate/" + file_path):
                print("Already evaluated: ", file_path)
                continue

            # read jsonl file
            attack_results = JailbreakDataset.load_jsonl("results/" + file_path)

            # evaluator
            eval_model = OpenaiModel(model_name='glm-4', api_keys='a3f0ac96680ea6732fb338792dc49300.sT499r9X5SBTVxto')
            evaluator = EvaluatorGenerativeJudge(eval_model)

            # evaluate
            print("Evaluating...")
            evaluator(attack_results)
            attack_results.save_to_jsonl("evaluate/" + file_path)

            # print
            current_jailbreak = 0
            current_query = 0
            current_reject = 0
            for result in attack_results:
                current_jailbreak += result.num_jailbreak
                current_query += result.num_query
                current_reject += result.num_reject

            print("======Jailbreak report:======")
            print(f"Total queries: {current_query}")
            print(f"Total jailbreak: {current_jailbreak}")
            print(f"Total reject: {current_reject}")
            print("========Report End===========")
