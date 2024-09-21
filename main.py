import argparse
from defensekit.datasets import Dataset
from defensekit.model import get_model
from defensekit.defense import Defenser
from defensekit.evaluate import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="defense")
    # evaluated model
    parser.add_argument("--model_name", type=str, default='llava-hf/llava-1.5-7b-hf', help="Model being evaluated")
    parser.add_argument('--use_adapter', action='store_true', help='Use adapter')
    parser.add_argument('--no-use_adapter', dest='use_adapter', action='store_false', help='Do not use adapter')
    parser.add_argument("--adapter_dir", type=str, default='./resources/adapter/llava-7b', help="Adapter directory")
    parser.add_argument("--device", type=str, default='0', help="Device")  # 0, 1, 2, 3, auto

    # dataset
    parser.add_argument("--dataset_path", type=str, default='./datasets/mmsafetybench/train.json', help="Dataset Path")

    # defense
    parser.add_argument("--input_defense_strategy", type=str, default='', help="Input Defense Strategy")
    parser.add_argument("--decoding_strategy", type=str, default='', help="Decoding Strategy")
    parser.add_argument("--output_defense_strategy", type=str, default='', help="Output Defenser Strategy")

    # task
    parser.add_argument("--task", type=str, default='classification',choices=["generation", "classification", "both"], help="Task")  #

    # evaluation
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--judger", type=str, default='KeywordJudger', choices=["KeywordJudger", "LLMJudger"], help="Judger")  # keyword, model
    parser.add_argument("--logging_path", type=str, default='./logs', help="Logging Path")
    parser.add_argument("--output_path", type=str, default='./outputs/output.jsonl', help="Output Path")

    return parser.parse_args()

def main():
    args = parse_args()

    dataset = Dataset(args.dataset_path)  
    
    model = get_model(args)

    defenser = Defenser(model, args.input_defense_strategy, args.output_defense_strategy, args.decoding_strategy)

    evaluator = Evaluator(args, defenser, dataset)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
