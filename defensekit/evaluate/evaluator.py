from .judger import * 
from tqdm import tqdm
from defensekit.utils import setup_logger, PROMPT_FOR_CLASSIFICATION, OPTIONS
import logging

class Evaluator:
    """Evaluator class for performing model evaluations."""
    
    def __init__(self, args, defenser, dataset):
        self.defenser = defenser
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.output_path = args.output_path
        self.task = args.task
        self.judger = globals().get(args.judger)()
        
        setup_logger(args.logging_path)
        # save args items to log
        logging.info("Arguments:")
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")

    def evaluate(self):
        """Main evaluation loop."""
        for batch_instances in tqdm(self.dataset.batch_iter(self.batch_size)):
            if self.has_classification_task():
                self.perform_classification_task(batch_instances)

            if self.has_generation_task():
                self.perform_generation_task(batch_instances)

        self.report()
        self.dataset.save_to_jsonl(self.output_path)

    def has_generation_task(self):
        """Check if the task includes generation."""
        return self.task != 'classification'
    
    def has_classification_task(self):
        """Check if the task includes classification."""
        return self.task != 'generation'

    def perform_generation_task(self, instances):
        """Perform generation-related tasks."""
        instances.reset_question()
        self.defenser(instances)
        self.judger(instances)
    
    def perform_classification_task(self, instances):
        """Perform classification-related tasks."""
        for options in OPTIONS:
            instances.reset_question()
            prompt_cls = PROMPT_FOR_CLASSIFICATION.replace("[[option0]]", options[0])
            prompt_cls = prompt_cls.replace("[[option1]]", options[1])

            # Assign classification prompt to each instance
            instances.update_attribute('classification_prompt', prompt_cls)
            
            self.defenser(instances)

    def report(self):
        # 生成相关的结果并保存到日志中
        pass