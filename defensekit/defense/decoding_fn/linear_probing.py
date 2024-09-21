from defensekit.datasets import Dataset
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from .decoding_fn import DecodingFunction
import logging
import torch

class LinearProbing(DecodingFunction):
    def __init__(self, model, filter_path=None, force_train=False, trainset_path=None, token_id=0, train_batch_size=50):
        super().__init__(model)
        self.token_id = token_id
        self.train_batch_size = train_batch_size
        self.filter_model = self._load_or_train_filter(filter_path, trainset_path, force_train, token_id)

    def __call__(self, instances):
        inputs = self.prepare_inputs(instances)
        input_length = len(inputs.input_ids[0])

        outputs = self.model.generate(**inputs,
                                            max_new_tokens=self.model.max_new_tokens,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
        responses = self.extract_response(outputs['sequences'], input_length)
        logits = outputs['scores']

        X = np.array(logits[self.token_id].cpu())
        rejection_probs = self.filter_model.predict_proba(X)[:, 1]  # Probability of rejection

        del inputs, outputs
        torch.cuda.empty_cache()

        for instance, response, prob in zip(instances, responses, rejection_probs):
            if prob > 0.5:
                instance.is_rejected = True
                instance.response = "Sorry, rejected by <linear probing> defense."
            else:
                instance.is_rejected = False
                instance.response = response
            instance.rejection_probabilities.append(prob)

    def _load_or_train_filter(self, filter_path, trainset_path, force_train, token_id):
        if force_train or not os.path.exists(filter_path):
            return self._train_filter(filter_path, trainset_path, token_id)
        return joblib.load(filter_path)

    def _train_filter(self, filter_path, trainset_path, token_id):
        dataset = Dataset(trainset_path)
        logits, labels = [], []

        logging.info("Training filter model...")
        for instances in tqdm(dataset.batch_iter(self.train_batch_size)):
            inputs = self.prepare_inputs(instances)
            outputs = self.model.generate(**inputs,
                                            max_new_tokens=self.token_id + 1,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            logits.extend(outputs['scores'][token_id].cpu())
            labels.extend([0 if instance.is_safe else 1 for instance in instances])

            # instances.clear_img()
            del inputs, outputs
            torch.cuda.empty_cache()


        filter_model = LogisticRegression()
        filter_model.fit(np.array(logits), labels)
        if not os.path.exists(os.path.dirname(filter_path)):
            os.makedirs(os.path.dirname(filter_path))
        joblib.dump(filter_model, filter_path)
        return filter_model
