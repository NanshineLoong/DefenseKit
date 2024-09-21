from .decoding_fn import DecodingFunction
import torch
import copy
import logging
from torch import nn

class SafeDecoding(DecodingFunction):
    def __init__(self, model, first_m=5, top_k = 10, alpha=1, do_sample=False, top_p=None, num_common_tokens=3):
        '''
        Args:
            model: The model to be used for decoding.
            first_m: The number of tokens to be generated using the safedecoding method.
            top_k: The number of top tokens to be considered for safedecoding.
            alpha: The weight to be given to the probability diff.
            do_sample: Whether to sample from the top-p tokens for safedecoding.
            top_p: The probability threshold for top-p sampling with do_sample=True.
            num_common_tokens: The number of common tokens to be considered for safedecoding.'''
        super().__init__(model)
        assert "expert" in model.peft_config, "Model must have adapter 'expert'."

        self.first_m = first_m
        self.top_k = top_k
        self.alpha = alpha
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_common_tokens = num_common_tokens

    def __call__(self, instances):
        inputs = self.prepare_inputs(instances)
        input_length = len(inputs.input_ids[0])

        is_cls = instances[0].classification_prompt is not None
        max_new_tokens = self.model.max_new_tokens if not is_cls else 2

        logits = []

        step = 1  # Keep track of generation steps
        while step <= min(max_new_tokens, self.first_m):  # Loop until we reach the first m tokens
            output_expert = self.model.generate(**inputs,
                                            max_new_tokens=1,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            self.model.disable_adapters()
            output_base = self.model.generate(**inputs,
                                            max_new_tokens=1,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            self.model.enable_adapters()

            scores_base = output_base.scores[-1]
            scores_expert = output_expert.scores[-1]

            if is_cls:
                score_diff = scores_expert - scores_base
                updated_score = scores_base + self.alpha * score_diff
                logits.append(updated_score)

            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
            scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
            sorted_indices_base = torch.argsort(scores_base, descending=True)
            sorted_indices_expert = torch.argsort(scores_expert, descending=True)

            # Step 1: Define Sample Space
            intersection_indices = self.apply_along_axis(
                self.get_intersection_indices, 0, sorted_indices_base, sorted_indices_expert
            )
            
            # Step 2: New Probability Calculation
            sorted_token_ids = self.get_updated_sorted_indices(intersection_indices, scores_base, scores_expert)

            selected_token_id = sorted_token_ids[:, 0].unsqueeze(1)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id], dim=1)

            step += 1
            # Free up memory
            del output_base, output_expert
            torch.cuda.empty_cache()


        # Use the normal model to generate the rest of the tokens
        remaining_steps = max_new_tokens - min(max_new_tokens, self.first_m)

        if is_cls:
            if remaining_steps <= 0:
                probs = self.extract_probs(logits)
                responses = self.extract_response(inputs['input_ids'], input_length)
                for instance, response, ratio in zip(instances, responses, probs):
                    instance.response_options.append(response)
                    instance.rejection_probabilities.append(ratio)
            else:
                inputs_dict = {
                    'input_ids': inputs['input_ids'],
                }
                if 'pixel_values' in inputs:
                    inputs_dict['pixel_values'] = inputs['pixel_values']
                output_base = self.model.generate(
                                        **inputs_dict,
                                        max_new_tokens=remaining_steps,
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        output_scores=True)
                probs = self.extract_probs(logits + list(output_base.scores))
                responses = self.extract_response(output_base['sequences'], input_length)
                for instance, response, ratio in zip(instances, responses, probs):
                    instance.response_options.append(response)
                    instance.rejection_probabilities.append(ratio)
        else:
            if remaining_steps <= 0:
                instance.response = self.extract_response(inputs['input_ids'], input_length)
            inputs_dict = {
                'input_ids': inputs['input_ids'],
            }
            if 'pixel_values' in inputs:
                inputs_dict['pixel_values'] = inputs['pixel_values']

            self.model.disable_adapters()
            output_base = self.model.generate(
                                            **inputs_dict,
                                            max_new_tokens=remaining_steps,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            self.model.enable_adapters()

            responses = self.extract_response(output_base['sequences'], input_length)
            for instance, response in zip(instances, responses):
                instance.response = response

        del inputs, output_base
        torch.cuda.empty_cache()

    def apply_along_axis(self, function, axis, *args):
        # 将所有输入张量沿指定的轴展开
        unbound_tensors = [torch.unbind(arg, dim=axis) for arg in args]
        
        # 使用 zip 组合展开的张量，并对每组应用给定的函数
        tensors = [function(*tensors) for tensors in zip(*unbound_tensors)]
        result = torch.stack(tensors, dim=axis)
        
        return result
    
    def get_intersection_indices(self, sorted_indices_base, sorted_indices_expert):
        common_tokens = set()
        iter_range = self.num_common_tokens
        while len(common_tokens) < self.num_common_tokens:
            current_indices_base = sorted_indices_base[:iter_range]
            current_indices_expert = sorted_indices_expert[:iter_range]

            common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
            common_tokens.update(common_in_iteration)

            iter_range += 1

            if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                break
        # if len(common_tokens) > 3:
        #     return torch.tensor([sorted_indices_base[0]], device=self.model.device)
        return torch.tensor(list(common_tokens)[:self.num_common_tokens], device=self.model.device)
    
    def get_updated_sorted_indices(self, intersection_indices, scores_base, scores_expert):
        # Steer probabilities using vectorized operations
        scores_base_exp = torch.exp(torch.gather(scores_base, dim=1, index=intersection_indices))
        scores_expert_exp = torch.exp(torch.gather(scores_expert, dim=1, index=intersection_indices))
        
        prob_diff = scores_expert_exp - scores_base_exp
        updated_prob = scores_base_exp + self.alpha * prob_diff
        
        # Floor the probabilities to avoid log(0)
        updated_prob = torch.clamp(updated_prob, min=1e-8)
        
        # Convert back to scores
        updated_scores = torch.log(updated_prob)
        
        # Normalize the scores using softmax
        normalized_probs = torch.nn.functional.softmax(updated_scores, dim=-1)
        
        # Sort indices based on the normalized probabilities
        sorted_indices = torch.argsort(normalized_probs, descending=True)
        
        # Return the sorted token IDs
        return torch.gather(intersection_indices, dim=1, index=sorted_indices)
    
    # Display the top tokens
    def display(self, scores_base, scores_expert):
        step = 1
        topk_scores_base, topk_indices_base = scores_base.topk(self.top_k) 
        topk_scores_expert, topk_indices_expert = scores_expert.topk(self.top_k) 
        logging.info("\n-----------------------------------------------")
        logging.info(f"Generation Step {step}")
        logging.info("Original Model")
        logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
        logging.info("|----|----------|---------|----------|---------|")
        for idx, (score, token_id) in enumerate(zip(topk_scores_base[0], topk_indices_base[0])):
            token = self.model.tokenizer.decode(token_id.item())
            prob = torch.exp(score)
            logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

        logging.info("Expert Model")
        logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
        logging.info("|----|----------|---------|----------|---------|")
        for idx, (score, token_id) in enumerate(zip(topk_scores_expert[0], topk_indices_expert[0])):
            token = self.model.tokenizer.decode(token_id.item())
            prob = torch.exp(score)
            logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

        