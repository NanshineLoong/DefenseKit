from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, LlavaNextForConditionalGeneration
import torch
from peft import PeftModel
import torch.nn.functional as F
from typing import Optional, List
from .base_model import BaseModel

from defensekit.utils import OPTION_ID

class HFModel(BaseModel):
    def __init__(self, args, model_class, max_new_tokens=512):
        """
        Initialize the HFModel with a specific model class.

        Args:
            args: Configuration arguments.
            model_class: The class of the model to be instantiated.
        """
        super().__init__()
        self.args = args
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(args.model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = self._load_model(args, model_class)
        if args.use_adapter:
            self.model.load_adapter(args.adapter_dir, adapter_name="expert")

    def _load_model(self, args, model_class):
        """
        Load the model based on the provided model class and arguments.

        Args:
            args: Configuration arguments.
            model_class: The class of the model to be instantiated.

        Returns:
            The instantiated model.
        """
        if args.device == 'auto':
            return model_class.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto',
                use_flash_attention_2=True
            ).eval()
        return model_class.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(int(args.device)).eval()


    def __call__(self, instances):
        inputs = self.prepare_inputs(instances)
        input_length = len(inputs.input_ids[0])
        if instances[0].classification_prompt is None:
            outputs = self.model.generate(**inputs,
                                            max_new_tokens=self.max_new_tokens,
                                            do_sample=False)

            responses = self.extract_response(outputs, input_length)
            for instance, response in zip(instances, responses):
                instance.response = response
        else:
            outputs = self.model.generate(**inputs,
                                            max_new_tokens=2,
                                            do_sample=False,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            responses = self.extract_response(outputs['sequences'], input_length)
            probs = self.extract_probs(outputs['scores'])
            for instance, response, ratio in zip(instances, responses, probs):
                instance.response_options.append(response)
                instance.rejection_probabilities.append(ratio)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    def _generate_conversation(self, instance) -> List[dict]:
        """Generate a conversation for a general question."""
        conversation = []
        if instance.system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": instance.system_prompt}]
            })

        content = instance.classification_prompt.replace('[[query]]', instance.question) if instance.classification_prompt else instance.question
        user_content = []
        if instance.image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": content})
        
        conversation.append({
            "role": "user",
            "content": user_content
        })
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def prepare_inputs(self, instances):
        images = instances.get_attribute('image')
        texts = [self._generate_conversation(instance) for instance in instances]

        if images is None or None in images:
            inputs = self.tokenizer(
                texts, padding=True, return_tensors='pt'
            ).to(self.model.device.index)
        else:
            inputs = self.processor(
                texts, images, padding=True, return_tensors='pt'
            ).to(self.model.device.index)
        return inputs
    
    def extract_response(self, outputs, input_length):
        return self.processor.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

    def extract_probs(self, logits):
        # Get the option indices based on the tokenizer's vocabulary
        option_indices = [self.processor.tokenizer(option).input_ids[-1] for option in OPTION_ID]

        # Decoding the generated sequences to verify correct token mapping
        # decoded_option_indices = [self.processor.tokenizer.decode(option) for option in option_indices]
        # print(decoded_option_indices)

        # Extract logits for the first token that matches the options
        logits_class = logits[1][:, option_indices]
        probs = F.softmax(logits_class, dim=-1).detach().cpu().numpy() 
        probs = probs.tolist()
        return probs
    
    def disable_adapters(self):
        self.model.disable_adapters()
    
    def enable_adapters(self):
        self.model.enable_adapters()

    def __getattr__(self, name):
        return getattr(self.model, name)

class LLaVA(HFModel):
    def __init__(self, args):
        """
        Initialize the LLaVA model.

        Args:
            args: Configuration arguments.
        """
        super().__init__(args, LlavaForConditionalGeneration)
        # self.processor.patch_size = 14
        # self.processor.vision_feature_select_strategy = 'default'
        self.processor.tokenizer.padding_side = 'left'

class Qwen(HFModel):
    def __init__(self, args):
        """
        Initialize the Qwen model.

        Args:
            args: Configuration arguments.
        """
        super().__init__(args, Qwen2VLForConditionalGeneration)


class LLaVA_Next(HFModel):
    def __init__(self, args):
        """
        Initialize the LLaVA_Next model.

        Args:
            args: Configuration arguments.
        """
        super().__init__(args, LlavaNextForConditionalGeneration)
        self.processor.tokenizer.padding_side = 'left'
