from .hf_model import HFModel, LLaVA, Qwen, LLaVA_Next
from .openai_model import OpenAIModel
from .base_model import BaseModel

def get_model(args):
    if 'llava-1.5' in args.model_name:
        return LLaVA(args)
    elif 'Qwen' in args.model_name:
        return Qwen(args)
    elif 'llava-v1.6' in args.model_name or 'llava-next' in args.model_name:
        return LLaVA_Next(args)
    else:
        return OpenAIModel(args)
