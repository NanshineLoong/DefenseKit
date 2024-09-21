from .hf_model import HFModel, LLaVA
from .openai_model import OpenAIModel
from .base_model import BaseModel

def get_model(args):
    if 'llava' in args.model_name:
        return LLaVA(args)
    # elif 'qwen' in args.model_name:
    #     return Qwen(args)
    else:
        return OpenAIModel(args)
