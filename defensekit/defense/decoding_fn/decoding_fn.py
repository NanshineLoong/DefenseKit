from defensekit.model import BaseModel

class DecodingFunction:
    def __init__(self, model: BaseModel):
        self.model = model

    def prepare_inputs(self, instances):
        return self.model.prepare_inputs(instances)
    
    def extract_response(self, outputs, input_length):
        return self.model.extract_response(outputs, input_length)
    
    def extract_probs(self, outputs):
        return self.model.extract_probs(outputs)


    