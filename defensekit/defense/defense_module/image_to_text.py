class ImageToText:
    def __init__(self, model, intermediate_prompt, rephrased_prompt, delete_image=True):
        self.model = model
        self.intermediate_prompt = intermediate_prompt
        self.rephrased_prompt = rephrased_prompt
        self.delete_image = delete_image

    def __call__(self, instances):
        instances_pre = instances.clone_from_initial()

        for instance_pre in instances_pre:
            instance_pre.question = self.intermediate_prompt.format(instance_pre.question)
        
        self.model(instances_pre)
        
        for instance, instance_pre in zip(instances, instances_pre):
            instance.question = self.rephrased_prompt.format(
                answer=instance_pre.response, 
                question=instance.question)
            if self.delete_image:
                instance.image = None

        return instances