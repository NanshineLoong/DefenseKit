from .defense_module import *
from .decoding_fn import *
from defensekit.utils.constants import DefenseModuleInfo

class Defenser:
    def __init__(self, model, input_modules=None, output_modules=None, decoding_fn=None):
        self.model = model
        self.input_modules = self._initialize_modules(input_modules) if input_modules else []
        self.output_modules = self._initialize_modules(output_modules) if output_modules else []
        self.decoding_fn = self._initialize_module(decoding_fn) if decoding_fn else model

    def _initialize_module(self, module_name):
        """Initialize a single module or decoding function."""
        module_info = DefenseModuleInfo.get(module_name)
        if module_info:
            module_class = globals().get(module_info['module'])
            if module_class:
                params = self._prepare_module_params(module_info['params'])
                return module_class(**params)
            else:
                raise ValueError(f"Module class '{module_name}' not found.")
        
        raise ValueError(f"ModuleInfo for '{module_name}' not found.")

    def _initialize_modules(self, module_names):
        """Initialize multiple modules by splitting comma-separated names."""
        return [self._initialize_module(name.strip()) for name in module_names.split(',')]

    def _get_module_class(self, class_name):
        return globals().get(class_name)

    def _prepare_module_params(self, params):
        if 'model' in params and params['model'] == 'self':
            params['model'] = self.model
        return params

    def __call__(self, instances):
        for module in self.input_modules:
            module(instances.filter_unrejected())

        self.decoding_fn(instances.filter_unrejected())

        for module in self.output_modules:
            module(instances.filter_unrejected())
