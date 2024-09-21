import json
import os
from typing import List, Optional, Union
from .instance import Instance

class Dataset:
    def __init__(self, data: Union[str, List[Instance]]):
        if isinstance(data, str):
            self.instances = self._load_from_json(data)
        elif isinstance(data, list) and all(isinstance(i, Instance) for i in data):
            self.instances = data
        else:
            raise ValueError("Input must be either a file path (str) or a list of Instance objects.")

    def _load_from_json(self, path: str) -> List[Instance]:
        with open(path, 'r') as file:
            data = json.load(file)
        return [Instance(**item) for item in data]
        # return [Instance(**item) for item in (data[:10] + data[-10:])]

    def save_to_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for instance in self.instances:
                json.dump(instance.to_dict_for_save(), file)
                file.write('\n')

    def batch_iter(self, batch_size: int):
        for i in range(0, len(self.instances), batch_size):
            yield Dataset(self.instances[i:i + batch_size])

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def __iter__(self):
        return iter(self.instances)

    def filter_unrejected(self) -> 'Dataset':
        """Return a new Dataset with only unrejected instances."""
        return Dataset([instance for instance in self.instances if not instance.is_rejected])

    def clone_from_initial(self) -> 'Dataset':
        """Create a new Dataset based on the initial state of each instance."""
        return Dataset([instance.clone_from_initial() for instance in self.instances])

    def update_attribute(self, name: str, value):
        """Update a specific attribute for all instances."""
        for instance in self.instances:
            setattr(instance, name, value)
    
    def get_attribute(self, name: str):
        """Get a specific attribute from all instances."""
        return [getattr(instance, name) for instance in self.instances]
    
    def reset_question(self):
        """Reset all instances to their original versions."""
        for instance in self.instances:
            instance.reset_question()

    # def clear_img(self):
    #     """Clear the image attribute for all instances."""
    #     for instance in self.instances:
    #         instance.image = None
