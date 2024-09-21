from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def _generate_conversation(self, *args, **kwargs):
        pass
