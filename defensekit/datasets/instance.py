from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image

@dataclass
class Instance:
    """
    Represents a data instance for model interaction.

    Attributes:
        category (str): Category of the instance.
        is_safe (bool): Indicates if the instance is safe.
        image_path (Optional[str]): Path to the image associated with the instance.
        image (Optional[Image.Image]): Loaded image object.
        system_prompt (Optional[str]): System prompt text.
        classification_prompt (Optional[str]): Classification prompt template.
        question (str): The main question posed to the model.
        original_question (str): The original unaltered question.
        responses (Optional[str]): The model's response.
        response_options (List[Optional[List[str]]]): List of model responses for classification options.
        is_rejected (Optional[bool]): Indicates if the response was rejected.
        rejection_probabilities (List[float]): List of rejection probabilities.
    """
    category: str
    is_safe: bool
    question: str
    original_question: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    system_prompt: Optional[str] = None
    classification_prompt: Optional[str] = None
    response: Optional[str] = None
    response_options: List[Optional[List[str]]] = field(default_factory=list)
    is_rejected: Optional[bool] = None
    rejection_probabilities: List[float] = field(default_factory=list)
    _initial_state: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.image = Image.open(self.image_path) if self.image_path else None
        # Save original parameters for reset and copying
        self._initial_state = {
            'category': self.category,
            'is_safe': self.is_safe,
            'image_path': self.image_path,
            'image': self.image,
            'question': self.question,
            'original_question': self.original_question,
            'system_prompt': self.system_prompt,
            'classification_prompt': self.classification_prompt,
            'response': self.response,
            'response_options': self.response_options,
            'is_rejected': self.is_rejected,
            'rejection_probabilities': self.rejection_probabilities,
        }

    def clone_from_initial(self) -> 'Instance':
        """Create a new instance based on the initial state."""
        return Instance(**self._initial_state)

    def reset_question(self):
        """Reset the current instance to its original version."""
        self.question = self._initial_state['question']
        self.image = self._initial_state['image']
        self.system_prompt = self._initial_state['system_prompt']
        self.classification_prompt = self._initial_state['classification_prompt']

    def to_dict_for_save(self) -> dict:
        """Return a dictionary with selected instance data for saving."""
        return {
            'category': self.category,
            'is_safe': self.is_safe,
            'image_path': self.image_path,
            'question': self.question,
            'original_question': self.original_question,
            'response': self.response,
            'response_options': self.response_options,
            'is_rejected': self.is_rejected,
            'rejection_probabilities': self.rejection_probabilities,
        }
