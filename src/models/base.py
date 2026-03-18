from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    @abstractmethod
    def predict(self, text_list: List[str]) -> List[List[int]]:
        """
        Returns a list of binary lists representing the 6 emotions.
        E.g., [[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
        """
        pass
        
    @abstractmethod
    def save_model(self, path: str):
        pass
        
    @abstractmethod
    def load_model(self, path: str):
        pass
