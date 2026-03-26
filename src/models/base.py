from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    @abstractmethod
    def predict(self, text_list: List[str]) -> List[List[int]]:
        pass
        
    @abstractmethod
    def save_model(self, path: str):
        pass
        
    @abstractmethod
    def load_model(self, path: str):
        pass
