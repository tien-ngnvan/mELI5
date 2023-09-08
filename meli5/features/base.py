from abc import ABC, abstractclassmethod

class BaseDataProcess(ABC):   
    @abstractclassmethod
    def process(self):
        pass
    
    @abstractclassmethod
    def run(self):
        pass
    
    @abstractclassmethod
    def save(self):
        pass
    