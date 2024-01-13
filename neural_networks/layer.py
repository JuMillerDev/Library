from abc import ABC,abstractmethod

#abstract base class for different kinds of layers
class LayerInterface(ABC):
    @abstractmethod
    def __init__(self):
        self.input = None
        self.output = None
      
    @abstractmethod    
    def forward_propagation(self, input):
        pass
    
    @abstractmethod
    def backward_propagation(self, output_gradient, learning_rate):
        pass