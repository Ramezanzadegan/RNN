import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.shape = input_tensor.shape 
        input_tensor =  np.reshape(input_tensor, (input_tensor.shape[0],np.prod(self.shape[1:])))
        return input_tensor
    
    def backward(self, error_tensor):
        
        return  np.reshape(error_tensor, self.input_tensor.shape)

        