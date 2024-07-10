from Layers.Base import BaseLayer
import numpy as np 

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0,input_tensor)
    

    def  backward(self, error_tensor):
        gradiant = (self.input_tensor >= 0)
        return np.multiply(gradiant,error_tensor) 

    
