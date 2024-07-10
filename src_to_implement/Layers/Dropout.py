from Layers.Base import BaseLayer   
import numpy as np
import random

class Dropout(BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability = probability
        self.mask = None
    def forward(self,input_tensor):
        if self.testing_phase is False:
            self.mask = np.random.rand(*input_tensor.shape)<self.probability 
            return input_tensor * self.mask / self.probability
        else:
            return input_tensor

    def backward(self,error_tensor):
         return error_tensor * self.mask / self.probability