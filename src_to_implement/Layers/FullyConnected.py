from Layers.Base import BaseLayer
import numpy as np 
from Optimization.Optimizers import Sgd
from Layers.Initializers import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random_sample((input_size+1, output_size))
        self._optimizer = None
        self.input_tensor = None
        self.gradient_weights = None

    def forward(self, input_tensor):
        bias = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate( (input_tensor, bias), axis=1 )
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor
    
    def backward(self, error_tensor):
        self.gradient_weights  = np.dot(self.input_tensor.T, error_tensor)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        
        return np.dot(error_tensor, self.weights.T)[:, :-1]
    
    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        bias = bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)
        self.weights = np.concatenate((weights,bias),axis=0)

        

    '''@property 
    def optimizer(self):
        return self.optimizer
    
    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer'''
    
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)
