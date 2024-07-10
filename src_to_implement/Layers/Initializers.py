import numpy as np
class Constant:
    def __init__(self,constant_value=None):
        self.constant_value = constant_value
        if constant_value is None:
            self.constant_value = .1
    def initialize(self,weights_shape,fan_in, fan_out):
        initialized_tensor = np.full(weights_shape,self.constant_value)
        return initialized_tensor

class UniformRandom:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in=None, fan_out=None):
        initialized_tensor = np.random.random_sample(weights_shape)
        return initialized_tensor



class Xavier:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out ))
        initialized_tensor = np.random.normal(0,sigma,weights_shape)
        return initialized_tensor



class He:
    def __init__(self):
        pass
    def initialize(self,weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        initialized_tensor = np.random.normal(0,sigma,weights_shape)
        return initialized_tensor
