import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = np.finfo(float).eps
    
    def forward(self, prediction_tensor, label_tensor) :
        self.prediction_tensor = prediction_tensor
        loss = -np.sum(np.log(prediction_tensor[label_tensor==1] + self.epsilon))
        return loss

    def backward(self, label_tensor):
        return -1*(label_tensor) / (self.prediction_tensor + self.epsilon)