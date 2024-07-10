import copy


class NeuralNetwork:
    def __init__(self, optimizer , weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss= list()
        self.layers= list()
        self.data_layer=None
        self.loss_layer = None
        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer
       

    def forward(self):
        norm_term= 0
        input_tensor , self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor= layer.forward(input_tensor)
            if layer.trainable:
                if self.optimizer.regularizer:
                    norm_term += self.optimizer.regularizer.norm(layer.weights) 
        
        return self.loss_layer.forward(input_tensor, self.label_tensor) + norm_term

    def backward(self):
        err_tensor = self.loss_layer.backward(self.label_tensor)
        for i in range(len(self.layers)-1, -1, -1):
            err_tensor= self.layers[i].backward(err_tensor)
        return

    def append_layer(self, layer):
        if layer.trainable:
            layer.set_optimizer(copy.deepcopy(self.optimizer))
            layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)
        return

    def train(self, iterations):
        self.iterations=iterations
        for i in range (iterations):
            loss= self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        y_hat = input_tensor
        return y_hat
