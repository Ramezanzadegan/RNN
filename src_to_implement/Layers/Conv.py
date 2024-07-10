from Layers.Base import BaseLayer   
import numpy as np
from scipy import signal
from Optimization.Optimizers import *
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels ):
        super().__init__()
        self.trainable = True


        #can be a single value or a tuple. The latter allows for diﬀerent strides in the spatial dimensions
        self.stride_shape = stride_shape
        #print("s shape -- ", stride_shape)

        #determines whether this object provides a 1D or a 2D con-
        #volution layer. For 1D, it has the shape [c, m], whereas for 2D, it has the shape
        #[c, m, n], where c represents the number of input channels, and m, n represent the
        #spatial extent of the ﬁlter kernel.
        self.convolution_shape = convolution_shape
        #print("c shape -- ", convolution_shape)

        #is an integer value.   
        self.num_kernels = num_kernels
        #print("num_kernels",num_kernels)

        self.weights= np.random.uniform(0,1,(num_kernels, *convolution_shape))
        #print("w -- ", self.weights.shape)
        self.bias=np.random.uniform(0,1,num_kernels)

       

        self.bias_optimizer=None
        self.weights_optimizer=None
        
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        #print(input_tensor.shape)
        if (len(input_tensor.shape)==3):       # 1D
            batches,channels,height = input_tensor.shape #(batch,channels,hieght)

            output = np.zeros((batches,self.num_kernels,height))
            for batch in range(batches):
                for each_kernel in range(self.num_kernels):
                    output_producer = np.zeros_like(output[batch,each_kernel])
                    for channel in range(channels):
                        output_producer += signal.correlate(input_tensor[batch,channel], self.weights[each_kernel,channel], mode = "same")
                        
                    output[batch,each_kernel] = output_producer
                    output[batch,each_kernel] = output[batch,each_kernel] + self.bias[each_kernel]
            self.up_shape= output.shape
            output = output[:,:,::self.stride_shape[0]]
        elif (len(input_tensor.shape)==4):      # 2D
            batches,channels,height,width = input_tensor.shape  
            output = np.zeros((batches,self.num_kernels,height,width))
            for batch in range(batches):
                for each_kernel in range(self.num_kernels):
                    output_producer = np.zeros_like(output[batch,each_kernel])
                    for channel in range(channels):
                        output_producer += signal.correlate(input_tensor[batch,channel], self.weights[each_kernel,channel], mode = "same")

                    output[batch, each_kernel] = output_producer
                    output[batch, each_kernel] = output[batch,each_kernel] + self.bias[each_kernel]
            self.up_shape= output.shape
            output = output[:,:, ::self.stride_shape[0], ::self.stride_shape[1]]
        return output
    
    def backward(self,error_tensor):
        """self.error_tensor = error_tensor
        self._gradient_weights = np.zeros((self.weights).shape)
        self._gradient_bias = np.zeros(error_tensor.shape)
        input_gradiants = np.zeros(self.input_tensor.shape)

        if (len(error_tensor.shape)==3):
            batches, num_of_kernels, height = error_tensor.shape
            batches,channels,height = self.input_tensor.shape
            for batch in range(batches):
                for kernel in range(num_of_kernels):
                    for channel in range(channels) :
                        self._gradient_weights[kernel,channel] = signal.correlate(self.input_tensor[batch,channel],self.error_tensor[batch,kernel], mode="same")
                        input_gradiants[batch,channel] += signal.convolve(self.error_tensor[batch,kernel],self.gradient_weights[kernel,channel], "same")
            self._gradient_bias = np.sum(error_tensor, axis=(0,2))

        elif (len(error_tensor.shape) == 4):
            batches, num_of_kernels, height,width = error_tensor.shape
            batches,channels,heigh,width = self.input_tensor.shape
            for batch in range(batches):
                for kernel in range(num_of_kernels):
                    for channel in range(channels) :    
                        self._gradient_weights[kernel,channel] = signal.correlate(self.input_tensor[batch,channel],self.error_tensor[batch,kernel], mode="same")
                        input_gradiants[batch,channel] += signal.convolve(self.error_tensor[batch,kernel],self.gradient_weights[kernel,channel], "same")
                        
            self._gradient_bias = np.sum(error_tensor, axis=(0,2,3))

        if self.bias_optimizer is not None:
            opt= self.bias_optimizer
            self.bias= opt.calculate_update(self.bias, self._gradient_bias)

        if self.weights_optimizer is not None:
            opt= self.weights_optimizer
            self.weights= opt.calculate_update(self.weights, self._gradient_weights)

        return input_gradiants"""
        back_out= np.zeros_like(self.input_tensor)
        px,py= self.calculate_padding()
        err_plane= np.zeros(self.up_shape)
        new_weights= np.swapaxes(self.weights, axis1=0, axis2=1)

        if (len(error_tensor.shape)==3):                    #1D
            err_plane[:,:,::self.stride_shape[0]]=error_tensor
            pad_err_tensor= np.pad(err_plane,((0,0), (0,0), py))
            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel= new_weights[j]
                    back_out[i,j]= signal.convolve(pad_err_tensor[i], new_kernel, mode='valid')
            
            self._gradient_weights=np.zeros_like(self.weights)
            padded_input= np.pad(self.input_tensor, ((0,0), (0,0), py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j,k]+=signal.correlate(padded_input[i,k], err_plane[i,j], mode="valid")
            
            self._gradient_bias=np.sum(error_tensor, axis=(0,2))

                                              
        else:                                         #2D       
            err_plane[:,:,::self.stride_shape[0],::self.stride_shape[1]]=error_tensor
            pad_err_tensor= np.pad(err_plane,((0,0),(0,0), px, py))
            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel = np.flip(new_weights[j], axis=0)
                    back_out[i,j]= signal.convolve(pad_err_tensor[i], new_kernel, mode='valid') 

            self._gradient_weights= np.zeros_like(self.weights)
            padded_input= np.pad(self.input_tensor,((0,0), (0,0), px, py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, k ]+= signal.correlate(padded_input[i,k] ,  err_plane[i,j], mode='valid')

            self._gradient_bias= np.sum( error_tensor, axis=(0,2,3))
        
        if self.bias_optimizer is not None:
            opt= self.bias_optimizer
            self.bias= opt.calculate_update(self.bias, self._gradient_bias)

        if self.weights_optimizer is not None:
            opt= self.weights_optimizer
            self.weights= opt.calculate_update(self.weights, self._gradient_weights)


        return back_out
    

    def set_optimizer(self, optimizer):
        self.weights_optimizer=optimizer
        self.bias_optimizer=copy.deepcopy(optimizer)

    def get_optimizer(self):
        return self.weights_optimizer, self.bias_optimizer
    
    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    optimizer = property(get_optimizer,set_optimizer)
    gradient_weights= property(get_gradient_weights)
    gradient_bias= property(get_gradient_bias)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights=weights_initializer.initialize(self.weights.shape , np.prod(self.convolution_shape), self.num_kernels*np.prod(self.convolution_shape[1:]) )
        self.bias=bias_initializer.initialize(self.bias.shape, 0, 0)
    
    def calculate_padding(self):
        if (len(self.input_tensor.shape)==3):
            px_1=0
            px_2=0
            #c,y
            py_1= self.convolution_shape[1]//2
            py_2= py_1
            if self.convolution_shape[1]%2==0:
                py_2-=1
            pass
        else:
            #c,x,y
            px_1 = self.convolution_shape[1]//2
            px_2 = px_1
            if self.convolution_shape[1]%2==0:
                px_2-=1

            py_1= self.convolution_shape[2]//2
            py_2= py_1
            if self.convolution_shape[2]%2==0:
                py_2-=1

        return (px_1,px_2),(py_1,py_2)

    
    

    
        





            



        