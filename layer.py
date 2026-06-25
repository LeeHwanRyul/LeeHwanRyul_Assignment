import numpy as np
from utility import Utility 

class Layer:
    class Relu:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0

            return out

        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout

            return dx


    class Sigmoid:
        def __init__(self):
            self.out = None
            self.util = Utility()

        def forward(self, x):
            out = self.util.sigmoid(x)
            self.out = out
            return out

        def backward(self, dout):
            dx = dout * (1.0 - self.out) * self.out

            return dx


    class Affine:
        def __init__(self, w, b):
            self.w = w
            self.b = b
            
            self.x = None
            self.original_x_shape = None

            self.dw = None
            self.db = None

        def forward(self, x):
            # 
            self.original_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x

            out = np.dot(self.x, self.w) + self.b

            return out

        def backward(self, dout):
            dx = np.dot(dout, self.w.T)
            self.dw = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)
            
            dx = dx.reshape(*self.original_x_shape)  
            return dx


    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None 
            self.y_hat = None    
            self.y = None    
            self.util = Utility()
            
        def forward(self, x, y):
            self.y = y
            self.y_hat = self.util.softmax(x)
            self.loss = self.util.cross_entropy_error_batch(self.y_hat, self.y)
            
            return self.loss

        def backward(self, dout=1):
            batch_size = self.y.shape[0]
            #if self.y.size == self.y_hat.size: # one hot encoding
            
            dx = (self.y_hat - self.y) / batch_size
            
            """
            else:
                dx = self.y_hat.copy()
                dx[np.arange(batch_size), self.t] -= 1
                dx = dx / batch_size
            """
            return dx