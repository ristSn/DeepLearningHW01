from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        W = initialize_method(size=(in_dim, out_dim))
        self.W = W / np.sqrt(np.sum(W**2))
        b = initialize_method(size=(1, out_dim))
        self.b = b / np.sqrt(np.sum(b**2))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        W = self.params['W']
        b = self.params['b']
        output = np.dot(X, W) + b
        return output


    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        grad_W = np.dot(self.input.T, grad) / batch_size
        grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size  # 列平均

        W = self.params['W']

        if self.weight_decay:
            grad_W += self.weight_decay_lambda * W

        self.grads['W'] = grad_W
        self.grads['b'] = grad_b

        grad_input = np.dot(grad, self.W.T)
        return grad_input

    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.W = W / np.sqrt(np.sum(W**2))
        self.b = np.zeros((out_channels, 1, 1))
        self.input = None
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k] -- 这里应该错了，应该是[out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        out_channels, _, kernel_size, _ = self.W.shape
        new_H = (H + 2*self.padding - kernel_size) // self.stride + 1
        new_W = (W + 2*self.padding - kernel_size) // self.stride + 1

        W = self.params['W']
        b = self.params['b']

        output = np.zeros((batch_size, out_channels, new_H, new_W))
        for i in range(new_H):
            for j in range(new_W):
                for k in range(out_channels):
                    h_start = i * self.stride - self.padding
                    h_end = h_start + kernel_size
                    w_start = j * self.stride - self.padding
                    w_end = w_start + kernel_size
                    X_slice = X[:, :, h_start:h_end, w_start:w_end]
                    output[:, k, i, j] = np.sum(X_slice * W[k], axis=(1, 2, 3)) + b[k]

        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, out_channels, new_H, new_W = grads.shape
        _, in_channels, kernel_size, _ = self.W.shape
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_input = np.zeros_like(self.input)

        W = self.params['W']


        # for i in range(new_H):
        #     for j in range(new_W):
        #         h_start = i * self.stride - self.padding
        #         h_end = h_start + kernel_size
        #         w_start = j * self.stride - self.padding
        #         w_end = w_start + kernel_size
        #         X_slice = self.input[:, :, h_start:h_end, w_start:w_end]
        #
        #         grad_X_slice = np.dot(grads[:, :, i, j].reshape(-1, out_channels), W.reshape(out_channels, -1))
        #         grad_X_slice = grad_X_slice.reshape(batch_size, in_channels, kernel_size, kernel_size)
        #         grad_input[:, :, h_start:h_end, w_start:w_end] += grad_X_slice
        #
        #         grad_W += np.dot(grads[:, :, i, j].reshape(-1, out_channels).T, X_slice.reshape(batch_size, -1)).reshape(out_channels, in_channels, kernel_size, kernel_size)
        #         grad_b += np.sum(grads[:, :, i, j], axis=0, keepdims=True)

        for i in range(new_H):
            for j in range(new_W):
                for k in range(out_channels):
                    h_start = i * self.stride - self.padding
                    h_end = h_start + kernel_size
                    w_start = j * self.stride - self.padding
                    w_end = w_start + kernel_size
                    X_slice = self.input[:, :, h_start:h_end, w_start:w_end]

                    grad_X_slice = np.dot(grads[:, k, i, j].reshape(-1, 1), W[k].reshape(1, -1))
                    grad_X_slice = grad_X_slice.reshape(batch_size, in_channels, kernel_size, kernel_size)
                    grad_input[:, :, h_start:h_end, w_start:w_end] += grad_X_slice

                    grad_W[k:k+1] += np.dot(grads[:, k, i, j].reshape(-1, 1).T, X_slice.reshape(batch_size, -1)).reshape(1, in_channels, kernel_size, kernel_size)
                    grad_b[k:k+1] += np.sum(grads[:, k, i, j], axis=0, keepdims=True)

        if self.weight_decay:
            grad_W += self.weight_decay_lambda * W

        self.grads['W'] = grad_W
        self.grads['b'] = grad_b

        return grad_input

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.input = None
        self.labels = None
        self.grads = None


    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.input = predicts
        self.labels = labels
        if self.has_softmax:
            predicts = softmax(predicts)
        else:
            predicts = predicts
        loss = -np.log(predicts[np.arange(labels.shape[0]), labels])
        return np.mean(loss)
    
    def backward(self):
        # first compute the grads from the loss to the input
        if self.has_softmax:
            predicts = softmax(self.input)
            grads = predicts.copy()
            grads[np.arange(self.labels.shape[0]), self.labels] -= 1
            grads /= self.labels.shape[0]
        else: # 没有使用softmax的情况下也使用softmax的方式计算梯度
            predicts = softmax(self.input)
            grads = predicts.copy()
            grads[np.arange(self.labels.shape[0]), self.labels] -= 1
            grads /= self.labels.shape[0]

        self.grads = grads

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass

class MaxPool2D(Layer):
    """
    A max pooling layer.
    """
    def __init__(self, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.grads = None
        self.channels = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, channels, H, W]
        output: [batch_size, channels, new_H, new_W]
        """
        self.input = X
        batch_size, channels, H, W = X.shape
        self.channels = channels
        new_H = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        new_W = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, channels, new_H, new_W))
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_start = j * self.stride - self.padding
                w_end = w_start + self.kernel_size
                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(X_slice, axis=(2, 3))

        return output

    def backward(self, grads):
        """
        grads: [batch_size, channels, new_H, new_W]
        """
        batch_size, channels, new_H, new_W = grads.shape
        _, _, H, W = self.input.shape
        grad_input = np.zeros_like(self.input)

        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride - self.padding
                h_end = h_start + self.kernel_size
                w_start = j * self.stride - self.padding
                w_end = w_start + self.kernel_size

                X_slice = self.input[:, :, h_start:h_end, w_start:w_end]
                X_slice_grad = np.zeros_like(X_slice)
                X_slice_grad[:, :, :, :] = (X_slice == np.max(X_slice, axis=(2, 3), keepdims=True)) * grads[:, :, i:i+1, j:j+1]
                grad_input[:, :, h_start:h_end, w_start:w_end] += X_slice_grad

        self.grads = grad_input
        return grad_input

    def clear_grad(self):
        self.grads = None

class Flatten(Layer):
    """
    A flatten layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.grads = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, channels, H, W]
        output: [batch_size, channels*H*W]
        """
        self.input = X
        batch_size, channels, H, W = X.shape
        output = X.reshape(batch_size, channels*H*W)
        return output

    def backward(self, grads):
        """
        grads: [batch_size, channels*H*W]
        """
        batch_size, channels, H, W = self.input.shape
        grad_input = grads.reshape(batch_size, channels, H, W)
        self.grads = grad_input
        return grad_input

    def clear_grad(self):
        self.grads = None

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition