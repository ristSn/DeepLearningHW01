from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, lambda_list=None, ks_list=None):
        # size_list should be [(H1, W1), (H2, W2),..., (H_n, W_n)]
        # layers should be [conv2D, Linear, conv2D, Linear,..., conv2D, Linear]
        self.layers = []
        self.size_list = size_list
        self.lambda_list = lambda_list
        self.ks_list = ks_list
        if size_list is not None:
            for i in range(len(size_list)-1):
                h, w = size_list[i]
                h_new, w_new = size_list[i+1]
                ks = self.ks_list[i] if self.ks_list is not None else 3
                layer = conv2D(in_channels=1, out_channels=1, kernel_size=ks, padding=0)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]

                self.layers.append(layer)
                h_c = (h + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                w_c = (w + 2*layer.padding - layer.kernel_size) // layer.stride + 1

                layer_f = Linear(in_dim=h_c*w_c, out_dim=h_new*w_new)
                self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # input X : [batch_size, channel, height, width]
        # conv out X: [batch_size, out_channel, height, width]
        # linear out X: [batch_size, out_dim]
        outputs = X
        batch_size = X.shape[0]
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            if i % 2 == 1: # 线性层输入前后需要reshape
                outputs = outputs.reshape(batch_size, -1)
                outputs = lyr(outputs)
                h_new, w_new = self.size_list[(i+1)//2]
                outputs = outputs.reshape(batch_size, 1, h_new, w_new)
            else: # 卷积层
                outputs = lyr(outputs)
        # 最后整理成[batch_size, max_classes]
        outputs = outputs.reshape(batch_size, -1)
        return outputs


    def backward(self, loss_grad):
        grads = loss_grad
        for i in range(len(self.layers)-1, -1, -1):
            lyr = self.layers[i]
            if i % 2 == 1: # 线性层直接反向传播
                grads = lyr.backward(grads)
            else: # 卷积层反向传播前后需要reshape
                h_old, w_old = self.size_list[i//2]
                h_c = (h_old + 2*lyr.padding - lyr.kernel_size) // lyr.stride + 1
                w_c = (w_old + 2*lyr.padding - lyr.kernel_size) // lyr.stride + 1
                grads = grads.reshape(grads.shape[0], 1, h_c, w_c)
                grads = lyr.backward(grads)
                grads = grads.reshape(grads.shape[0], -1)
        # 最后整理成[batch_size, channel, height, width]
        h0, w0 = self.size_list[0]
        grads = grads.reshape(grads.shape[0], 1, h0, w0)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.lambda_list = param_list[1]
        self.ks_list = param_list[2]
        bias = 3
        self.layers = []
        for i in range(len(self.size_list)-1):
            h, w = self.size_list[i]
            h_new, w_new = self.size_list[i+1]
            ks = self.ks_list[i] if self.ks_list is not None else 3
            layer = conv2D(in_channels=1, out_channels=1, kernel_size=ks, padding=0)
            layer.W = param_list[2*i+bias]['W']
            layer.b = param_list[2*i+bias]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[2*i+bias]['weight_decay']
            layer.weight_decay_lambda = param_list[2*i+bias]['lambda']
            self.layers.append(layer)

            h_c = (h + 2*layer.padding - layer.kernel_size) // layer.stride + 1
            w_c = (w + 2*layer.padding - layer.kernel_size) // layer.stride + 1

            layer_f = Linear(in_dim=h_c*w_c, out_dim=h_new*w_new)
            layer_f.W = param_list[2*i+bias+1]['W']
            layer_f.b = param_list[2*i+bias+1]['b']
            layer_f.params['W'] = layer_f.W
            layer_f.params['b'] = layer_f.b
            layer_f.weight_decay = param_list[2*i+bias+1]['weight_decay']
            layer_f.weight_decay_lambda = param_list[2*i+bias+1]['lambda']
            self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.lambda_list, self.ks_list]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN_v2(Layer):
    # the lyr should be [conv2d, relu, maxpool, conv2d, relu, maxpool, linear, relu, linear, softmax]
    def __init__(self):
        self.layers = []

        # (1, 28, 28) -> (4, 24, 24) -> (4, 12, 12)
        self.layers.append(conv2D(in_channels=1, out_channels=4, kernel_size=5, padding=0, weight_decay=True, weight_decay_lambda=1e-4)) # 0
        self.layers.append(ReLU()) # 1
        self.layers.append(MaxPool2D(kernel_size=2, stride=2)) # 2

        # (4, 12, 12) -> (8, 8, 8) -> (8, 4, 4)

        self.layers.append(conv2D(in_channels=4, out_channels=8, kernel_size=5, padding=0, weight_decay=True, weight_decay_lambda=1e-4)) # 3
        self.layers.append(ReLU()) # 4
        self.layers.append(MaxPool2D(kernel_size=2, stride=2)) # 5

        self.layers.append(Flatten()) # 6

        self.layers.append(Linear(in_dim=8*4*4, out_dim=32, weight_decay=True, weight_decay_lambda=1e-4)) # 7
        self.layers.append(ReLU()) # 8

        self.layers.append(Linear(in_dim=32, out_dim=10, weight_decay=True, weight_decay_lambda=1e-4)) # 9

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def save_model(self, save_path):
        conv1_W = self.layers[0].params['W']
        conv1_b = self.layers[0].params['b']
        conv2_W = self.layers[3].params['W']
        conv2_b = self.layers[3].params['b']
        fc1_W = self.layers[7].params['W']
        fc1_b = self.layers[7].params['b']
        fc2_W = self.layers[9].params['W']
        fc2_b = self.layers[9].params['b']
        with open(save_path, 'wb') as f:
            pickle.dump([conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b], f)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b = param_list
        self.layers[0].params['W'] = conv1_W
        self.layers[0].params['b'] = conv1_b
        self.layers[3].params['W'] = conv2_W
        self.layers[3].params['b'] = conv2_b
        self.layers[7].params['W'] = fc1_W
        self.layers[7].params['b'] = fc1_b
        self.layers[9].params['W'] = fc2_W
        self.layers[9].params['b'] = fc2_b


class Model_CNN_v2_1(Layer):
    # the lyr should be [conv2d, relu, maxpool, conv2d, relu, maxpool, linear, softmax]
    def __init__(self):
        self.layers = []

        # (1, 28, 28) -> (4, 24, 24) -> (4, 12, 12)
        self.layers.append(conv2D(in_channels=1, out_channels=4, kernel_size=5, padding=0, weight_decay=True, weight_decay_lambda=1e-4)) # 0
        self.layers.append(ReLU()) # 1
        self.layers.append(MaxPool2D(kernel_size=2, stride=2)) # 2

        # (4, 12, 12) -> (8, 8, 8) -> (8, 4, 4)

        self.layers.append(conv2D(in_channels=4, out_channels=8, kernel_size=5, padding=0, weight_decay=True, weight_decay_lambda=1e-4)) # 3
        self.layers.append(ReLU()) # 4
        self.layers.append(MaxPool2D(kernel_size=2, stride=2)) # 5

        self.layers.append(Flatten()) # 6

        self.layers.append(Linear(in_dim=8*4*4, out_dim=32, weight_decay=True, weight_decay_lambda=1e-4)) # 7

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def save_model(self, save_path):
        conv1_W = self.layers[0].params['W']
        conv1_b = self.layers[0].params['b']
        conv2_W = self.layers[3].params['W']
        conv2_b = self.layers[3].params['b']
        fc1_W = self.layers[7].params['W']
        fc1_b = self.layers[7].params['b']

        with open(save_path, 'wb') as f:
            pickle.dump([conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b], f)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b = param_list
        self.layers[0].params['W'] = conv1_W
        self.layers[0].params['b'] = conv1_b
        self.layers[3].params['W'] = conv2_W
        self.layers[3].params['b'] = conv2_b
        self.layers[7].params['W'] = fc1_W
        self.layers[7].params['b'] = fc1_b
