import numpy as np

class Network:
    def __init__(self, widths):
        self.widths = widths
        self.layers = []
        for i in range(len(self.widths)-1):
            self.layers.append(Layer(self.widths[i], self.widths[i+1]))

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights.tolist())
        return weights

    @weights.setter
    def weights(self, weights):
        for layer_weights, layer in zip(weights, self.layers):
            layer.weights = np.asarray(layer_weights)

    @property
    def biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.bias.tolist())
        return biases

    @biases.setter
    def biases(self, biases):
        for bias, layer in zip(biases, self.layers):
            layer.bias = np.asarray(bias)

    def forward_pass(self, activation):
        for layer in self.layers:
            activation = layer.update_activation(activation)
        return activation

    def backward_pass(self, y):
        layer = self.layers[-1]
        dCda = 2*(layer.activation_out - y)
        for layer in reversed(self.layers):
            dCda = layer.update_partial_derivatives(dCda)

    def update_weights(self, lrate, batch_size):
        for layer in self.layers:
            layer.update_weights(lrate, batch_size)

    def update_biases(self, lrate, batch_size):
        for layer in self.layers:
            layer.update_bias(lrate, batch_size)

    def cost(self, activation, y):
        activation = self.forward_pass(activation)
        return (sum(np.power(activation - y, 2))/len(y))[0]

class Layer:
    def __init__(self, width_in, width_out):
        self._width_in = width_in
        self._width_out = width_out
        self._weights = (np.random.random((self.width_out, self.width_in)))*0.001
        self._bias = np.random.random((self.width_out, 1))
        self._activation_in = np.random.random((self.width_in, 1))
        self._activation_out = None
        self.update_activation(self._activation_in)
        self._dCdw = None
        self._dCdb = None
        self.clear_partial_derivatives()

    def update_partial_derivatives(self, dCda):
        #dCdw = dzdw*dadz*dCda
        #dCdb = dzdb*dadz*dCda
        #dCda(L-1) = SUM(dzda(L-1)*da(L)dz*dCda(L))

        dadz = self.activation_out*(1-self.activation_out)
        dCdz = dadz*dCda

        self._dCdw += np.matmul(dCdz, self._activation_in.T)
        self._dCdb += dCdz     #dCdz*dzdb = dCdz*1

        return np.matmul(self.weights.T, dCdz)  #return dCda

    def clear_dCdw(self):
        self._dCdw = np.zeros((self.width_out, self.width_in))

    def clear_dCdb(self):
        self._dCdb = np.zeros((self.width_out, 1))

    def clear_partial_derivatives(self):
        self.clear_dCdw()
        self.clear_dCdb()

    def update_weights(self, lrate, batch_size):
        self.weights -= lrate*self._dCdw/batch_size
        self.clear_dCdw()

    def update_bias(self, lrate, batch_size):
        self.bias -= lrate*self._dCdb/batch_size
        self.clear_dCdb()

    def update_activation(self, activation):
        self._activation_out = self.sigmoid(np.matmul(self._weights, activation) + self._bias)
        return self._activation_out

    @property
    def activation_out(self):
        return self._activation_out

    def sigmoid(self, v):
        return 1/(1 + np.exp(-v))

    @property
    def width_in(self):
        return self._width_in

    @property
    def width_out(self):
        return self._width_out

    @property
    def weights(self):
         return self._weights

    @weights.setter
    def weights(self, weights):
         self._weights = weights

    @property
    def bias(self):
         return self._bias

    @bias.setter
    def bias(self, bias):
         self._bias = bias
