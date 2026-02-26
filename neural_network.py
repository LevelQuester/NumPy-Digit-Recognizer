import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_layer_sizes, inputLayerSize, OutputLayerSize, seed=0):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.numberOfHiddenLayers = len(self.hidden_layer_sizes)
        self.inputLayerSize = inputLayerSize
        self.OutputLayerSize = OutputLayerSize

        self.layer_dimensions = [self.inputLayerSize] + self.hidden_layer_sizes + [self.OutputLayerSize]
        self.all_weights = [None] * (len(self.layer_dimensions) - 1)
        self.all_biases = [None] * (len(self.layer_dimensions) - 1)

        rng = np.random.default_rng(seed)
        for idx in range(len(self.all_weights)):
            out_dim = self.layer_dimensions[idx + 1]
            in_dim = self.layer_dimensions[idx]
            weight_scale = np.sqrt(2.0 / in_dim)
            self.all_weights[idx] = rng.normal(0.0, weight_scale, size=(out_dim, in_dim))
            self.all_biases[idx] = np.zeros((out_dim, 1))

    @staticmethod
    def ReLU(preactivation):
        activation = preactivation.clip(0.0)
        return activation

    @staticmethod
    def softmax(preactivation):
        shifted = preactivation - np.max(preactivation, axis=0, keepdims=True)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return probabilities
        
    def forwardPassForOneInput(self, net_input):

        # Retrieve number of layers
        K = self.numberOfHiddenLayers

        # We'll store the pre-activations at each layer in a list "all_f"
        # and the activations in a second list "all_h".
        all_f = [None] * (K+1)
        all_h = [None] * (K+2)

        #For convenience, we'll set
        # all_h[0] to be the input, and all_f[K] will be the output
        all_h[0] = net_input

        # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
        for layer in range(K):
            all_f[layer] = self.all_biases[layer] + np.matmul(self.all_weights[layer], all_h[layer])
            all_h[layer+1] = self.ReLU(all_f[layer])

        # Compute the output from the last hidden layer
        all_f[K] = self.all_biases[K] + np.matmul(self.all_weights[K], all_h[K])
        all_h[K+1] = self.softmax(all_f[K])

        # Retrieve the output
        net_output = all_h[K+1]

        return net_output, all_f, all_h
    
    @staticmethod
    def cross_entropy_loss(probabilities, y):
        eps = 1e-12
        return -np.sum(y * np.log(probabilities + eps))

    @staticmethod
    def d_loss_d_output(probabilities, y):
        return probabilities - y
    

    @staticmethod
    def indicator_function(x):
        x_in = np.array(x)
        x_in[x_in>0] = 1
        x_in[x_in<=0] = 0
        return x_in
    
        # Main backward pass routine
    def backward_pass(self, all_f, all_h, y):
        K = self.numberOfHiddenLayers
        # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
        all_dl_dweights = [None] * (K+1)
        all_dl_dbiases = [None] * (K+1)
        # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
        all_dl_df = [None] * (K+1)
        all_dl_dh = [None] * (K+1)
        # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

        # Compute derivatives of the loss with respect to the network output
        probabilities = all_h[K+1]
        all_dl_df[K] = np.array(self.d_loss_d_output(probabilities, y))

        # Now work backwards through the network
        for layer in range(K,-1,-1):
            all_dl_dbiases[layer] = np.array(all_dl_df[layer])
            all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T)

            all_dl_dh[layer] = np.matmul(self.all_weights[layer].T, all_dl_df[layer])

            if layer > 0:
                all_dl_df[layer-1] = all_dl_dh[layer] * self.indicator_function(all_f[layer-1])

        return all_dl_dweights, all_dl_dbiases
        
