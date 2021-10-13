import numpy as np
from activation_function import *

class DeepNeuralNetwork:
    """
    Constructor:
        learning_rate -- hyperparameter learning rate for each iteration
        layer_dims -- python array (list) containing the number of node in each layer
        num_iterations -- number of iterations which use for training
    """
    def __init__(self, learning_rate, layer_dims, num_iterations ):
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.num_iterations = num_iterations
        self.parameters = self.initialize_parameters()


    def fit(self, X, Y, print_cost):
        self.costs = []    
        for i in range(0, self.num_iterations):
            AL, caches = self.forward_propagation(X, self.parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            self.parameters = self.update_parameters(self.parameters, grads, self.learning_rate)
            
            if print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, cost))
            if i % 100 == 0 or i == self.num_iterations:
                self.costs.append(cost)


    def evaluate(self, X, Y):
        m = X.shape[1]
        p = np.zeros((1,m))
        
        probabilities, _ = self.forward_propagation(X, self.parameters)

        for i in range(0, probabilities.shape[1]):
            if probabilities[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        print("Accuracy: {}%".format(str(np.sum((p == Y)*100/m))))


    def predict(self, x):
        prob, _ = self.forward_propagation(x, self.parameters)

        print("Predict y: {}".format(prob))


    def initialize_parameters(self):
        layer_dims = self.layer_dims
        L = len(layer_dims)
        parameters = {}
        
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        return parameters


    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache


    def linear_activition_forward(self, A_previous, W, b, activation):
        Z, linear_cache = self.linear_forward(A_previous, W, b)
        
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)

        return A, cache


    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_previous = A
            A, cache = self.linear_activition_forward(A_previous, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            caches.append(cache)

        AL, cache = self.linear_activition_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)

        return AL, caches


    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m)*np.sum(Y*np.log(AL) + (1 - Y)*np.log(1-AL))
        cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[n]] into n).

        return cost
    
    def linear_backward(self, dZ, cache):
        A_previous, W, b = cache
        m = A_previous.shape[1]

        dW = (1/m)*np.dot(dZ, A_previous.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_previous = np.dot(W.T, dZ)

        return dA_previous, dW, db

    
    def linear_activation_backward(self, dA, cache, activition):
        linear_cache, activition_cache = cache

        if activition == "sigmoid":
            dZ = sigmoid_backward(dA, activition_cache)
        elif activition == "relu":
            dZ = relu_backward(dA, activition_cache)

        dA_previous, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_previous, dW, db


    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        dA_previous_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L-1)] = dA_previous_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_previous_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_previous_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters