import numpy as np
import matrix_operations as mxops

class NeuralNetwork():
    
    def __init__(self, nodes, layers, output):
        self.input_nodes = nodes
        self.hidden_nodes = layers
        self.output_nodes = output
        
        self.bias_input_to_hidden = np.random.random((1, layers))
        self.bias_hidden_to_output = np.random.random((output, 1))
        
        self.weight_input_to_hidden = np.random.random((self.input_nodes, self.hidden_nodes))
        self.weight_hidden_to_output = np.random.random((self.output_nodes, self.hidden_nodes))
        
        self.learning_rate = 0.1
    
    
    def fit(self, x_train, y_train):
        x_train = np.asmatrix(x_train)
        hidden = mxops.matrix_mult(x_train, self.weight_input_to_hidden)
        hidden = mxops.matrix_sum(hidden, self.bias_input_to_hidden)
        hidden = mxops.sigmoid(hidden)
        
        output = mxops.matrix_mult(self.weight_hidden_to_output, hidden.T)
        output = mxops.matrix_sum(output, self.bias_hidden_to_output)
        output = mxops.sigmoid(output)
        
        expected = np.asmatrix(y_train)
        output_error = mxops.matrix_sub(expected, output.T)
        derivative_output = mxops.derivative_sigmoid(output)
        
        gradient = mxops.hadamard(output_error, derivative_output.T)
        gradient = mxops.scalar_multiply(gradient, self.learning_rate)
        
        self.bias_hidden_to_output = mxops.matrix_sum(self.bias_hidden_to_output, gradient.T)
        
        weight_hidden_to_output_deltas = mxops.matrix_mult(gradient.T, hidden)
        self.weight_hidden_to_output = mxops.matrix_sum(self.weight_hidden_to_output, weight_hidden_to_output_deltas)
        
        hidden_error = mxops.matrix_mult(output_error, self.weight_hidden_to_output)
        derivative_hidden = mxops.derivative_sigmoid(hidden)
        
        gradient_hidden = mxops.hadamard(hidden_error, derivative_hidden)
        gradient_hidden = mxops.scalar_multiply(gradient_hidden, self.learning_rate)
        
        self.bias_input_to_hidden = mxops.matrix_sum(self.bias_input_to_hidden, gradient_hidden)
        
        weight_input_to_hidden_deltas = mxops.matrix_mult(gradient_hidden.T, x_train)
        self.weight_input_to_hidden = mxops.matrix_sum(self.weight_input_to_hidden, weight_input_to_hidden_deltas.T)
        

    def predict(self, x_test):
        x_test = np.asmatrix(x_test)
        hidden = mxops.matrix_mult(x_test, self.weight_input_to_hidden)
        hidden = mxops.matrix_sum(hidden, self.bias_input_to_hidden)
        hidden = mxops.sigmoid(hidden)
        
        output = mxops.matrix_mult(self.weight_hidden_to_output, hidden.T)
        output = mxops.matrix_sum(output, self.bias_hidden_to_output)
        output = mxops.sigmoid(output)
        output = output.tolist()
        return output[0][0]
