from numpy import exp, array, random, dot

class NeuronLayer(fwd, bkwd):

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
    
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def getOutput():
        return this.output
    
    def forward(feed):
        this.output =  self.__sigmoid(dot(feed, self.synaptic_weights))
        if (fwd != None):
            fwd.forward(this.output)


#update NN's to have variable number of layers, increase polymorphism of the class or devide the class up into subgroups
class NeuralNetwork():
    def __init__(self, layers):
        self.numLayers = layers.length
        for i in range(self.numLayers - 1):
            layers[i].fwd = layers[i + 1]
            layers[i + 1].bkwd = layers[i]
        self.layerI = layers[0]
        self.layerN = layers[self.numLayers]

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            outputs = self.think(training_set_inputs)
            
            current = layerN
            # Calculate the error for layer n (The difference between the desired output
            # and the predicted output).
            deltas[self.numLayers] = (training_set_outputs - outputs[self.numLayers]) * self.__sigmoid_derivative(outputs[self.numLayers])
            adjustments[self.numLayers] = layerN.bkwd.getOutput().T.dot(deltas[self.numLayers])

            for i in range(self.numLayers - 1):
                current = current.bkwd
                # Calculate the error for layer 1 (By looking at the weights in layer 1,
                # we can determine by how much layer 1 contributed to the error in layer 2).
                current_error = deltas[self.numLayers - i].dot(current.fwd.synaptic_weights.T)
                current_delta = current_error * self.__sigmoid_derivative(current.getOutput())
                deltas[self.numLayers - i - 1] = current_delta
                adjustments[self.numLayers - i - 1] = current.bkwd.getOutput().T.dot(current_delta)

            # Adjust the weights.
            for i in range(self.numLayers - 1):
                current.synaptic_weights += adjustments[i]
                current = current.fwd

    # The neural network thinks.
    def think(self, inputs):
        self.layerI.forward(inputs)
        current = layerI
        for i in range(self.numLayers - 1):
            outputs[i] = current.getOutput()
            current = current.fwd
        return outputs

#increase polymorphism of print weights function
    # The neural network prints its weights
    def print_weights(self):
        current = layerI
        for i in range(self.numLayers - 1):
            print("    Layer  ( neurons, each with  inputs): ")
            print(self.current.synaptic_weights)
            current = current.fwd
