import nengo
import nengo_dl

def build_model(self):
    # input_shape: Shape of the input data
    # hidden_layers: List of integers defining the number of neurons in each hidden layer
    # output_dim: Dimension of the output layer
    # activation: Activation function for neurons, defaulting to SpikingRectifiedLinear
    # seed: Random seed for reproducibility

    model = nengo.Network(seed=self.seed)
    with model:
        # Define the input layer
        input_layer = nengo.Node(nengo.processes.PresentInput(self.input_shape, presentation_time=0.1))

        # Define hidden layers with specified activation functions
        prev_layer = input_layer
        for layer_size in self.hidden_layers:
            layer = nengo.Ensemble(n_neurons=layer_size, dimensions=1, neuron_type=nengo.SpikingRectifiedLinear())
            nengo.Connection(prev_layer, layer, synapse=None)
            prev_layer = layer

        # Define the output layer
        output_layer = nengo.Ensemble(n_neurons=self.output_dim, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(prev_layer, output_layer, synapse=None)

    return model
