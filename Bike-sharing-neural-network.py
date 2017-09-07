# Sigmoid function is applied in the hidden layer in order to introduce
# non-linearity in the model to make more accurate predictions.

# Sigmoid functions are not applied in the final output node since it's
# a regression problem. Activation function f(x) = x is used though.

# During back-prop, the derivative of the activation function is used
# in the final output layer in order to get the error term (its value is 1).

# Hidden grad is the derivative of the activation function applied in the
# hidden layer, so derivative of the sigmoid functions is used.

# '*' is element-wise multiplication while np.dot is the matrix one.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open the dataset
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)

# Create dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

# Scaling target variables
# Store scalings in a dictionary so we can convert back later
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# Splitting the data into training, testing and validation sets
# Save the last 21 days 
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

# Build the network -------------------------------------------------------------------------
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

         # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        # Inputs is a vector of shape (56, 1)
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        ### Forward pass ###
        # Hidden layer - OK
        # weights_input_to_hidde.shape = (2, 56) [2 hidden nodes, 56 input nodes]
        # inputs.shape = (56, 1)
        # After dot product, hidden_inputs.shape = (2, 1) [2 hidden output nodes]
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        # The hidden_output needs an activation function
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        # weights_hidden_to_output.shape = (1, 2) [1 output node, 2 hidden nodes]
        # hidden_outputs.shape = (2, 1) [2 hidden output nodes]
        # After dot product, final_inputs.shape = (1, 1) [1 output node]
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # We don't use the sigmoid function here since it's a regression problem
        final_outputs = final_inputs
        
        ### Backward pass ###        
        # Output error
        # Output layer error is the difference between desired target and actual output
        # (Activation function derivatied is missing)
        output_errors = targets - final_outputs      
        # Backpropagated error
        # errors propagated to the hidden layer
        # Gradient is a vector (N, 1) of all the partial derivaties of the function itself
        # weights_hidden_to_output.T.shape = (2, 1) [2 hidden nodes, 1 output node]
        # output_errors.shape = (1, 1)
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        hidden_grad = hidden_outputs * (1 - hidden_outputs)

        # Update the weights
        # Update hidden-to-output weights with gradient descent step
        # output_errors.shape = (1, 1) [1 error from the output node]
        # hidden_outputs.T.shape = (1, 2) [2 hidden output nodes]
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        # update input-to-hidden weights with gradient descent step
        # hidden_errors.shape = (1, 2)
        # inputs.T.shape = (1, 56)
        self.weights_input_to_hidden += self.lr * np.dot((hidden_grad * hidden_errors), inputs.T)
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs

        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

# Training the network -------------------------------------------------------------
import sys

# Hyper-parameters
epochs = 2500
learning_rate = 0.15
hidden_nodes = 15
output_nodes = 1

# N_i = 56 (last 6 days of features)
N_i = train_features.shape[1]
# Network is the instance of the class NeuralNetwork
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # Go through a random batch of n = 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

# Plot the graphs
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)

# Checking out the predictions
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()
