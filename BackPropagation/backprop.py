# coding=utf-8
"""
@author: beyourself
@time: 2017/10/5 18:53
"""

import numpy as np
from BackPropagation.data_prep import features, targets, features_test, targets_test

np.random.seed(21)


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    v = sigmoid(x)
    return v * (1 - v)


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

import time

t1 = time.time()

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        # The activation function of output layer is also sigmoid
        o_input = np.dot(hidden_output, weights_hidden_output)
        o_output = sigmoid(o_input)

        ## Backward pass ##
        # Calculate the network's prediction error
        error = y - o_output

        # Calculate error term for the output unit
        # output_error_term = error * o_output * (1 - o_output)  # This save time
        output_error_term = error * sigmoid_prime(o_input)  # This is a formula

        ## propagate errors to hidden layer

        # Calculate the hidden layer's contribution to the error
        # This is a special case where n_output is 1.
        # Normally, hidden_error = np.dot(output_error_term, weights_hidden_output.T)
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # Calculate the error term for the hidden layer
        # * --- (m, ) * (m, ) is ok
        # * --- (m, ) * (n, 1) is ok = (n, m)
        # hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)  # This save time
        hidden_error_term = hidden_error * sigmoid_prime(hidden_input)  # This is a formula

        # Update the change in weights
        # special case
        # Normally, del_w_hidden_output = error_term * vin[:, None]
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # Update weights
    # weight matrix += learnrate * del_w_input_hidden
    # Sometimes need divide n
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

t2 = time.time()
print('training time : %s' % int(t2 - t1))

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
