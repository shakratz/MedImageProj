import numpy as np
import cv2 as cv

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'

# Hyper Parameters
patch_size = 32
mean = 0
std = 1
mini_batch_size = 5 #?


num_of_neurons = patch_size*patch_size


# Initialize wights and biases

Weights = np.random.normal(mean, std, num_of_neurons)
Biases = np.random.normal(mean, std, num_of_neurons)

# training_set_arr = []

#for each epoch in epochs:

    # 1. Sample a random mini-batch and prepare input vectors and labels

    # 2. Forward propagation of input vectors through the network
    # and cache forward pass variables
    # sigmoid(pixel*W + B)

    # 3. Compute MSE and accuracy
    # accuracy = []
    # loss = []
    #for each sample in minibatch:
        # loss[i] = ((A-B)**2).mean(axis=None)
        #accuracy:
        #if label == np.round(output):
            #accuracy[sample] = 1
        #else:
            #accuracy[sample] = 0
    # avg_loss = np.average(loss)
    # avg_accuracy = np.average(accuracy)
    # 4. Compute gradients of the training loss
    # using back propagation equations

    # 5. Update weights and biases using calculated gradients and step size

    # 6. Forward propagate the validation examples,
    # and compute loss and accuracy for them (decide if to stop the training)

# Visualize a learning curve for training set and validation set:
# Plot of loss and accuracy as a function of epochs
