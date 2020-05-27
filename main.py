import numpy as np
from PIL import Image
import glob
import os
from random import shuffle

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'

# Hyper Parameters
mean = 0
std = 1
patch_size = 32
mini_batch_size = 5
epochs = 100
mini_batches = epochs // mini_batch_size
num_of_neurons = patch_size * patch_size
num_of_weights = 10
pixel_max_value = 255

# Initialize wights and biases

W1 = np.random.normal(mean, std, (num_of_neurons,num_of_weights))
W2 = np.random.normal(mean, std, (num_of_weights,1))
B1 = np.random.normal(mean, std, (num_of_weights,1))
B2 = np.random.normal(mean, std)


# loading the training set
imList = glob.glob(TRAINING_PATH + '*.png')
training_set_arr = []
for img in imList:
    im = Image.open(img)
    filepath, filenameExt = os.path.split(img)
    filename, fileExt = os.path.splitext(filenameExt)
    file_label = filename.split("_")[0]
    if file_label == 'pos':
        label = 1
    else:
        label = 0
    training_set_arr.append((im, label))

# loading the validation set
imList = glob.glob(VALIDATION_PATH + '*.png')
validation_set_arr = []
for img in imList:
    im = Image.open(img)
    filepath, filenameExt = os.path.split(img)
    filename, fileExt = os.path.splitext(filenameExt)
    file_label = filename.split("_")[0]
    if file_label == 'pos':
        label = 1
    else:
        label = 0
    validation_set_arr.append((im, label))

for epoch in range(epochs):
    # Shuffeling the set so every epoch will use different mini batches
    shuffle(training_set_arr)
    for mini_batch in range(mini_batches):
        # 1. Sample a random mini-batch
        current_batch = training_set_arr[(mini_batch) * mini_batch_size:(mini_batch + 1) * mini_batch_size]

        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample = current_batch[i][0]  # Get image
            sample_array = np.array(sample) / np.sum(sample)  # convert to array & normalize
            sample_vector = (sample_array.flatten()).reshape(-1, 1).T  # reshape to 1*1024
            label = current_batch[i][1]  # get label

            # Forward propagation - hidden layer W and B
            W_multiplied = np.dot(sample_vector, W1)[0]  # X*W1     dims are (1,1024)*(1024,10)
            z1 = (W_multiplied[0] + B1)  # X*W1 + B1  # output dmin is (10,1)
            h1 = np.maximum(z1, 0, z1)  # f(X*W+b) with RelU # output dmin is (10,1)

            # Output
            W_multiplied = np.dot(h1.T, W2)[0]    # X*W2 # dims are (1,10)*(10,1)
            z2 = (W_multiplied[0] + B2)  # X*W2 + B2  # output dmin is a single output
            h2 = max(z2, 0, z2)  # f(X*W+b) with RelU

            # 3. Compute MSE and accuracy
            # accuracy = []
            # loss = []
            # for each sample in minibatch:
            # loss[i] = ((A-B)**2).mean(axis=None)
            # accuracy:
            # if label == np.round(output):
            # accuracy[sample] = 1
            # else:
            # accuracy[sample] = 0
            # avg_loss = np.average(loss)
            # avg_accuracy = np.average(accuracy)
            # 4. Compute gradients of the training loss
            # using back propagation equations

            # 5. Update weights and biases using calculated gradients and step size

            # 6. Forward propagate the validation examples,
            # and compute loss and accuracy for them (decide if to stop the training)

# Visualize a learning curve for training set and validation set:
# Plot of loss and accuracy as a function of epochs
