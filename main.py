import numpy as np
from PIL import Image
import glob
import os
from random import shuffle
import matplotlib.pyplot as plt

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'


def actv_func_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Hyper Parameters
learning_rate = 0.001
mean = 0
std = 0.1
patch_size = 32
mini_batch_size = 5
epochs = 100
num_of_pixels = patch_size * patch_size
num_of_neurons = 10
total_weights = 2
pixel_max_value = 255

# Initialize wights and biases

W1 = np.random.normal(mean, std, (num_of_neurons, num_of_pixels))  # (10, 1024)
W2 = np.random.normal(mean, std, (1, num_of_neurons))  # (1,10)
B1 = np.random.normal(mean, std, (num_of_neurons, 1))  # (10,1)
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

mini_batches_training = len(training_set_arr) // mini_batch_size
mini_batches_validation = len(validation_set_arr) // mini_batch_size

mini_batch_results_training = np.zeros((mini_batches_training, 2))
mini_batch_results_validation = np.zeros((mini_batches_validation, 2))
total_results_training = np.zeros((epochs, 2))
total_results_validation = np.zeros((epochs, 2))
for epoch in range(epochs):
    # Shuffeling the set so every epoch will use different mini batches
    shuffle(training_set_arr)
    for mini_batch in range(mini_batches_training):
        # 1. Sample a random mini-batch
        current_batch = training_set_arr[(mini_batch) * mini_batch_size:(mini_batch + 1) * mini_batch_size]

        accuracy = np.zeros(mini_batch_size)
        loss = np.zeros(mini_batch_size)
        pixels = np.zeros((num_of_pixels, mini_batch_size))
        z_L = np.zeros(mini_batch_size)
        z_1 = np.zeros((num_of_neurons, mini_batch_size))
        h_1 = np.zeros((num_of_neurons, mini_batch_size))

        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample = current_batch[i][0]  # Get image
            sample_array = np.array(sample) / pixel_max_value  # convert to array & normalize
            sample_vector = (sample_array.flatten()).reshape(-1, 1)  # reshape to 1024*1
            label = current_batch[i][1]  # get label
            pixels[:, i] = sample_vector[:, 0]  # store data for back propogation

            # Forward propagation - hidden layer W and B
            W1_multiplied = np.dot(W1, sample_vector)  # W1*X     dims are (10,1024)*(1024,1)
            z1 = (W1_multiplied[0] + B1)  # W1*X + B1  # output dmin is (10,1)
            h1 = np.maximum(z1, 0)  # f(W1*X+b) with RelU # output dmin is (10,1)

            z_1[:, i] = z1[:, 0]  # store data for back propogation
            h_1[:, i] = h1[:, 0]  # store data for back propogation

            W2_multiplied = np.dot(W2, h1)[0]  # W2*h1 # dims are (1,10)*(10,1)
            z2 = (W2_multiplied[0] + B2)  # W2*h1 + B2  # output is a single output
            output = max(z2, 0)  # f(W2*h1+b) with RelU

            # output = min(1, h2)  # limit output to 1
            z_L[i] = z2  # store data for back propogation

            # 3. Compute MSE and accuracy
            loss[i] = (output - label) ** 2
            #loss[i] = np.power((output - label),2)
            #loss[i] += learning_rate * np.sum(W1*W1) + np.sum(W2*W2)

            # accuracy:
            if label == np.round(output):
                accuracy[i] = 1
            else:
                accuracy[i] = 0

        # Calculate average loss and accuracy
        avg_loss = np.average(loss)
        avg_accuracy = np.average(accuracy)

        mini_batch_results_training[mini_batch, 0] = avg_loss
        mini_batch_results_training[mini_batch, 1] = avg_accuracy

        # 4. Compute gradients of the training loss
        # using back propagation equations
        delta_C = 2 * avg_loss  # size: (1,1)
        sigma_tag = np.mean(actv_func_deriv(z_L))  # size: (1,1)
        delta_L = delta_C * sigma_tag.T  # size: (1,1)

        gradient_b_2 = delta_L  # size:(1,1)

        a_k_1 = (h_1.mean(axis=1)).reshape(1, -1)
        gradient_w_2 = np.dot(a_k_1, delta_L)  # size:(1,10) as W2 size

        delta_l_1 = np.dot(W2.T, delta_L) * actv_func_deriv(z_1.mean(axis=1)).reshape(-1, 1)  # size: (10,1)

        gradient_b_1 = delta_l_1  # size:(10,1)

        a_k_1 = (pixels.mean(axis=1)).reshape(-1, 1)
        gradient_w_1 = np.dot(a_k_1, delta_l_1.T).T  # size :(10, 1024)

        gradient_w_1 += learning_rate * 2 * W1
        gradient_w_2 += learning_rate * 2 * W2

        # 5. Update weights and biases using calculated gradients and step size
        W1 = W1 - learning_rate * gradient_w_1  # (10, 1024)
        W2 = W2 - learning_rate * gradient_w_2  # (1,10)
        B1 = B1 - learning_rate * gradient_b_1  # (10, 1)
        B2 = B2 - learning_rate * gradient_b_2

    total_results_training[epoch, 0] = np.mean(mini_batch_results_training[:, 0])
    total_results_training[epoch, 1] = np.mean(mini_batch_results_training[:, 1])

    ############################# VALIDATION ################################

    # 6. Forward propagate the validation examples,
    # and compute loss and accuracy for them (decide if to stop the training)
    shuffle(validation_set_arr)
    for mini_batch in range(mini_batches_validation):
        # 1. Sample a random mini-batch
        current_batch = validation_set_arr[mini_batch * mini_batch_size:(mini_batch + 1) * mini_batch_size]

        accuracy = np.zeros(mini_batch_size)
        loss = np.zeros(mini_batch_size)
        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample = current_batch[i][0]  # Get image
            sample_array = np.array(sample) / np.sum(sample)  # convert to array & normalize
            sample_vector = (sample_array.flatten()).reshape(-1, 1)  # reshape to 1*1024
            label = current_batch[i][1]  # get label

            # Forward propagation - hidden layer W and B
            W1_multiplied = np.dot(W1, sample_vector)  # W1*X     dims are (1,1024)*(1024,10)
            z1 = (W1_multiplied[0] + B1)  # W1*X + B1  # output dmin is (10,1)
            h1 = np.maximum(z1, 0, z1)  # f(W1*X+b) with RelU # output dmin is (10,1)

            # Output
            W2_multiplied = np.dot(W2, h1)[0]  # W2*h1 # dims are (1,10)*(10,1)
            z2 = (W2_multiplied[0] + B2)  # W2*h1 + B2  # output is a single output
            h2 = max(z2, 0, z2)  # f(W2*h1+b) with RelU
            output = min(1, h2)  # limit output to 1

            # 3. Compute MSE and accuracy
            loss[i] = (output - label) ** 2
            # accuracy:
            if label == np.round(output):
                accuracy[i] = 1
            else:
                accuracy[i] = 0

        # Calculate average loss and accuracy
        avg_loss = np.average(loss)
        avg_accuracy = np.average(accuracy)

        mini_batch_results_validation[mini_batch, 0] = avg_loss
        mini_batch_results_validation[mini_batch, 1] = avg_accuracy

    total_results_validation[epoch, 0] = np.mean(mini_batch_results_training[:, 0])
    total_results_validation[epoch, 1] = np.mean(mini_batch_results_training[:, 1])

# Visualize a learning curve for training set and validation set:
# Plot of loss and accuracy as a function of epochs
######### PLOTTING TRAINING RESULTS #########
plt.figure()
plt.subplot(211)
plt.plot(range(epochs), total_results_training[:, 0], linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.title('Training loss')
axes = plt.gca()
axes.set_ylim([0, 1])

plt.subplot(212)
plt.plot(range(epochs), total_results_training[:, 1], linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average accuracy')
plt.title('Training accuracy')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.show()

"""
######### PLOTTING VALIDATION RESULTS #########
plt.figure()
plt.subplot(211)
plt.plot(range(epochs), total_results_validation[:, 0], linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.title('Validation loss')
axes = plt.gca()
axes.set_ylim([0, 1])

plt.subplot(212)
plt.plot(range(epochs), total_results_validation[:, 1], linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average accuracy')
plt.title('Validation accuracy')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.show()
"""