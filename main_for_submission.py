import numpy as np
import glob
import os
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import json
import os

start_time = time.time()
TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'


def CrossEntropy(y, ytag):  # y=output , ytag = label
    epsilon = 1e-5
    return -(ytag * np.log(y + epsilon) + (1 - ytag) * np.log(1 - y + epsilon))


def MSE(y, ytag):
    return 0.5 * (y - ytag) ** 2


def ReLU_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Parameters
patch_size = 32
num_of_pixels = patch_size * patch_size
mean = 0
std = 0.01

# Hyper Parameters
learning_rate = 5e-3
mini_batch_size = 1
epochs = 300
num_of_neurons = 15

# Initialize wights and biases
W1 = np.random.normal(mean, std, (num_of_neurons, num_of_pixels))  # (10, 1024)
W2 = np.random.normal(mean, std, (1, num_of_neurons))  # (1,10)
B1 = np.random.normal(mean, std, (num_of_neurons, 1))  # (10,1)
B2 = np.random.normal(mean, std)

# loading the training set
imList = glob.glob(TRAINING_PATH + '*.png')
training_set_arr = []
for img in imList:
    im = mpimg.imread(img)
    im_vector = (im.flatten()).reshape(-1, 1)  # reshape to 1024*1
    filepath, filenameExt = os.path.split(img)
    filename, fileExt = os.path.splitext(filenameExt)
    file_label = filename.split("_")[0]
    if file_label == 'pos':
        label = 1
    else:
        label = 0
    training_set_arr.append((im_vector, label))

# loading the validation set
imList = glob.glob(VALIDATION_PATH + '*.png')
validation_set_arr = []
for img in imList:
    im = mpimg.imread(img)
    im_vector = (im.flatten()).reshape(-1, 1)  # reshape to 1024*1
    filepath, filenameExt = os.path.split(img)
    filename, fileExt = os.path.splitext(filenameExt)
    file_label = filename.split("_")[0]
    if file_label == 'pos':
        label = 1
    else:
        label = 0
    validation_set_arr.append((im_vector, label))

# Calculating amount of mini batches
mini_batches_training = len(training_set_arr) // mini_batch_size
mini_batches_validation = len(validation_set_arr) // mini_batch_size

# Setting up arrays
mini_batch_results_training = np.zeros((mini_batches_training, 2))
mini_batch_results_validation = np.zeros((mini_batches_validation, 2))
total_results_training = np.zeros((epochs, 2))
total_results_validation = np.zeros((epochs, 2))

for epoch in range(epochs):
    # Shuffling the set so every epoch will use different mini batches
    shuffle(training_set_arr)
    for mini_batch in range(mini_batches_training):
        # 1. Sample a random mini-batch
        current_batch = training_set_arr[mini_batch * mini_batch_size:(mini_batch + 1) * mini_batch_size]

        # Setting up arrays
        accuracy = np.zeros(mini_batch_size)
        loss = np.zeros(mini_batch_size)
        pixels = np.zeros((num_of_pixels, mini_batch_size))
        z_L = np.zeros(mini_batch_size)
        z_1 = np.zeros((num_of_neurons, mini_batch_size))
        h_1 = np.zeros((num_of_neurons, mini_batch_size))
        y_minus_ytag = np.zeros(mini_batch_size)
        #############################################
        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample_vector = current_batch[i][0]  # Get image
            label = current_batch[i][1]  # get label
            pixels[:, i] = sample_vector[:, 0]  # store data for back propagation

            # Forward propagation - hidden layer W and B
            # Layer 1
            W1_multiplied = np.dot(W1, sample_vector)  # W1*X     dims are (10,1024)*(1024,1)
            z1 = (W1_multiplied + B1)  # W1*X + B1  # output dmin is (10,1)
            h1 = np.maximum(z1, np.zeros((len(z1), 1)))  # f(W1*X+b) with ReLU  # output dmin is (10,1)

            z_1[:, i] = z1[:, 0]  # store data for back propagation
            h_1[:, i] = h1[:, 0]  # store data for back propagation

            # Layer 2
            W2_multiplied = np.dot(W2, h1)  # W2*h1 # dims are (1,10)*(10,1)
            z2 = W2_multiplied + B2  # W2*h1 + B2  # output is a single output
            h2 = sigmoid(z2)
            output = float(h2)  # f(W2*h1+b) with sigmoid

            z_L[i] = z2  # store data for back propagation
            y_minus_ytag[i] = output - label  # store data for back propagation
            #############################################
            # 3. Compute MSE and accuracy
            # loss[i] = CrossEntropy(output, label)
            loss[i] = MSE(output, label)

            # accuracy:
            if label == np.round(output):
                accuracy[i] = 1
            else:
                accuracy[i] = 0

        # Calculate average loss and accuracy of mini batch
        avg_loss = np.average(loss)
        avg_accuracy = np.average(accuracy)

        # Store average loss and accuracy for plotting
        mini_batch_results_training[mini_batch, 0] = avg_loss
        mini_batch_results_training[mini_batch, 1] = avg_accuracy

        #############################################
        # 4. Compute gradients of the training loss
        # using back propagation equations

        # Deltas
        delta_C = np.mean(y_minus_ytag)  # size: (1,1)
        sigma_tag = np.mean(sigmoid_prime(z_L))  # size: (1,1)
        delta_L = delta_C * sigma_tag.T  # size: (1,1)

        delta_l_1 = np.dot(W2.T, delta_L) * ReLU_prime(z_1.mean(axis=1)).reshape(-1, 1)  # size: (10,1)

        # Gradients
        gradient_b_1 = delta_l_1  # size:(10,1)
        gradient_b_2 = delta_L  # size:(1,1)

        a_k_0 = (pixels.mean(axis=1)).reshape(-1, 1)
        gradient_w_1 = np.dot(a_k_0, delta_l_1.T).T  # size :(10, 1024) as W1 size

        a_k_1 = (h_1.mean(axis=1)).reshape(1, -1)
        gradient_w_2 = np.dot(a_k_1, delta_L)  # size:(1,10) as W2 size

        # 5. Update weights and biases using calculated gradients and step size
        W1 -= learning_rate * gradient_w_1  # (10, 1024)
        W2 -= learning_rate * gradient_w_2  # (1,10)
        B1 -= learning_rate * gradient_b_1  # (10, 1)
        B2 -= learning_rate * gradient_b_2

    if epoch % 100 == 0:
        print('average training accuracy = {0}'.format(np.mean(mini_batch_results_training[:, 1])))
    # Store average loss and accuracy for plotting
    total_results_training[epoch, 0] = np.mean(mini_batch_results_training[:, 0])
    total_results_training[epoch, 1] = np.mean(mini_batch_results_training[:, 1])

    # ############################ VALIDATION ############################### #

    # 6. Forward propagate the validation examples,
    # and compute loss and accuracy for them (decide if to stop the training)
    shuffle(validation_set_arr)
    for mini_batch in range(mini_batches_validation):
        # 1. Sample a random mini-batch
        current_batch = validation_set_arr[mini_batch * mini_batch_size:(mini_batch + 1) * mini_batch_size]

        # Setting up arrays
        accuracy = np.zeros(mini_batch_size)
        loss = np.zeros(mini_batch_size)
        pixels = np.zeros((num_of_pixels, mini_batch_size))
        z_L = np.zeros(mini_batch_size)
        z_1 = np.zeros((num_of_neurons, mini_batch_size))
        h_1 = np.zeros((num_of_neurons, mini_batch_size))
        y_minus_ytag = np.zeros(mini_batch_size)
        #############################################
        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample_vector = current_batch[i][0]  # Get image
            label = current_batch[i][1]  # get label
            pixels[:, i] = sample_vector[:, 0]  # store data for back propagation

            # Forward propagation - hidden layer W and B
            # Layer 1
            W1_multiplied = np.dot(W1, sample_vector)  # W1*X     dims are (10,1024)*(1024,1)
            z1 = (W1_multiplied + B1)  # W1*X + B1  # output dmin is (10,1)
            h1 = np.maximum(z1, np.zeros((len(z1), 1)))  # f(W1*X+b) with ReLU  # output dmin is (10,1)

            z_1[:, i] = z1[:, 0]  # store data for back propagation
            h_1[:, i] = h1[:, 0]  # store data for back propagation

            # Layer 2
            W2_multiplied = np.dot(W2, h1)  # W2*h1 # dims are (1,10)*(10,1)
            z2 = W2_multiplied + B2  # W2*h1 + B2  # output is a single output
            h2 = sigmoid(z2)
            output = float(h2)  # f(W2*h1+b) with sigmoid

            z_L[i] = z2  # store data for back propagation
            y_minus_ytag[i] = output - label  # store data for back propagation
            #############################################
            # 3. Compute MSE and accuracy
            # loss[i] = CrossEntropy(output, label)
            loss[i] = MSE(output, label)

            # accuracy:
            if label == np.round(output):
                accuracy[i] = 1
            else:
                accuracy[i] = 0

        # Calculate average loss and accuracy of mini batch
        avg_loss = np.average(loss)
        avg_accuracy = np.average(accuracy)

        # Store average loss and accuracy for plotting
        mini_batch_results_validation[mini_batch, 0] = avg_loss
        mini_batch_results_validation[mini_batch, 1] = avg_accuracy

    if epoch % 100 == 0:
        print('average validation accuracy = {0}'.format(np.mean(mini_batch_results_validation[:, 1])))
    # Store average loss and accuracy for plotting
    total_results_validation[epoch, 0] = np.mean(mini_batch_results_validation[:, 0])
    total_results_validation[epoch, 1] = np.mean(mini_batch_results_validation[:, 1])

# Visualize a learning curve for training set and validation set:
# Plot of loss and accuracy as a function of epochs
# ######## PLOTTING TRAINING RESULTS ######### #

print('### Total time: {0}'.format(time.time() - start_time))
"""
plt.figure()
plt.subplot(211)
loss = total_results_training[:, 0]
plt.plot(range(epochs), loss, linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.title('Training loss')
axes = plt.gca()
# axes.set_ylim([0, 1])

plt.subplot(212)
acc = total_results_training[:, 1]
plt.plot(range(epochs), acc, linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average accuracy')
plt.title('Training accuracy')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.show()

######### PLOTTING VALIDATION RESULTS #########
plt.figure()
plt.subplot(211)
loss = total_results_validation[:, 0]
plt.plot(range(epochs), total_results_validation[:, 0], linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.title('Validation loss')
axes = plt.gca()
# axes.set_ylim([0, 1])

plt.subplot(212)
acc = total_results_validation[:, 1]
plt.plot(range(epochs), acc, linewidth=2.0)
plt.xlabel('Epochs')
plt.ylabel('Average accuracy')
plt.title('Validation accuracy')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.show()
"""


def make_json(trained_dict, path_to_save):
    """
    make json file with trained parameters.
    W1: numpy arrays of shape (1024, nn_h_dim)
    W2: numpy arrays of shape (nn_h_dim, 1)
    b1: numpy arrays of shape (1, nn_h_dim)
    b2: numpy arrays of shape (1, 1)
    id1: id1 - int
    id2: id2 - int
    activation1: one of only: 'sigmoid', 'tanh', 'ReLU', 'final_act' - str
    activation2: 'sigmoid' - str
     number of neirons in hidden layer - int
    :param nn_h_dim: trained_dict = {'weights': (W1, W2),
                                    'biases': (b1, b2),
                                    'nn_hdim': nn_h_dim,
                                    'activation_1': activation1,
                                    'activation_2': activation2,
                                    'IDs': (id1, id2)}
    """
    file_path = os.path.join(path_to_save, 'trained_dict_{}_{}.json'.format(
        trained_dict.get('IDs')[0], trained_dict.get('IDs')[1])
                             )
    with open(file_path, 'w') as f:
        json.dump(trained_dict, f, indent=4)


# Preparing dims
W1_sub = (W1.T).tolist()
W2_sub = (W2.T).tolist()
B1_sub = (B1.T).tolist()
B2_sub = (np.array(B2)).tolist()
trained_dict = {
    'weights': (W1_sub, W2_sub),
    'biases': (B1_sub, B2),
    'nn_hdim': num_of_neurons,
    'activation_1': 'ReLU',
    'activation_2': 'sigmoid',
    'IDs': (307973693, 200940500)}

path = "C:\\Python\\Medical Images\\Project - medical"
make_json(trained_dict, path)
