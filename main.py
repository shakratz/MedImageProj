import numpy as np
from PIL import Image
import glob
import os
from random import shuffle
import matplotlib.pyplot as plt

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'


def init_params(layer_dims, mean = 0 ,std = 1): #layer_dims = [1024,10,1]
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.random.normal(mean, std, (layer_dims[l], layer_dims[l-1]))
        params['b' + str(l)] = np.random.normal(mean, std, (layer_dims[l], 1))
    return params


def sigmoid(Z):
    A = 1 / (1 + np.exp(np.dot(-1, Z)))
    cache = (Z)
    return A, cache

def reLU(Z):
    A = max(Z.any(),0)
    cache = (Z)
    return A, cache

def BackPropagation(avg_loss, caches):
    grads = {}
    delta_C = 2 * avg_loss  # size: (1,1)
    sigma_tag = np.mean(actv_func_deriv(z_L))  # size: (1,1)
    delta_L = delta_C * sigma_tag.T  # size: (1,1)

    grads['b' + str(2)] = delta_L  # size:(1,1)

    a_k_1 = (h_1.mean(axis=1)).reshape(1, -1)
    grads['W' + str(2)] = np.dot(a_k_1, delta_L)  # size:(1,10) as W2 size

    delta_l_1 = np.dot(W2.T, delta_L) * actv_func_deriv(z_1.mean(axis=1)).reshape(-1, 1)  # size: (10,1)

    grads['b' + str(1)] = delta_l_1  # size:(10,1)

    a_k_1 = (pixels.mean(axis=1)).reshape(-1, 1)
    grads['W' + str(1)] = np.dot(a_k_1, delta_l_1.T).T  # size :(10, 1024)

    return grads

def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L - 1)], grads['db' + str(L - 1)] = one_layer_backward(dAL,
                                                                                                      current_cache)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def ForwardPropagation(X, params):
    A = X  # input to first layer i.e. training data
    caches = []
    L = len(params) // 2
    for l in range(1, L + 1):
        A_prev = A

        # Z = W*x + b
        Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]

        # Storing the linear cache
        linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])

        # H = sigmoid(Z)
        A, activation_cache = sigmoid(Z)

        # storing both the linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    return A, caches


def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))  # Derivative of the sigmoid function

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def LoadSet(path):
    imList = glob.glob(path + '*.png')
    set_arr = []
    for img in imList:
        im = Image.open(img)
        filepath, filenameExt = os.path.split(img)
        filename, fileExt = os.path.splitext(filenameExt)
        file_label = filename.split("_")[0]
        if file_label == 'pos':
            label = 1
        else:
            label = 0
        set_arr.append((im, label))
    return set_arr


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['W' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['b' + str(l + 1)]

    return parameters

def actv_func_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def PlotResult(epochs, total_results, method):
    plt.figure()
    plt.subplot(211)
    plt.plot(range(epochs), total_results[:, 0], linewidth=2.0)
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')
    plt.title(method + ' loss')
    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.subplot(212)
    plt.plot(range(epochs), total_results[:, 1], linewidth=2.0)
    plt.xlabel('Epochs')
    plt.ylabel('Average accuracy')
    plt.title('Validation accuracy')
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.show()


# Hyper Parameters
learning_rate = 2
mean = 0
std = 1
patch_size = 32
mini_batch_size = 5
epochs = 100
num_of_pixels = patch_size * patch_size
num_of_weights_W1 = 10
total_weights = 2
pixel_max_value = 255

layer_dims = [num_of_pixels, num_of_weights_W1, 1]
# Initialize wights and biases

W1 = np.random.normal(mean, std, (num_of_weights_W1, num_of_pixels))  # (10, 1024)
W2 = np.random.normal(mean, std, (1, num_of_weights_W1))  # (1,10)
B1 = np.random.normal(mean, std, (num_of_weights_W1, 1))  # (10,1)
B2 = np.random.normal(mean, std)

# loading the training set
training_set_arr = LoadSet(TRAINING_PATH)

# loading the validation set
validation_set_arr = LoadSet(VALIDATION_PATH)


mini_batches_training = len(training_set_arr) // mini_batch_size
mini_batches_validation = len(validation_set_arr) // mini_batch_size

mini_batch_results_training = np.zeros((mini_batches_training, 2))
mini_batch_results_validation = np.zeros((mini_batches_validation, 2))
total_results_training = np.zeros((epochs, 2))
total_results_validation = np.zeros((epochs, 2))
grads = {}
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
        z_1 = np.zeros((num_of_weights_W1, mini_batch_size))
        h_1 = np.zeros((num_of_weights_W1, mini_batch_size))

        # 2. Forward propagation of input vectors through the network
        for i in range(len(current_batch)):
            # prepare input vector and label
            sample = current_batch[i][0]  # Get image
            sample_array = np.array(sample, dtype = 'f')/255  # convert to array & normalize
            sample_vector = (sample_array.flatten('F')).reshape(-1,1)  # reshape to 1*1024
            label = current_batch[i][1]  # get label
            pixels[:, i] = sample_vector[:, 0] # save pixels for backward propogation


            params = init_params(layer_dims)
            output, caches = ForwardPropagation(sample_vector, params)

            # 3. Compute MSE and accuracy
            loss[i] = (output - label) ** 2
            # accuracy:
            if label == np.round(output):
                accuracy[i] = 1
            else:
                accuracy[i] = 0

        # 3. Calculate average loss and accuracy
        avg_loss = np.average(loss)
        avg_accuracy = np.average(accuracy)

        mini_batch_results_training[mini_batch, 0] = avg_loss
        mini_batch_results_training[mini_batch, 1] = avg_accuracy

        # 4. Compute gradients of the training loss
        # using back propagation equations

        #Y = []
        #for i in range(len(current_batch)):
        #    Y.append(current_batch[i][1])

        grads = BackPropagation(output, caches)

        # delta_l_0 = np.dot(W1.T, delta_l_1) * actv_func_deriv(pixels)  # size: (1024,5) **pixels are z0?**

        # 5. Update weights and biases using calculated gradients and step size
        update_parameters(params, grads, learning_rate)

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
            sample_array = np.array(sample) / 255  # convert to array & normalize
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
PlotResult(epochs, total_results_training, 'Training')

######### PLOTTING VALIDATION RESULTS #########
PlotResult(epochs, total_results_validation, 'Validation')