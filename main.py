import numpy as np
from PIL import Image
import glob
import os
from random import shuffle

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'


def actv_func_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Hyper Parameters
mean = 0
std = 1
patch_size = 32
mini_batch_size = 5
epochs = 100
mini_batches = epochs // mini_batch_size
num_of_neurons = patch_size * patch_size
num_of_weights_W1 = 10
total_weights = 2
pixel_max_value = 255

# Initialize wights and biases

W1 = np.random.normal(mean, std, (num_of_weights_W1, num_of_neurons))  # (10, 1024)_
W2 = np.random.normal(mean, std, (1, num_of_weights_W1))  # (1,10)
B1 = np.random.normal(mean, std, (num_of_weights_W1, 1))  # (1,10)
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

        accuracy = np.zeros(mini_batch_size)
        loss = np.zeros(mini_batch_size)
        z_L = np.zeros(mini_batch_size)
        z_1 = np.zeros((num_of_weights_W1, mini_batch_size))
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
            z_1[:, i] = z1[:, 0]
            # Output
            W2_multiplied = np.dot(W2, h1)[0]  # W2*h1 # dims are (1,10)*(10,1)
            z2 = (W2_multiplied[0] + B2)  # W2*h1 + B2  # output is a single output
            h2 = max(z2, 0, z2)  # f(W2*h1+b) with RelU
            output = min(1, h2)  # limit output to 1
            z_L[i] = z2

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

        # 4. Compute gradients of the training loss
        # using back propagation equations
        delta_C = 2 * avg_loss
        sigma_tag = actv_func_deriv(z_L).reshape(-1, 1)
        delta_L = delta_C * sigma_tag.T

        delta_l_1 = np.dot(W2.T, delta_L) * actv_func_deriv(z_1)
        gradient_b_L = 1
        gradient_w_L = 1
        gradient_b_1 = 1
        gradient_w_1 = 1
        # BUILT A LOOP BUT ITS A WASTE OF ONLY 2 DELTAS
        # delta_l=[]
        # delta_l.append(delta_L)
        # for i in range(total_weights-1):
        #    cur_delta_l = W*delta_l[i]
        #    np.insert(delta_l,0,cur_delta_l)

        # 5. Update weights and biases using calculated gradients and step size

    # 6. Forward propagate the validation examples,
    # and compute loss and accuracy for them (decide if to stop the training)

# Visualize a learning curve for training set and validation set:
# Plot of loss and accuracy as a function of epochs
