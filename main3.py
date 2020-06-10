from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from random import shuffle

TRAINING_PATH = 'training\\'
VALIDATION_PATH = 'validation\\'


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size=1, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        Z1 = X.dot(W1) + b1
        Z1[Z1 <= 0] = 0
        A1 = Z1
        Z2 = A1.dot(W2) + b2
        scores = Z2

        if y is None:
            return scores

        # Compute the loss
        loss = None

        scores -= scores.max()
        correct = y
        correct_exp = np.exp(correct)
        loss = np.sum(-np.log(correct_exp / np.sum(np.exp(scores), axis=1))) / N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass: compute gradients
        grads = {}

        num = np.exp(y)
        denom = np.sum(np.exp(scores), axis=1)
        mask = -(denom - num)/denom
        mask= mask.reshape(-1, 1)
        mask /= N
        grads['W2'] = A1.T.dot(mask) + 2 * reg * W2
        grads['b2'] = np.sum(mask, axis=0)  # mask.dot(np.ones((mask.shape[1],1)))
        PD = mask.dot(W2.T)
        PD[Z1 == 0] = 0
        grads['W1'] = X.T.dot(PD) + 2 * reg * W1
        grads['b1'] = np.sum(PD, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = len(X)
        iterations_per_epoch = num_train // batch_size

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ind = np.random.choice(num_train, batch_size)
            X_batch = X[ind, :]
            y_batch = y[ind]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        y_pred = None

        y_pred = np.argmax(
            np.dot(np.maximum(0, X.dot(self.params['W1']) + self.params['b1']), self.params['W2']) + self.params['b2'],
            axis=1)

        return y_pred


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def init_data():
    pixel_max_value = 255
    # loading the training set
    imList = glob.glob(TRAINING_PATH + '*.png')
    X_train = []
    y_train = []
    for img in imList:
        sample = Image.open(img)
        filepath, filenameExt = os.path.split(img)
        filename, fileExt = os.path.splitext(filenameExt)
        file_label = filename.split("_")[0]
        if file_label == 'pos':
            label = 1
        else:
            label = 0
        sample_array = np.array(sample) / pixel_max_value  # convert to array & normalize
        sample_vector = (sample_array.flatten()).reshape(-1, 1)
        X_train.append(sample_vector)
        y_train.append(label)

    # loading the validation set
    imList = glob.glob(VALIDATION_PATH + '*.png')
    X_val = []
    y_val = []
    for img in imList:
        sample = Image.open(img)
        filepath, filenameExt = os.path.split(img)
        filename, fileExt = os.path.splitext(filenameExt)
        file_label = filename.split("_")[0]
        if file_label == 'pos':
            label = 1
        else:
            label = 0
        sample_array = np.array(sample) / pixel_max_value  # convert to array & normalize
        sample_vector = (sample_array.flatten()).reshape(-1, 1)
        X_val.append(sample_vector)
        y_val.append(label)

    return np.array(X_train)[:, :, 0], np.array(y_train), np.array(X_val)[:, :, 0], np.array(y_val)


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

learning_rate = 1e-4
patch_size = 32
mini_batch_size = 5
epochs = 1000
reg = 0.25
learning_rate_decay = 0.95

input_size = patch_size * patch_size
hidden_size = 10  # neurons

net = TwoLayerNet(input_size, hidden_size)
X_train, y_train, X_val, y_val = init_data()

stats = net.train(X_train, y_train, X_val, y_val,
                  num_iters=epochs, batch_size=mini_batch_size,
                  learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                  reg=reg, verbose=True)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
