import math
import os
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Input:
    def __init__(self, vec, label):
        self.vec = vec
        self.label = label


class DataSet:
    def __init__(self, path=os.getcwd()):
        self.training = []
        self.validation = []
        self.test = []
        for i in listdir(path + '/training'):
            if i[0:3] == 'pos':
                l = 1
            else:
                l = 0
            temp = plt.imread(path + '/training/' + i)
            vec = np.reshape(temp, (1, temp.size))
            self.training.append(Input(vec, l))
        for i in listdir(path + '/validation'):
            if i[0:3] == 'pos':
                l = 1
            else:
                l = 0
            temp = plt.imread(path + '/validation/' + i)
            vec = np.reshape(temp, (1, temp.size))
            self.validation.append(Input(vec, l))
        """
        for i in listdir(path + '/test'):
            if i[0:3] == 'pos':
                l = 1
            else:
                l = 0
            temp = plt.imread(path + '/test/' + i)
            vec = np.reshape(temp, (1, temp.size))
            self.test.append(Input(vec, l))
        """

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            m = np.dot(w, a)
            z = m + b
            a = sigmoid(z)
        return a

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def SGD(self, data, epochs, mini_batch_size, eta):
        training_data = []
        train_len = len(data.training)
        for k in range(train_len):
            train = data.training[k].vec.reshape((-1, 1))
            label = data.training[k].label
            training_data.append((train, label))

        validation_rate_list = []
        training_rate_list = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batces = [training_data[k:k + mini_batch_size] for k in
                           range(0, train_len, mini_batch_size)]
            sum_loss_val = 0
            sum_acc_val = 0
            sum_loss_tr = 0
            sum_acc_tr = 0
            count = 0
            for mini_batce in mini_batces:
                self.update_mini_batch(mini_batce, eta)
                val_acc, val_loss = CalcLossAccuracy(data.validation, self)
                sum_loss_val = sum_loss_val + val_loss
                sum_acc_val = sum_acc_val + val_acc
                tr_acc, tr_loss = CalcLossAccuracy(data.training, self)
                sum_loss_tr = sum_loss_tr + tr_loss
                sum_acc_tr = sum_acc_tr + tr_acc
                count = count + 1

            sum_loss_val = sum_loss_val / count
            sum_acc_val = sum_acc_val / count
            sum_loss_tr = sum_loss_tr / count
            sum_acc_tr = sum_acc_tr / count

            validation_rate_list.append((sum_acc_val, sum_loss_val))
            training_rate_list.append((sum_acc_tr, sum_loss_tr))
            print('epoch:' + str(j + 1))

        return validation_rate_list, training_rate_list

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


def CalcLossAccuracy(Adata, net):
    data1 = [Adata[k].vec for k in range(len(Adata))]
    labels = [Adata[k].label for k in range(len(Adata))]
    acc = 0.0
    loss = 0.0
    for i in range(len(data1)):
        temp = net.feed_forward((data1[i]).reshape((-1, 1)))
        loss = loss + (labels[i] - temp) ** 2
        if temp < 0.5:
            res = 0
        else:
            res = 1
        acc = acc + float(res == labels[i])

    acc = acc / len(data1)
    loss = np.sqrt(loss)
    loss = loss / len(data1)
    return acc, loss


def main():
    topology = []
    topology.append(1024)
    topology.append(30)
    topology.append(1)
    net = Network(topology)
    eta = 0.005
    data = DataSet(os.getcwd())
    mini_batch_size = 32
    epochs = 500
    validation_rate_list, training_rate_list = net.SGD(data, epochs,
                                                       mini_batch_size, eta)
    loss = [validation_rate_list[k][1][0][0] for k in
            range(len(validation_rate_list))]
    acc = [validation_rate_list[k][0] for k in
           range(len(validation_rate_list))]
    plt.figure(1)
    plt.plot(acc)
    plt.title('validation - accuracy as a function of epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()
    plt.figure(2)
    plt.plot(loss)
    plt.title('validation - loss as a function of epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
    loss = [training_rate_list[k][1][0][0] for k in
            range(len(training_rate_list))]
    acc = [training_rate_list[k][0] for k in range(len(training_rate_list))]
    plt.figure(3)
    plt.plot(acc)
    plt.title('training - accuracy as a function of epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()
    plt.figure(4)
    plt.plot(loss)
    plt.title('training - loss as a function of epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()


if __name__ == '__main__':
    main()
