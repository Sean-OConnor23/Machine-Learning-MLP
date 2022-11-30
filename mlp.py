import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

def get_genre(genres):
    answers = []
    for genre in genres:
        if genre == "blues":
            answers.append(0)
        elif genre == "classical":
            answers.append(1)
        elif genre == "country":
            answers.append(2)
        elif genre == "disco":
            answers.append(3)
        elif genre == "hiphop":
            answers.append(4)
        elif genre == "jazz":
            answers.append(5)
        elif genre == "metal":
            answers.append(6)
        elif genre == "pop":
            answers.append(7)
        elif genre == "reggae":
            answers.append(8)
        elif genre == "rock":
            answers.append(9)
    return answers


class Layer:
    # a layer is a building block of the neural network
    # each layer is capable of performing a forward pass and a backward pass

    def __init__(self):
        pass

    def forward(self, input):
        # input: data of shape [batch, input_units]
        # output: data of shape [batch, output_units]
        return input

    def backward(self, input, grad_output):
        # backpropagation wrt given input (chain rule)

        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


# Nonlinearity ReLU layer
class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        # compute gradient loss wrt ReLU input
        relu_grad = input > 0
        return grad_output*relu_grad


# Dense layer
#   f(X) = WX+b
#      - X is an object-feature matrix of shape [batch_size, num_features]
#      - W is a weight matrix pnum_features, num_outputs]
#      - b is a vector of num_outputs biases
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units+output_units)), size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        # input shape: [batch, input_units]
        # output shape: [batch, output_units]
        # f(x) = WX+b

        return np.dot(input, self.weights) + self.biases

    def backward (self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # stochastic gradient descent
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

# Loss function
def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    return xentropy

def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]
    
# MLP
# read data
df = pd.read_csv("data/features_30_sec.csv")
# shuffle data
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,1:-1]
y = df['label']
y = y.replace(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# split train and test data 80/20
X_train = X.iloc[:800].to_numpy()
y_train = y.iloc[:800].to_numpy()

X_test = X.iloc[800:].to_numpy()
y_test = y.iloc[800:].to_numpy()

network = []
network.append(Dense(X_train.shape[1],100))
network.append(ReLU())
network.append(Dense(100,200))
network.append(ReLU())
network.append(Dense(200,10))

def forward(network, X):
    activations = []
    input = X

    for l in network:
        activations.append(l.forward(input))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

def predict(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)

def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad) # grad wrt input

    return np.mean(loss)

from tqdm import trange
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

from IPython.display import clear_output
train_log = []
test_log = []

for epoch in range(25):
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
        train(network, x_batch, y_batch)
    
    train_log.append(np.mean(predict(network, X_train)==y_train))
    test_log.append(np.mean(predict(network,X_test)==y_test))

    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Test accuracy:", test_log[-1])
    plt.plot(train_log, label='train accuracy')
    plt.plot(test_log, label='test accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
