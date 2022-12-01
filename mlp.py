import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from wav_feat_extract import wav_extract
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# HELPER FUNCTIONS
def accuracy_score(y_true, y_pred):
    # compute accuracy
    accuracy = np.sum(np.equal(y_true, y_pred)) / len(y_true)
    return accuracy

def normalize(X):
    # normalize the dataset
    X = (X - X.min()) / (X.max() - X.min())
    return X

def to_categorical(x, n_col=None):
    # because the data is in categories, use one-hot encoding
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_gradient(x):
    return softmax(x) * (1 - softmax(x))

def loss(y, p):
    # Avoid division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - y * np.log(p) - (1 - y) * np.log(1 - p)

def loss_acc(y, p):
    return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

def loss_gradient(y, p):
    # Avoid division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)

def get_genre(songs):
    genres = []
    for song in songs:
        if song == 0:
            genres.append("blues")
        elif song == 1:
            genres.append("classical")
        elif song == 2:
            genres.append("country")
        elif song == 3:
            genres.append("disco")
        elif song == 4:
            genres.append("hiphop")
        elif song == 5:
            genres.append("jazz")
        elif song == 6:
            genres.append("metal")
        elif song == 7:
            genres.append("pop")
        elif song == 8:
            genres.append("reggae")
        elif song == 9:
            genres.append("rock")
    return genres

# MULTILAYER PERCEPTRON
# DATA
# read data
df = pd.read_csv("data/features_30_sec.csv")
# preprocess data
df = df.sample(frac=1).reset_index(drop=True) # shuffle data
X = normalize(df.iloc[:,1:-1].to_numpy()) # normalize
y = df['label'].replace(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = to_categorical(y.to_numpy()) # one-hot

# split train and test data 80/20
X_train = X[:800]
y_train = y[:800]

X_test = X[800:]
y_test = y[800:]

train_log = []
test_log = []

# HYPER-PARAMETERS
n_hidden = 15 # number of neurons in hidden layer
n_epochs = 9400
learning_rate = 0.01

# initial weights - Xavier Weight Initialization
n_samples, n_features = X.shape
_, n_outputs = y.shape
weights = []
# hidden layer
limit = 1 / math.sqrt(n_features)
weights.append(np.random.uniform(-limit, limit, (n_features, n_hidden)))
weights.append(np.zeros((1, n_hidden)))
# output layer
limit = 1 / math.sqrt(n_hidden)
weights.append(np.random.uniform(-limit, limit, (n_hidden, n_outputs)))
weights.append(np.zeros((1, n_outputs)))

# CLASSIFIER
def fit(X, y, weights):
    W = weights[0]
    w0 = weights[1]
    V = weights[2]
    v0 = weights[3]
    for _ in range(n_epochs):
        # make prediction (forward pass)
        # hidden layer
        hidden_input = X.dot(W) + w0
        hidden_output = sigmoid(hidden_input)
        # output layer
        output_layer_input = hidden_output.dot(V) + v0
        y_pred = softmax(output_layer_input)

        # backward
        # output layer
        # gradient descent wrt input of output layer
        grad_wrt_out_l_input = loss_gradient(y, y_pred) * softmax_gradient(output_layer_input)
        grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
        grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
        # hidden layer
        # gradient descent wrt input of hidden layer
        grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(V.T) * sigmoid_gradient(hidden_input)
        grad_w = X.T.dot(grad_wrt_hidden_l_input)
        grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

        # Update weights (by gradient descent)
        # Move against the gradient to minimize loss
        V  -= learning_rate * grad_v
        v0 -= learning_rate * grad_v0
        W  -= learning_rate * grad_w
        w0 -= learning_rate * grad_w0
    weights = [W, w0, V, v0]

# Use the trained model to predict labels of X
def predict(X, weights):
    W = weights[0]
    w0 = weights[1]
    V = weights[2]
    v0 = weights[3]
    # forward pass
    hidden_input = X.dot(W) + w0
    hidden_output = sigmoid(hidden_input)
    output_layer_input = hidden_output.dot(V) + v0
    y_pred = softmax(output_layer_input)
    return y_pred

# use classifier on data
fit(X_train, y_train, weights)
    
train_log = []
test_log = []

y_pred = np.argmax(predict(X_train, weights), axis=1)
y_train = np.argmax(y_train, axis=1)
train_log.append(accuracy_score(y_train, y_pred))

y_pred = np.argmax(predict(X_test, weights), axis=1)
y_test = np.argmax(y_test, axis=1)
test_log.append(accuracy_score(y_test, y_pred))

print('Train accuracy:', train_log[-1])
print('Test accuracy:', test_log[-1])

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print('##############################################')
print('#               WELCOME TO THE               #')
print('#            SONG GENRE CLASSIFIER           #')
print('##############################################')

another = 'y'
while(another != 'n'):
    file = input('\nEnter song file: ')
    print('\nProcessing your song.....')
    data = wav_extract(file)
    data = normalize(data)

    from tqdm import tqdm
    for i in tqdm(range(int(9e6))):
        pass

    print('\nProcessing complete.....')
    y_pred = np.argmax(predict(data, weights), axis=1)
    print('\n\nThis song is: ')
    print(get_genre(y_pred))
    
    another = input('\nWould you like to classify another song? (y/n): ')

print('Goodbye.')
