import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    # because softmax(x) = softmax(x-c) for any constant c,
    # we can subtract the maximum value of x from each element of x.
    # this does not change the result of softmax, but it makes the numbers in x much smaller,
    # which is more numerically stable.

    # We will use the identity: softmax(x) = softmax(x+c).
    # We will use c = the minus of the largest logit.
    new_x = x - np.max(x)
    np_exp = np.exp(new_x)
    return np_exp / np.sum(np_exp)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def classifier_output(x, params):
    W1, b1, W2, b2 = params
    return softmax(np.dot(W2.T, np.tanh(np.dot(W1.T, x) + b1)) + b2)

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # TODO: YOU CODE HERE
    return ...

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    return params

