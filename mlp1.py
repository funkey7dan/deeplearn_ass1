import numpy as np

STUDENT = {'name': 'YOUR NAME',
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
    gW: matrix, gradients of W -> W1
    gb: vector, gradients of b -> b1
    gU: matrix, gradients of U -> W2
    gb_tag: vector, gradients of b_tag -> b2
    """
    W1, b1, W2, b2 = params

    # Forward
    h1 = np.dot(W1.T, x) + b1
    a1 = np.tanh(h1)
    h2 = np.dot(W2.T, a1) + b2
    a2 = softmax(h2)
    y_hat = a2

    # one hot vector since we use hard CE
    y_vec = np.zeros(len(y_hat))
    y_vec[y] = 1

    # HARD cross entropy loss
    loss = -np.log(y_hat[y])

    # Backward
    dLoss_dA2 = -y_vec / a2  # ∂Loss\∂a2 -> vector

    # Now - calculate the derivative of the loss w.r.t h2:
    # ∂Loss\∂h2 = ∂Loss\∂a2 * ∂a2\∂h2  (* is np.dot)

    # calculate the Jacobian matrix - the derivative of a2 (=softmax) w.r.t h2
    # meaning - ∂a2\∂h2
    temp = np.reshape(y_hat, (-1, 1))  # helper vector for turning the prediction to a vector
    dA2_dH2 = np.diagflat(temp) - np.dot(temp, temp.T)  # Jacobian -> y (1 - y) derivative of output w.r.t h2 ∂y_hat \∂h2

    # now ∂Loss\∂h2 (∂Loss\∂h2 = ∂Loss\∂a2 DOT ∂a2\∂h2)
    dLoss_dH2 = np.dot(dA2_dH2, dLoss_dA2)  #

    # Now - calculate the derivative of h2 w.r.t W2
    # NOTE - this is NOT Jacobian matrix again -> this time only one neuron is affected by the multiplication
    dH2_dW2 = np.outer(a1, dLoss_dH2)
    dH2_dB2 = dH2_dW2  # calculate the derivative of h2 w.r.t b2

    dloss_W2 = 0  # use the above to calculate the derivative of the loss w.r.t W2

    gb = out.copy()
    # gradient of b = p(y|x) - 1
    gb[y] -= 1
    # gradient of W = x * (p(y|x) - 1)
    gW = np.outer(x, out.copy())
    gW[:, y] -= x
    return loss, [gW, gb]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W1 = np.zeros((in_dim, hid_dim))  # W matrix
    b1 = np.zeros(hid_dim)  # b
    W2 = np.zeros((hid_dim, out_dim))  # U matrix
    b2 = np.zeros(out_dim)  # b_tag
    return [W1, b1, W2, b2]
