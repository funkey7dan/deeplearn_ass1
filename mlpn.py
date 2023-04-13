import numpy as np


STUDENT1={'name': 'Coral Kuta',
         'ID': 'CORAL_ID'}
STUDENT2={'name': 'Daniel Bronfman ',
         'ID': 'DANIEL_ID '}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
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


def tanh_grad(x):
    return 1. -np.tanh(x) ** 2  # sech^2{x}


def classifier_output(x, params):
    result = x
    for i in range(0, len(params) - 3, 2):
        result = np.dot(params[i].T, result) + params[i + 1]
        result = np.tanh(result)
    result = np.dot(params[len(params) - 2].T, result) + params[len(params) - 1]
    return softmax(result)


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """

    # Forward
    forward = [x, np.dot(params[0].T, x) + params[1]]

    for i in range(2, len(params) - 1, 2):
        forward.append(np.tanh(forward[len(forward) - 1]))
        forward.append(np.dot(params[i].T, forward[len(forward) - 1]) + params[i + 1])
    forward.append(softmax(forward[len(forward) - 1]))
    y_hat = forward[len(forward) - 1]

    # one hot vector since we use hard CE
    y_vec = np.zeros(len(y_hat))
    y_vec[y] = 1

    # HARD cross entropy loss
    loss = -np.log(y_hat[y])

    grads = []

    # Backward
    dLoss_dAn = -y_vec / y_hat  # ∂Loss\∂an -> vector

    # Now - calculate the derivative of the loss w.r.t h2:
    # ∂Loss\∂hn = ∂Loss\∂an * ∂an\∂hn  (* is np.dot)

    # calculate the Jacobian matrix - the derivative of an (=softmax) w.r.t hn
    # meaning - ∂an\∂hn
    temp = np.reshape(y_hat, (-1, 1))  # helper vector for turning the prediction to a vector
    dAn_dHn = np.diagflat(temp) - np.dot(temp, temp.T)  # Jacobian -> y (1 - y) derivative of output w.r.t hn ∂y_hat \∂hn

    # now ∂Loss\∂hn (∂Loss\∂hn = ∂Loss\∂an DOT ∂an\∂hn)
    dLoss_dHn = np.dot(dAn_dHn, dLoss_dAn)  # -> vector

    # Now - calculate the derivative of the loss w.r.t Wn
    # NOTE - this is NOT Jacobian matrix again -> this time only one neuron is affected by the multiplication

    # ∂Loss\∂Wn = ∂Loss\∂hn DOT ∂hn\∂wn
    grads.insert(0, np.outer(forward[len(forward) - 3], dLoss_dHn))
    grads.insert(1, dLoss_dHn)

    for i in reversed(range(2, len(params) - 1, 2)):
        # dH2_dA1 is Jacobian matrix = W2.T
        dLoss_dAn = np.dot(params[i], dLoss_dHn)

        # ∂Loss\∂an
        dAn_dHn = tanh_grad(forward[i - 1])  # derivative of tanh is 1-tanh^2

        # every element of Hn affect one element of An - tanh is element-wise
        dLoss_dHn = dLoss_dAn * dAn_dHn

        # ∂Loss\∂Wn = ∂Loss\∂hn DOT ∂hn\∂wn
        grads.insert(0, np.outer(forward[i - 2], dLoss_dHn))
        grads.insert(1, dLoss_dHn)

    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(1, len(dims)):
        params.append(np.random.randn(dims[i - 1], dims[i]) * 0.01)
        #params.append(np.zeros(dims[i]))
        params.append(np.random.randn(dims[i]) * 0.01)
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    list_params = [create_classifier([3, 4]),
                   create_classifier([3, 4, 5]),
                   create_classifier([3, 4, 5, 6])]
    parameters = None
    t = 0

    def _loss_and_grad(x):
        global parameters
        global t
        parameters[t] = x
        loss, grads = loss_and_gradients([1, 2, 3], 0, parameters)
        return loss, grads[t]

    for k in range(2, 5):
        print("number of parameters: " + str(k))
        parameters = list_params[k - 2]
        for _ in range(10):
            # W
            for j in range(0, len(parameters), 2):
                parameters[j] = np.random.randn(parameters[j].shape[0], parameters[j].shape[1])

            # b
            for j in range(1, len(parameters), 2):
                parameters[j] = np.random.randn(parameters[j].shape[0])

            for j in range(len(parameters)):
                t = j
                gradient_check(_loss_and_grad, parameters[j])
        print(" -------------------------- ")
