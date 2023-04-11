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
    return 1. - np.tanh(x) ** 2


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
    dLoss_dH2 = np.dot(dA2_dH2, dLoss_dA2)   # -> vector

    # Now - calculate the derivative of the loss w.r.t W2
    # NOTE - this is NOT Jacobian matrix again -> this time only one neuron is affected by the multiplication

    # ∂Loss\∂W2 = ∂Loss\∂h2 DOT ∂h2\∂w2
    gW2 = np.outer(a1, dLoss_dH2)
    gb2 = dLoss_dH2

    # dH2_dA1 is Jacobian matrix = W2.T
    dLoss_dA1 = np.dot(W2, dLoss_dH2)

    # ∂Loss\∂a1
    dA1_dH1 = tanh_grad(h1)  # derivative of tanh is 1-tanh^2

    # every element of H1 affect one element of A1 - tanh is element-wise
    dLoss_dH1 = dLoss_dA1 * dA1_dH1

    # ∂Loss\∂W1 = ∂Loss\∂h1 DOT ∂h1\∂w1
    gW1 = np.outer(x, dLoss_dH1)
    gb1 = dLoss_dH1

    return loss, [gW1, gb1, gW2, gb2]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W1 = np.random.randn(in_dim, hid_dim) # W1 matrix
    b1 = np.zeros(hid_dim)  # b
    W2 = np.random.randn(hid_dim, out_dim)  # W2 matrix
    b2 = np.zeros(out_dim)  # b_tag
    return [W1, b1, W2, b2]


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    parameters = create_classifier(3, 4, 5)
    t = 0

    def _loss_and_grad(x):
        global parameters
        global t
        parameters[t] = x
        loss, grads = loss_and_gradients([1, 2, 3], 0, parameters)
        return loss, grads[t]

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


