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
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W, b = params
    return softmax(np.dot(W.T, x) + b)

def predict(x, params):
    """
    Returns the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """

    # Forward
    W, b = params
    z = np.dot(W.T, x) + b
    y_hat = softmax(z)

    # one hot vector since we use hard CE
    y_vec = np.zeros(len(y_hat))
    y_vec[y] = 1

    # HARD cross entropy loss
    loss = -np.log(y_hat[y])

    # Backward
    dLoss_dA = -y_vec / y_hat  # ∂Loss\∂a -> vector

    # calculate the Jacobian matrix of the softmax function at z
    S = np.reshape(y_hat, (-1, 1))
    J = np.diagflat(S) - np.dot(S, S.T)

    # calculate the gradient of the loss with respect to z.
    # the g of loss w.r.t y_hat. this gives in dl_dz_i the sum of the chain rule mult.
    dLoss_dz = np.dot(J, dLoss_dA)

    # calculate the gradient of z with respect to W
    dLoss_dW = np.outer(x, dLoss_dz)

    # calculate the gradient of z with respect to b. b is simply added making dz_db = 1
    dLoss_dB = dLoss_dz

    return loss, [dLoss_dW, dLoss_dB]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.random.randn(in_dim, out_dim) * 0.01  # W matrix
    b = np.zeros(out_dim)
    return [W, b]


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print(f'test 1: {test1}')
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6
    print(f'test 1: Passed')
    test2 = softmax(np.array([1001,1002]))
    print(f'test 2: {test1}')
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6
    print(f'test 2: Passed')

    test3 = softmax(np.array([-1001,-1002]))
    print(f'test 3: {test3}')
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6
    print(f'test 3: Passed')

    test4 = softmax(np.array([1]))
    print(f'test 4: {test4}')
    assert np.amax(np.fabs(test4 - np.array([1.0]))) <= 1e-6
    print(f'test 4: Passed')

    test5 = softmax(np.array([0, 0, 0]))
    print(f'test 5: {test5}')
    assert np.amax(np.fabs(test5 - np.array([0.33333333, 0.33333333, 0.33333333]))) <= 1e-6
    print(f'test 5: Passed')


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b = create_classifier(3, 4)

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
