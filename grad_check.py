import numpy as np


STUDENT1={'name': 'Coral Kuta',
         'ID': 'CORAL_ID'}
STUDENT2={'name': 'Daniel Bronfman ',
         'ID': 'DANIEL_ID '}

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        ### YOUR CODE HERE:
        x1 = x.copy()
        x1[ix] += h

        x2 = x.copy()
        x2[ix] -= h

        fx1, _ = f(x1)
        fx2, _ = f(x2)

        numeric_gradient = np.divide(np.subtract(fx1, fx2), (2.0 * h))
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))

        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return
    
        it.iternext() # Step to next index

    print("Gradient check passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)
    tahn = lambda x: (np.sum(np.tanh(x)), 1 - np.tanh(x) ** 2)
    leaky_relu = lambda x: (np.sum(np.maximum(0.01 * x, x)), np.maximum(0.01, np.sign(x)))
    cubic = lambda x: (np.sum(x ** 3), 3 * (x ** 2))
    exp = lambda x: (np.sum(np.exp(x)), np.exp(x))


    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))        # scalar test
    gradient_check(quad, np.random.randn(3,))      # 1-D test
    gradient_check(quad, np.random.randn(4, 5))    # 2-D test

    # additional sanity checks
    gradient_check(quad, np.random.randn(3,4,5))   # 3-D test
    gradient_check(quad, np.random.randn(3,4,5,6)) # 4-D test
    gradient_check(tahn, np.random.randn(4,5))
    gradient_check(leaky_relu, np.random.randn(4,5))
    gradient_check(cubic, np.random.randn(4,))      # 1-D test
    gradient_check(exp, np.random.randn(3,))        # 1-D test
    gradient_check(cubic, np.random.randn(4,5))     # 2-D test
    gradient_check(exp, np.random.randn(3,4))       # 2-D test
    print()

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
