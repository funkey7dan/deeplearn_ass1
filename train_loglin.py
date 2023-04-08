import loglinear as ll
import random
from utils import *
import numpy as np


STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # TODO: YOU CODE HERE
    # Should return a numpy vector of features.
    vec = np.zeros(len(vocab))
    for f in features:
        if f in F2I:
            vec[F2I[f]] = 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # TODO: YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        prediction = np.argmax(feats_to_vec(features).dot(params[0]) + params[1])
        if prediction == L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # TODO: YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        #print(I, train_loss, train_accuracy, dev_accuracy)
        print(f'iteration {round(I,2)}: train_loss={round(train_loss,2)}, train_accuracy={round(train_accuracy,2)}, dev_accuracy={round(dev_accuracy,2)}')
    return params

if __name__ == '__main__':
    # TODO: YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # Initialize the training arguments
    train_data = TRAIN
    dev_data = DEV
    num_iterations = 10
    learning_rate = 0.001
    in_dim = len(F2I.keys()) # the number of features we input
    out_dim = len(L2I.keys()) # the number of labels, in our case the different languages we have.
    
    # Create a log linear classifier with the specified input and output dimensions, repressented as the parameters W and b.
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

