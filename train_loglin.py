import loglinear as ll
import utils
from utils import *
import numpy as np


STUDENT1={'name': 'Coral Kuta',
         'ID': 'CORAL_ID'}
STUDENT2={'name': 'Daniel Bronfman ',
         'ID': 'DANIEL_ID '}

# NOTE - in order to run Unigram, turn to False
isBigram = True

if isBigram:
    vocab = utils.vocab_bi
    L2I = utils.L2I
    F2I = utils.F2I_BI
    TRAIN = utils.TRAIN_BI
    DEV = utils.DEV_BI
else:
    vocab = utils.vocab_uni
    L2I = utils.L2I
    F2I = utils.F2I_UNI
    TRAIN = utils.TRAIN_UNI
    DEV = utils.DEV_UNI


def feats_to_vec(features):
    # Should return a numpy vector of features.
    vec = np.zeros(len(F2I))
    for f in features:
        if f in F2I:
            vec[F2I[f]] = 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = utils.L2I[label]
        prediction = ll.predict(x, params)

        if prediction == y:
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
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # print(I, train_loss, train_accuracy, dev_accuracy)
        print(f'iteration {round(I,2)}: train_loss={round(train_loss,2)},'
              f' train_accuracy={round(train_accuracy,2)}, dev_accuracy={round(dev_accuracy,2)}')
    return params


if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # Initialize the training arguments
    train_data = TRAIN
    dev_data = DEV
    num_iterations = 20
    learning_rate = 0.001

    # the number of features we input
    in_dim = len(F2I.keys())
    # the number of labels, in our case the different languages we have.
    out_dim = len(utils.L2I.keys())
    
    # Create a log linear classifier with the specified input and output dimensions,
    # represented as the parameters W and b.
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]
    I2L = {v: k for k, v in L2I.items()}
    predictions = [I2L[ll.predict(feats_to_vec(feature), trained_params)] for label, feature in TEST]
    with open("test.pred", "w") as f:
        f.writelines(line + '\n' for line in predictions)
        
    

