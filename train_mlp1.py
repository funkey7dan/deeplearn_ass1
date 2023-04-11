import mlp1 as mlp
import utils
from utils import *
import numpy as np


STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # Should return a numpy vector of features.
    vec = np.zeros(len(vocab))
    for f in features:
        if f in F2I:
            vec[F2I[f]] = 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = utils.L2I[label]
        prediction = mlp.predict(x, params)

        if prediction == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def accuracy_on_dataset_xor(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        vec = np.zeros(2)
        for index,f in enumerate(features):
            vec[index] = f
        x = vec
        y = label               
        prediction = mlp.predict(x, params)

        if prediction == label:
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
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.L2I[label]        # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]
            params[2] -= learning_rate * grads[2]
            params[3] -= learning_rate * grads[3]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # print(I, train_loss, train_accuracy, dev_accuracy)
        print(f'iteration {round(I,2)}: train_loss={round(train_loss,2)},'
              f' train_accuracy={round(train_accuracy,2)}, dev_accuracy={round(dev_accuracy,2)}')
    return params



def train_xor(train_data, dev_data, num_iterations, learning_rate, params):
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
            x = np.array(features,dtype='float')
            y = label               # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]
            params[2] -= learning_rate * grads[2]
            params[3] -= learning_rate * grads[3]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset_xor(train_data, params)
        dev_accuracy = accuracy_on_dataset_xor(dev_data, params)
        # print(I, train_loss, train_accuracy, dev_accuracy)
        print(f'iteration {round(I,2)}: train_loss={round(train_loss,2)},'
              f' train_accuracy={round(train_accuracy,2)}, dev_accuracy={round(dev_accuracy,2)}')
    return params

def learn_xor():
    from xor_data import data
    
    for num_iterations in range(1,1000,10):
        learning_rate = 0.9
        params = mlp.create_classifier(2,2,2)
        trained_params = train_xor(data, data, num_iterations, learning_rate, params)
        t1 = (mlp.predict(np.array([0,0]), trained_params)==1)
        t2 = (mlp.predict(np.array([1,0]), trained_params)==0)
        t3 = (mlp.predict(np.array([0,1]), trained_params)==0)
        t4 = (mlp.predict(np.array([1,1]), trained_params)==1)
        if t1 and t2 and t3 and t4:
            print(f'num_iterations={num_iterations}, learning_rate={learning_rate}')
            break
        
    
if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    learn_xor()
    exit()
    # Initialize the training arguments
    train_data = TRAIN
    dev_data = DEV
    num_iterations = 20
    learning_rate = 0.01

    # the number of features we input
    in_dim = len(utils.F2I.keys())
    # the number of labels, in our case the different languages we have.
    out_dim = len(utils.L2I.keys())
    hid_dim = in_dim // 2

    # Create a log linear classifier with the specified input and output dimensions,
    # represented as the parameters W and b.
    params = mlp.create_classifier(in_dim, hid_dim, out_dim)

    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]
    I2L = {v: k for k, v in L2I.items()}
    predictions = [I2L[mlp.predict(feats_to_vec(feature), trained_params)] for label, feature in TEST]
    with open("test.pred", "w") as f:
        f.writelines(line + '\n' for line in predictions)
        
    

