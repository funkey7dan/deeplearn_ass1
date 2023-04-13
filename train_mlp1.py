import mlp1 as mlp
import utils
from utils import *
import numpy as np
from xor_data import data as xor_data


STUDENT1={'name': 'Coral Kuta',
         'ID': 'CORAL_ID'}
STUDENT2={'name': 'Daniel Bronfman ',
         'ID': 'DANIEL_ID '}

# NOTE - in order to run XOR, turn to True
isXor = False

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
    vec = np.zeros(len(vocab))
    for f in features:
        if f in F2I:
            vec[F2I[f]] = 1
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if len(params[-1]) == 2:  # we use xor
            x = features
            y = label
        else:
            x = feats_to_vec(features)
            y = utils.L2I[label]
        prediction = mlp.predict(x, params)

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
            if isXor:
                x = features
                y = label
            else:
                x = feats_to_vec(features)  # convert features to a vector.
                y = utils.L2I[label]  # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)

        if dev_data:
            dev_accuracy = accuracy_on_dataset(dev_data, params)
            print(f'iteration {round(I,2)}: train_loss={round(train_loss,2)},'
                  f' train_accuracy={round(train_accuracy,2)}, dev_accuracy={round(dev_accuracy,2)}')

        else:
            #pass
            print(f'iteration {round(I, 2)}: train_loss={round(train_loss,2)},'
                  f' train_accuracy={round(train_accuracy,2)}')

        if isXor:
            t1 = (mlp.predict(np.array([0,0]), params)==1)
            t2 = (mlp.predict(np.array([1,0]), params)==0)
            t3 = (mlp.predict(np.array([0,1]), params)==0)
            t4 = (mlp.predict(np.array([1,1]), params)==1)
            if t1 and t2 and t3 and t4:
                print(f'Trained XOR in num_iterations={I}, learning_rate={learning_rate}')
                break
    return params

def find_best_hyperparams(model, train_data, dev_data, in_dim, out_dim):
    hyper_params = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                num_iterations = i * 10
                learning_rate = j * 0.01
                hid_dim = k * 10
                params = mlp.create_classifier(in_dim, hid_dim, out_dim)
                trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
                print(f'num_iterations={num_iterations}, learning_rate={learning_rate},hid_dim={hid_dim}, dev_accuracy={round(accuracy_on_dataset(dev_data, trained_params),2)}')
                hyper_params.append((num_iterations, learning_rate, hid_dim,accuracy_on_dataset(dev_data, trained_params)))
                hyper_params.sort(key=lambda x: x[2], reverse=True)
    print(hyper_params)
    

if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    if isXor:
        train_data = xor_data
        num_iterations = 200
        learning_rate = 1
        in_dim = 2
        out_dim = 2
        hid_dim = 2
        params = mlp.create_classifier(in_dim, hid_dim, out_dim)
        trained_params = train_classifier(train_data, None, num_iterations, learning_rate, params)



    else:
        # Initialize the training arguments
        train_data = TRAIN
        dev_data = DEV
        num_iterations = 8
        learning_rate = 0.005

        # the number of features we input
        in_dim = len(F2I.keys())
        # the number of labels, in our case the different languages we have.
        out_dim = len(L2I.keys())
        hid_dim = (in_dim+out_dim) // 2
        #find_best_hyperparams(mlp, train_data, dev_data, in_dim, out_dim)
        #exit()
        # Create a log linear classifier with the specified input and output dimensions,
        # represented as the parameters W and b.
        params = mlp.create_classifier(in_dim, hid_dim, out_dim,isXor)

        trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
        
        # Find best hyper-parameters:
        

        TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]
        I2L = {v: k for k, v in L2I.items()}
        predictions = [I2L[mlp.predict(feats_to_vec(feature), trained_params)] for label, feature in TEST]
        with open("test.pred", "w") as f:
            f.writelines(line + '\n' for line in predictions)
        
    

